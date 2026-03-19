"""Flat MLP baseline encoder for DX7 synthesizer parameters.

Uses extract_node_features() for proper one-hot encoding of categorical
parameters, then flattens to a 294-dim vector (6 operators x 49 features).
Concatenates with a learned algorithm embedding and processes through an
MLP. Supports residual and highway block types.

The embedding table has 33 entries: indices 0-31 for the 32 DX7 algorithms,
and index 32 for "unknown". Algorithm dropout randomly replaces the real
index with 32 during training. At test time on held out algorithms,
force_unknown_algorithm=True sets all indices to 32.
"""

import argbind
import jax
from audiotree import AudioTree
from flax import nnx
from jax import numpy as jnp

from synapse.activations import parse_activation

from .dx7_features import (
    NODE_INPUT_DIM,
    extract_node_features,
)


@argbind.bind()
class FlatParamEncoder(nnx.Module):
    """Flat MLP encoder for synthesizer parameters.

    Extracts one-hot encoded node features from raw DX7 parameters,
    flattens them, concatenates with a learned algorithm embedding,
    and processes through an MLP with residual or highway blocks.
    """

    def __init__(
        self,
        in_features: int = 6 * NODE_INPUT_DIM,
        hidden_dims: list[int] = (768, 1024, 1024, 768, 512),
        algo_embed_dim: int = 64,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = "gelu_tanh",
        block_type: str = "residual",
        algo_dropout_rate: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        """Initialize the flat parameter encoder.

        Args:
            in_features: Dimension of flattened node features (6 * 49 = 294 for DX7).
            hidden_dims: Hidden layer dimensions (last dim is output dim).
            algo_embed_dim: Dimension of learned algorithm embedding.
            dropout_rate: Dropout rate in blocks.
            use_layer_norm: Use LayerNorm (True) or BatchNorm (False).
            activation: Activation name for parse_activation (default "gelu_tanh").
            block_type: "residual" for residual MLP blocks, "highway" for
                highway network blocks with learned gating.
            algo_dropout_rate: Probability of replacing algorithm embedding with
                learned "unknown" token during training (0.0 = no dropout).
            rngs: Random number generators.
        """
        if not hidden_dims:
            raise ValueError("hidden_dims must have at least one dimension")
        if block_type not in ("residual", "highway"):
            raise ValueError(
                f"block_type must be 'residual' or 'highway', got '{block_type}'"
            )

        self.output_dim = hidden_dims[-1]
        self.use_layer_norm = bool(use_layer_norm)
        self.block_type = block_type
        self.algo_dropout_rate = algo_dropout_rate
        self.force_unknown_algorithm = False
        self.deterministic = False
        self.unknown_algo_index = 32

        # Activation function
        self.activation = parse_activation(activation)

        # Learned algorithm embedding: 32 real algorithms + 1 "unknown" slot (index 32)
        self.algorithm_embedding = nnx.Embed(33, algo_embed_dim, rngs=rngs)

        if algo_dropout_rate > 0:
            self.algo_dropout_rngs = nnx.Rngs(rngs.params())

        # Input dim: flattened node features + algorithm embedding
        input_dim = in_features + algo_embed_dim

        # Input normalization
        self.input_norm = nnx.BatchNorm(input_dim, momentum=0.9, rngs=rngs)

        # Build encoder blocks
        blocks = []
        gates = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            block_layers = []
            block_layers.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            if self.use_layer_norm:
                block_layers.append(nnx.LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs))
            else:
                block_layers.append(nnx.BatchNorm(hidden_dim, momentum=0.9, rngs=rngs))
            block_layers.append(self.activation)
            block_layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
            blocks.append(nnx.List(block_layers))

            # Gate / projection for skip connections
            if block_type == "highway" and prev_dim == hidden_dim:
                # Highway gate: bias init to -1 biases toward identity at init
                # (Srivastava et al. 2015, synth-proxy)
                gates.append(
                    nnx.Linear(
                        prev_dim,
                        hidden_dim,
                        bias_init=nnx.initializers.constant(-1.0),
                        rngs=rngs,
                    )
                )
            elif block_type == "residual" and prev_dim != hidden_dim:
                # Residual projection to match dimensions
                gates.append(nnx.Linear(prev_dim, hidden_dim, rngs=rngs))
            else:
                gates.append(None)

            prev_dim = hidden_dim

        self.blocks = nnx.List(blocks)
        self.gates = nnx.List(gates)

        # Final linear projection to output dim (no norm/activation/dropout)
        self.final_linear = nnx.Linear(prev_dim, self.output_dim, rngs=rngs)

    def __call__(self, audio_tree: AudioTree) -> jax.Array:
        """Forward pass.

        Args:
            audio_tree: AudioTree with extras containing:
                - params: [batch_size, 145] raw DX7 parameters in [0, 1]
                - algorithm: [batch_size,] algorithm indices (0-31)

        Returns:
            Embeddings of shape [batch_size, output_dim].
        """
        flat_params = audio_tree.extras["params"]  # [B, 145]
        algorithm_indices = audio_tree.extras["algorithm"]
        B = flat_params.shape[0]

        # Extract one-hot encoded node features and flatten
        node_features = extract_node_features(
            flat_params
        )  # [B, 6, 49], already in [-1, 1]
        x = node_features.reshape(B, -1)  # [B, 294]

        # Algorithm index: replace with "unknown" index (32) during dropout or forced mode
        if self.force_unknown_algorithm:
            algorithm_indices = jnp.full_like(
                algorithm_indices, self.unknown_algo_index
            )
        elif self.algo_dropout_rate > 0 and not self.deterministic:
            mask = self.algo_dropout_rngs.bernoulli(self.algo_dropout_rate, shape=(B,))
            algorithm_indices = jnp.where(
                mask, self.unknown_algo_index, algorithm_indices
            )

        algo_emb = self.algorithm_embedding(algorithm_indices)  # [B, algo_embed_dim]

        # Concatenate features + algorithm embedding
        x = jnp.concatenate([x, algo_emb], axis=-1)

        # Input normalization
        x = self.input_norm(x)

        # MLP blocks
        if self.block_type == "highway":
            for block, gate in zip(self.blocks, self.gates):
                fc_out = x
                for layer in block:
                    fc_out = layer(fc_out)
                if gate is not None:
                    gate_val = nnx.sigmoid(gate(x))
                    x = fc_out * gate_val + x * (1 - gate_val)
                else:
                    x = fc_out
        else:
            # Residual blocks
            for block, projection in zip(self.blocks, self.gates):
                residual = x
                for layer in block:
                    x = layer(x)
                if projection is not None:
                    residual = projection(residual)
                x = x + residual

        # Final projection
        x = self.final_linear(x)
        return x


if __name__ == "__main__":
    batch_size = 4

    for btype in ("residual", "highway"):
        print(f"\n{'='*60}")
        print(f"block_type = {btype}")
        print(f"{'='*60}")

        encoder = FlatParamEncoder(
            hidden_dims=(768, 768, 768, 768, 512),
            block_type=btype,
            rngs=nnx.Rngs(0),
        )

        at = AudioTree(
            jnp.zeros((batch_size, 1, 1)),
            sample_rate=44100,
            extras={
                "algorithm": jnp.full((batch_size,), 2),
                "params": jnp.ones((batch_size, 145)),
            },
        )

        print(nnx.tabulate(encoder, at, depth=3, console_kwargs={"width": 250}))
