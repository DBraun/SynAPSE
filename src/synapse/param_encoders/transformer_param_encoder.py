"""Transformer-based encoder for DX7 synthesizer parameters.

Treats each DX7 operator as a token (6 tokens) and processes them with standard
transformer self-attention. Global features and algorithm identity are injected
at every layer via cross-attention to a single context token, analogous to the
GNN's per-layer FiLM conditioning.

Uses the same input pipeline as the GNN: extract_node_features() (49-dim: 22
global broadcast + 27 per-op) → MLP, ensuring a fair architectural comparison
where the only difference is graph structure (GNN) vs. sequence structure (Transformer).
"""

import argbind
import jax
from audiotree import AudioTree
from flax import nnx
from flax.nnx import Dropout, LayerNorm, Linear, Module, MultiHeadAttention, Param, Rngs
from jax import numpy as jnp
from jax.typing import ArrayLike

from synapse.activations import parse_activation
from synapse.slap import MLP

from .dx7_features import (
    GLOBAL_DIM,
    NODE_INPUT_DIM,
    extract_global_features,
    extract_node_features,
)


class TransformerLayer(Module):
    """Single pre-norm transformer encoder layer with optional cross-attention.

    Architecture: LN -> MHSA -> residual -> [LN -> Cross-Attn -> residual] -> LN -> FFN -> residual
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout_rate: float,
        activation: callable,
        rngs: Rngs,
    ):
        # Self-attention
        self.norm1 = LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout1 = Dropout(dropout_rate, rngs=rngs)

        # Cross-attention to global context
        self.norm_cross = LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_dim,
            decode=False,
            rngs=rngs,
        )
        self.dropout_cross = Dropout(dropout_rate, rngs=rngs)

        # FFN
        self.norm2 = LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        ffn_dim = int(hidden_dim * mlp_ratio)
        self.ffn_linear1 = Linear(hidden_dim, ffn_dim, rngs=rngs)
        self.ffn_linear2 = Linear(ffn_dim, hidden_dim, rngs=rngs)
        self.dropout2 = Dropout(dropout_rate, rngs=rngs)
        self.activation = activation

    def __call__(self, x: ArrayLike, context: ArrayLike = None) -> jax.Array:
        """Forward pass.

        Args:
            x: Input tokens [batch_size, seq_len, hidden_dim].
            context: Global context for cross-attention [batch_size, 1, hidden_dim].

        Returns:
            Output tokens [batch_size, seq_len, hidden_dim].
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout1(x)
        x = residual + x

        # Cross-attention to global context
        if context is not None:
            residual = x
            x = self.norm_cross(x)
            x = self.cross_attn(x, context)  # Q=x, K/V=context
            x = self.dropout_cross(x)
            x = residual + x

        # FFN with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn_linear1(x)
        x = self.activation(x)
        x = self.ffn_linear2(x)
        x = self.dropout2(x)
        x = residual + x

        return x


@argbind.bind()
class TransformerParamEncoder(Module):
    """Transformer encoder for DX7 synthesizer parameters.

    Tokenizes DX7 presets as 6 operator tokens (same MLP as GNN), processes
    with transformer self-attention, and injects global features + algorithm
    identity at every layer via cross-attention to a context token.
    """

    def __init__(
        self,
        hidden_dim: int = 896,
        out_features: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout_rate: float = 0.2,
        pooling: str = "cls",
        activation: str = "gelu_tanh",
        input_ratios: list[float] = None,
        value_mask_rate: float = 0.0,
        algo_dropout_rate: float = 0.1,
        rngs: Rngs = None,
    ):
        """Initialize the Transformer parameter encoder.

        Args:
            hidden_dim: Hidden dimension for transformer layers.
            out_features: Output embedding dimension.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            mlp_ratio: FFN hidden dim = hidden_dim * mlp_ratio.
            dropout_rate: Dropout rate in attention and FFN.
            pooling: Pooling strategy ("cls" or "mean").
            activation: Activation name for parse_activation (default "gelu_tanh").
            input_ratios: Multipliers for hidden_dim in MLP intermediate layers
                (default [2], matching GNN). E.g., [2] creates [49, 2*H, H].
            value_mask_rate: Probability of replacing each operator token with a learned
                [MASK] embedding during training (0.0 = no masking).
            algo_dropout_rate: Probability of replacing the algorithm embedding with a
                learned "unknown" embedding during training (0.0 = no dropout). Enables
                evaluation on held out algorithms by training robustness to missing
                algorithm identity.
            rngs: Random number generators.
        """
        if rngs is None:
            rngs = Rngs(0)

        self.output_dim = out_features
        self.num_layers = num_layers
        self.pooling = pooling
        self.value_mask_rate = value_mask_rate
        self.deterministic = False

        activation_fn = parse_activation(activation)

        # Input projection: same pipeline as GNN (extract_node_features → MLP)
        if input_ratios is None:
            input_ratios = [2, 1]
        input_dims = (
            [NODE_INPUT_DIM]
            + [int(hidden_dim * r) for r in input_ratios]
            + [hidden_dim]
        )
        self.input_module = MLP(
            input_dims,
            activation=activation_fn,
            normalization="layer",
            last_layer=None,
            dropout_rate=0.0,
            bias=None,
            out_bias=False,
            rngs=rngs,
        )

        # Algorithm embedding: 32 real algorithms + 1 "unknown" slot (index 32)
        self.algorithm_embedding = nnx.Embed(33, hidden_dim, rngs=rngs)
        self.algo_dropout_rate = algo_dropout_rate
        self.force_unknown_algorithm = False
        self.unknown_algo_index = 32
        if algo_dropout_rate > 0:
            self.algo_dropout_rngs = Rngs(rngs.params())

        # Context projection: global features (22) + algo embedding (H) → H
        self.context_projection = Linear(GLOBAL_DIM + hidden_dim, hidden_dim, rngs=rngs)

        # CLS token (only if using CLS pooling)
        num_tokens = 6  # 6 operators (no global token)
        if pooling == "cls":
            self.cls_token = Param(jnp.zeros((1, 1, hidden_dim)))
            num_tokens += 1
        else:
            self.cls_token = None

        # Learnable positional encoding
        self.pos_embedding = Param(jnp.zeros((1, num_tokens, hidden_dim)))

        # Independent weights per layer, stacked via vmap for scan
        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_layer(rngs: Rngs):
            return TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout_rate=dropout_rate,
                activation=activation_fn,
                rngs=rngs,
            )

        self.layers = create_layer(rngs)

        # Output head: LayerNorm -> Linear
        self.output_norm = LayerNorm(hidden_dim, epsilon=1e-5, rngs=rngs)
        self.output_projection = Linear(hidden_dim, out_features, rngs=rngs)

        # Learned mask embedding for operator-level masking during training
        if self.value_mask_rate > 0:
            self.mask_embedding = Param(rngs.params.normal((hidden_dim,)) * 0.02)
            self.mask_rngs = Rngs(rngs.params())

    def __call__(self, audio_tree: AudioTree) -> jax.Array:
        """Forward pass through the Transformer encoder.

        Args:
            audio_tree: AudioTree with extras containing:
                - params: [batch_size, 145] flat parameters from Preset.to_array()
                - algorithm: [batch_size,] algorithm indices (0-31)

        Returns:
            Embeddings of shape [batch_size, out_features].
        """
        flat_params = audio_tree.extras["params"]  # [B, 145]
        algorithm_indices = audio_tree.extras["algorithm"]  # [B,]

        B = flat_params.shape[0]

        # Feature extraction (matching GNN pipeline)
        node_features = extract_node_features(flat_params)  # [B, 6, 49]
        op_tokens = self.input_module(node_features)  # [B, 6, H]

        # Operator-level masking: replace random operator tokens with learned [MASK]
        if self.value_mask_rate > 0 and not self.deterministic:
            mask = self.mask_rngs.bernoulli(self.value_mask_rate, shape=(B, 6, 1))
            mask_emb = jnp.broadcast_to(
                self.mask_embedding[...][None, None, :], op_tokens.shape
            )
            op_tokens = jnp.where(mask, mask_emb, op_tokens)

        # Global features for cross-attention context
        global_features = extract_global_features(flat_params)  # [B, 22]

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

        algo_emb = self.algorithm_embedding(algorithm_indices)  # [B, H]

        # Build cross-attention context: global features + algorithm embedding
        context_input = jnp.concatenate(
            [global_features, algo_emb], axis=-1
        )  # [B, 22+H]
        context = self.context_projection(context_input)[:, None, :]  # [B, 1, H]

        # Sequence: [op1, ..., op6] (6 tokens)
        tokens = op_tokens  # [B, 6, H]

        # Optionally prepend CLS token
        if self.cls_token is not None:
            cls = jnp.broadcast_to(self.cls_token[...], (B, 1, tokens.shape[-1]))
            tokens = jnp.concatenate([cls, tokens], axis=1)  # [B, 7, H]

        # Add positional encoding
        tokens = tokens + self.pos_embedding[...]

        # Transformer encoder (independent weights, scanned)
        @nnx.scan(length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def scan_fn(x, layer):
            x = layer(x, context=context)
            return x

        tokens = scan_fn(tokens, self.layers)

        # Pool
        if self.pooling == "cls":
            output = tokens[:, 0]  # [B, H]
        else:
            output = tokens.mean(axis=1)  # [B, H]

        # Output projection
        output = self.output_norm(output)
        output = self.output_projection(output)  # [B, out_features]

        return output


if __name__ == "__main__":
    batch_size = 8
    audio_tree = AudioTree(
        jnp.zeros((batch_size, 1, 44100)),
        sample_rate=44100,
        extras={
            "params": jnp.ones((batch_size, 145)) * 0.5,
            "algorithm": jnp.array([0, 1, 2, 3, 4, 5, 6, 7]),
        },
    )

    # 1. Standard mode (no dropout)
    enc = TransformerParamEncoder(rngs=Rngs(42))
    print(nnx.tabulate(enc, audio_tree, depth=3, console_kwargs={"width": 200}))

    # 2. Algo dropout during training
    enc_drop = TransformerParamEncoder(algo_dropout_rate=0.3, rngs=Rngs(42))
    out1 = enc_drop(audio_tree)
    out2 = enc_drop(audio_tree)
    print(f"Algo dropout outputs differ (stochastic): {not jnp.allclose(out1, out2)}")

    # 3. Force unknown algorithm (eval mode)
    enc_drop.eval(force_unknown_algorithm=True)
    out_a = enc_drop(audio_tree)
    audio_tree2 = AudioTree(
        audio_tree.waveform,
        sample_rate=audio_tree.sample_rate,
        extras={
            "params": audio_tree.extras["params"],
            "algorithm": jnp.array([31, 30, 29, 28, 27, 26, 25, 24]),
        },
    )
    out_b = enc_drop(audio_tree2)
    print(
        f"Force unknown: same output for different algos: {jnp.allclose(out_a, out_b)}"
    )
