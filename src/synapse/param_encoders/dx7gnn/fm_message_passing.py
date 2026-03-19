"""FM Synthesis-inspired Message Passing with Output Level Gating.

Message passing layer for DX7 GNN:
- Messages: gather source features (OL-gated from previous layer), scale by edge weight
- Aggregation: segment_sum
- Update: configurable via update_fn ("film", "concat", "add", "attn")
- Normalize: out_layer (layer_norm, batch_norm, tanh, sigmoid, none)
- Gate: multiply by output_level (OL=0 -> zero output regardless of other params)

The two multiplications (edge_weight scaling in message step, OL gating at end) are
adjacent across layers. Step 5 of layer L gates by OL; step 1 of layer L+1 gathers
those OL-gated states. This ensures operators with OL=0 contribute nothing.
"""

import argbind
import jax
from flax import nnx
from flax.nnx import (
    BatchNorm,
    LayerNorm,
    Linear,
    Module,
    Rngs,
)
from jax import numpy as jnp
from jax.ops import segment_sum
from jax.typing import ArrayLike

from synapse.slap import MLP

from .mlp_film import MLP_FiLM


@argbind.bind()
class FMMessagePassing(Module):
    """FM message passing layer with output level gating.

    Core inductive bias: operators with output_level=0 produce zero signal,
    regardless of their other parameters. This is enforced architecturally
    by multiplying node states by their output level after each layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 384,
        num_mlp_layers: int = 2,
        update_fn: str = "film",
        num_attn_heads: int = 8,
        out_layer: str = "layer_norm",
        use_residual: bool = False,
        rngs: Rngs = None,
    ):
        """Initialize FM message passing layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            hidden_dim: Hidden dimension for update MLP.
            num_mlp_layers: Number of layers in update MLP.
            update_fn: Update function type ("film", "concat", "add", "attn").
            num_attn_heads: Number of attention heads (only used when update_fn="attn").
            out_layer: Output normalization ("layer_norm", "batch_norm", "tanh", "sigmoid", "none").
            use_residual: Add residual connection x_new = update(x) + x.
            rngs: Random number generators.
        """
        if use_residual and in_features != out_features:
            raise ValueError(
                f"use_residual requires in_features == out_features, "
                f"got {in_features} != {out_features}"
            )
        self.out_features = out_features
        self.update_fn = update_fn
        self.use_residual = use_residual

        # Build update module based on update_fn
        if update_fn == "film":
            self.update_module = MLP_FiLM(
                in_features=in_features,
                features_list=(hidden_dim,) * num_mlp_layers + (out_features,),
                rngs=rngs,
            )
        elif update_fn == "concat":
            dims = [in_features * 2] + [hidden_dim] * num_mlp_layers + [out_features]
            self.update_module = MLP(
                dims=dims,
                activation=True,
                normalization="layer",
                dropout_rate=0.0,
                rngs=rngs,
            )
        elif update_fn == "add":
            dims = [in_features] + [hidden_dim] * num_mlp_layers + [out_features]
            self.update_module = MLP(
                dims=dims,
                activation=True,
                normalization="layer",
                dropout_rate=0.0,
                rngs=rngs,
            )
        elif update_fn == "attn":
            self.update_attn = nnx.MultiHeadAttention(
                num_heads=num_attn_heads,
                in_features=in_features,
                decode=False,
                rngs=rngs,
            )
            self.update_proj = Linear(in_features, out_features, rngs=rngs)
        else:
            raise ValueError(f"Unknown update_fn: {update_fn}")

        # Output normalization
        if out_layer == "layer_norm":
            self.out_layer = LayerNorm(out_features, epsilon=1e-5, rngs=rngs)
        elif out_layer == "batch_norm":
            self.out_layer = BatchNorm(
                out_features, use_bias=True, momentum=0.9, rngs=rngs
            )
        elif out_layer == "tanh":
            self.out_layer = nnx.tanh
        elif out_layer == "sigmoid":
            self.out_layer = nnx.sigmoid
        elif out_layer is None or out_layer == "none":
            self.out_layer = lambda x: x
        else:
            raise ValueError(f"Unknown out_layer: {out_layer}")

    def __call__(
        self,
        x: ArrayLike,
        edge_index: ArrayLike,
        edge_weight: ArrayLike | None = None,
        original_features: ArrayLike | None = None,
        output_level: ArrayLike | None = None,
    ) -> jax.Array:
        """Forward pass through message passing layer.

        Args:
            x: Current node states [N, D] (OL-gated from previous layer).
            edge_index: Edge connections [2, E] (src, dst).
            edge_weight: Edge weights [E] (1.0 for modulation, feedback_param*0.5 for feedback).
            original_features: Original node features [N, D] (not OL-gated).
            output_level: Per-node output levels [N] in [0, 1].

        Returns:
            Updated node states [N, out_features], gated by output_level.
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        # 1. Message: gather source features (already OL-gated from previous layer).
        #    Edge weights are 1.0 for modulation edges and feedback_param*0.5 for
        #    feedback (self-loop) edges, so this attenuates feedback connections
        #    while passing modulation signals through unchanged.
        messages = x[src_idx]
        if edge_weight is not None:
            messages = messages * edge_weight[:, None]

        # 2. Aggregate: segment sum
        aggr = segment_sum(messages, dst_idx, num_segments=num_nodes)

        # 3. Update: configurable update function
        if self.update_fn == "film":
            updated = self.update_module(original_features, aggr)
        elif self.update_fn == "concat":
            updated = self.update_module(
                jnp.concatenate([original_features, aggr], axis=-1)
            )
        elif self.update_fn == "add":
            updated = self.update_module(original_features + aggr)
        elif self.update_fn == "attn":
            # Stack as 2 tokens [N, 2, D], self-attend, take first token
            tokens = jnp.stack([original_features, aggr], axis=-2)
            attended = self.update_attn(tokens)
            updated = self.update_proj(attended[:, 0])

        # 4. Normalize
        updated = self.out_layer(updated)

        # 5. Residual from base features (before OL gating)
        if self.use_residual:
            updated = updated + original_features

        # 6. Gate by output level (OL=0 -> output=0)
        if output_level is not None:
            updated = updated * output_level[:, None]

        return updated
