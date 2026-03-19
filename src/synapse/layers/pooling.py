import jax
from flax import nnx
from jax import numpy as jnp
from jax.typing import ArrayLike


class AttentionPool(nnx.Module):
    """Learnable attention pooling: (B, seq_len, in_features) -> (B, out_features).

    Uses a single learned query vector to compute attention weights over
    the sequence, then linearly projects the weighted sum.
    """

    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.query = nnx.Param(
            nnx.initializers.normal(0.02)(rngs.params(), (1, 1, in_features))
        )
        self.proj = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: ArrayLike) -> jax.Array:
        """Pool a sequence via learned attention weights.

        Args:
            x: (B, seq_len, in_features)

        Returns:
            (B, out_features)
        """
        attn = jnp.sum(x * self.query, axis=-1) / jnp.sqrt(x.shape[-1])
        attn = nnx.softmax(attn, axis=-1)
        pooled = jnp.sum(x * attn[:, :, None], axis=1)
        return self.proj(pooled)
