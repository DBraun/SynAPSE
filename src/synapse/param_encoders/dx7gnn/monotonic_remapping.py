"""Monotonic remapping function for edge and carrier weights."""

import jax
from flax import nnx
from jax import numpy as jnp
from jax.typing import ArrayLike


class MonotonicRemapping(nnx.Module):
    """Learned monotonic remapping function that maps [0,1] to [0,1].

    Uses a piecewise linear function with learnable positive slopes between fixed knots.
    Initialized as the identity function.
    """

    def __init__(
        self,
        num_knots: int = 5,
    ):
        """Initialize the monotonic remapping module.

        Args:
            num_knots: Number of knot points for piecewise linear function.
        """
        self.num_knots = num_knots

        # Initialize slopes to 1.0 (identity function)
        # We have num_knots - 1 segments between knots
        # Store log slopes to ensure positivity via exp
        self.log_slopes = nnx.Param(jnp.zeros(num_knots - 1))

    def __call__(self, x: ArrayLike) -> jax.Array:
        """Apply monotonic remapping to input values.

        Args:
            x: Input values in [0, 1] of any shape.

        Returns:
            Remapped values in [0, 1] with same shape as input.
        """
        # Fixed knot positions evenly spaced in [0, 1]
        knot_positions = jnp.linspace(0.0, 1.0, self.num_knots)

        # Ensure slopes are positive
        slopes = jnp.exp(self.log_slopes[...])

        # Compute knot values (y-coordinates) from slopes
        # Start at 0, accumulate scaled slopes
        segment_heights = slopes / jnp.sum(slopes)  # Normalize to sum to 1
        knot_values = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(segment_heights)])

        # Clip input to [0, 1] for safety
        x_clipped = jnp.clip(x, 0.0, 1.0)

        # Piecewise linear interpolation using jnp.interp
        return jnp.interp(x_clipped, knot_positions, knot_values)
