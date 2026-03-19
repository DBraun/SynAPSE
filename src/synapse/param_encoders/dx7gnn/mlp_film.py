"""Feature-wise Linear Modulation (FiLM) MLP."""

import argbind
from flax import nnx
from flax.nnx import Dropout, Linear, Module, Rngs
from jax.typing import ArrayLike

from synapse.activations import parse_activation


@argbind.bind()
class MLP_FiLM(Module):
    """For primary input x and secondary input y, x through a stack of linear layers.
    After each layer, modulate with scale and bias according to y (FiLM)."""

    def __init__(
        self,
        in_features: int,
        features_list: list[int],
        dropout_rate: float = 0.0,
        activation: str = "relu",
        rngs: Rngs = None,
    ):
        assert features_list is not None

        self.features_list = features_list
        self.activation = parse_activation(activation)
        self.dropout = Dropout(rate=dropout_rate, rngs=rngs)

        # Create main transformation layers
        layers = []
        current_features = in_features
        for i, out_features in enumerate(features_list):
            layer = Linear(current_features, out_features, rngs=rngs)
            layers.append(layer)
            current_features = out_features
        self.layers = nnx.List(layers)

        # Create FiLM parameter generators (scale and bias)
        film_layers = []
        for out_features in features_list:
            # Small init so scale≈0, bias≈0 at start: FiLM acts as near-identity
            # ((1+0)*x + 0 = x), letting the base transform train before modulation kicks in
            kernel_init = nnx.initializers.variance_scaling(
                1 / 10, mode="fan_in", distribution="truncated_normal"
            )
            film_layer = Linear(
                in_features, out_features * 2, kernel_init=kernel_init, rngs=rngs
            )
            film_layers.append(film_layer)
        self.film_layers = nnx.List(film_layers)

    def __call__(self, x: ArrayLike, y: ArrayLike):
        """Apply MLP with FiLM modulation.

        Args:
            x: Primary input tensor
            y: Secondary input tensor for FiLM modulation

        Returns:
            Modulated output tensor
        """
        current = x

        for i, (layer, film_layer) in enumerate(zip(self.layers, self.film_layers)):
            # Apply linear transformation
            current = layer(current)

            # Generate FiLM parameters from y
            film_params = film_layer(y)  # [batch, out_features * 2]
            out_features = self.features_list[i]

            # Split into scale and bias
            scale = film_params[..., :out_features]
            bias = film_params[..., out_features:]

            # Apply FiLM modulation: x * (1 + scale) + bias
            current = current * (1 + scale) + bias

            # Apply dropout and activation (except for last layer)
            if i < len(self.layers) - 1:
                current = self.dropout(current)
                current = self.activation(current)

        return current
