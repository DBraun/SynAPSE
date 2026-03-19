"""SLAP (Siamese Language-Audio Pretraining) model in JAX/NNX.

BYOL-style non-contrastive multimodal representation learning: two encoders
(here audio and DX7 parameters) are each wrapped in a Siamese arm
(encoder → projector → predictor), and trained without negative samples. Also
defines the generic ``MLP`` used throughout the project's projection heads.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import argbind
import jax
from flax import nnx
from flax.nnx import BatchNorm, Dropout, LayerNorm, Linear, Module, Rngs, Sequential
from flax.struct import dataclass as flax_dataclass
from jax import numpy as jnp
from jax.tree_util import tree_map
from jax.typing import ArrayLike

from .base_module import SaveLoadModule


@flax_dataclass
class SiameseOutput:
    latent: jax.Array  # y
    projection: jax.Array  # z
    prediction: jax.Array  # q


def _normalize(x: ArrayLike) -> jax.Array:
    """Apply L2 normalization along the last axis.

    Args:
        x: Input tensor.

    Returns:
        L2-normalized tensor.
    """
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class MLP(Module):
    """Generic and customizable multi-layer perceptron.

    This module creates a flexible MLP with customizable activation functions,
    normalization layers, and dropout.
    """

    def __init__(
        self,
        dims: list[int],
        activation: bool | Callable = True,
        normalization: bool | str | Callable = True,
        batchnorm_momentum: float = 0.9,
        last_layer: Module | None = None,
        normalize_first: int = -1,
        dropout_rate: float = 0.0,
        bias: bool | None = None,
        out_bias: bool = True,
        rngs: Rngs = None,
    ):
        """Initialize the MLP.

        Args:
            dims: List of dimensions for each layer. For a d-deep MLP,
                d+1 dimensions should be provided (input + one per layer).
            activation: Activation function between layers. True for ReLU,
                False for no activation, or a callable activation function.
            normalization: Normalization between layers. True for BatchNorm,
                False for no normalization, "layer" for LayerNorm.
            batchnorm_momentum: Momentum for batch norm.
            last_layer: Optional final layer (e.g., sigmoid, softmax).
            normalize_first: If >= 0, applies L2 normalization after the layer at
                this index.
            dropout_rate: Dropout probability (0 for no dropout).
            bias: Whether linear layers should have bias. By default, adds bias
                when there's no normalization.
            out_bias: Whether the last layer has bias.
            rngs: Random number generators.
        """
        self.in_features = dims[0]
        self.out_features = dims[-1]

        # Define activation
        if activation is True:
            activation_fn = nnx.relu
        elif activation is False:
            activation_fn = None
        else:
            activation_fn = activation

        # Define normalization
        if normalization is True:
            norm_class = BatchNorm
        elif normalization == "layer":
            norm_class = LayerNorm
        elif normalization is False:
            norm_class = None
        else:
            norm_class = normalization

        # Determine bias (no bias before batch norm by default)
        if bias is None:
            bias = norm_class is None

        layers = []

        # Build intermediate layers
        for i in range(len(dims) - 2):
            in_dim, out_dim = dims[i], dims[i + 1]

            # Linear layer
            layers.append(Linear(in_dim, out_dim, use_bias=bias, rngs=rngs))

            # Check for L2 normalization at specific index
            if i == normalize_first:
                layers.append(_normalize)
            else:
                # Normalization layer
                if norm_class is not None:
                    if norm_class == BatchNorm:
                        layers.append(
                            norm_class(out_dim, momentum=batchnorm_momentum, rngs=rngs)
                        )
                    elif norm_class == LayerNorm:
                        layers.append(norm_class(out_dim, epsilon=1e-5, rngs=rngs))
                    else:
                        layers.append(norm_class(out_dim, rngs=rngs))

                # Activation
                if activation_fn is not None:
                    if callable(activation_fn):
                        layers.append(activation_fn)
                    else:
                        layers.append(lambda x: activation_fn(x))

                # Dropout
                layers.append(Dropout(dropout_rate, rngs=rngs))

        # Add final linear layer
        if len(dims) >= 2:
            layers.append(Linear(dims[-2], dims[-1], use_bias=out_bias, rngs=rngs))

        # Add optional final layer
        if last_layer is not None:
            layers.append(last_layer)

        # Create sequential model
        self.sequential = Sequential(*layers)

    def __call__(self, x: ArrayLike) -> jax.Array:
        return self.sequential(x)


class SiameseArm(Module):
    """Generic encoder wrapper with projection and predictor heads.

    Wraps any encoder and adds projection and prediction heads following the
    SLAP architecture.
    """

    def __init__(
        self,
        encoder: Module,
        embed_dim: int,
        pred_hidden_dim: int,
        normalize_y: bool = False,
        normalize_z_q: bool = False,
        projector_dropout: float = 0.0,
        predictor_dropout: float = 0.1,
        predictor_batchnorm_momentum: float = 0.9,
        projector: Module | None = None,
        predictor: Module | None = None,
        rngs: Rngs = None,
    ):
        """Initialize a Siamese arm.

        Args:
            encoder: The base encoder module.
            embed_dim: Dimension of the projection space.
            pred_hidden_dim: Hidden dimension of the predictor MLP.
            normalize_y: Whether to normalize representations (y).
            normalize_z_q: Whether to L2-normalize projections (z) and predictions (q).
            projector_dropout: Dropout rate for projector MLP.
            predictor_dropout: Dropout rate for predictor MLP.
            predictor_batchnorm_momentum: BatchNorm momentum in predictor's MLP layers.
            projector: Optional custom projector module (if None, creates default MLP).
            predictor: Optional custom predictor module (if None, creates default MLP).
            rngs: Random number generators.
        """
        self.encoder = encoder
        self.normalize_y = normalize_y
        self.normalize_z_q = normalize_z_q

        # Get encoder output dimension
        try:
            encoder_dim = encoder.output_dim
        except AttributeError:
            encoder_dim = encoder.out_features

        # Build projector: maps encoder output to shared projection space.
        if projector is not None:
            self.projector = projector
        else:
            # SLAP configs disable projector normalization.
            normalization = False

            self.projector = MLP(
                dims=[encoder_dim, embed_dim, embed_dim],
                rngs=rngs,
                activation=True,  # ReLU
                normalization=normalization,
                dropout_rate=projector_dropout,
                last_layer=None,
            )

        if predictor is not None:
            self.predictor = predictor
        else:
            # SLAP section 4.2 uses BatchNorm in the predictor.
            normalization = True
            self.predictor = MLP(
                dims=[embed_dim, pred_hidden_dim, embed_dim],
                rngs=rngs,
                activation=True,  # ReLU
                normalization=normalization,
                batchnorm_momentum=predictor_batchnorm_momentum,
                dropout_rate=predictor_dropout,
                last_layer=None,
            )

    def __call__(self, x: ArrayLike, **kwargs) -> SiameseOutput:
        """Forward pass through the Siamese arm.

        Args:
            x: Input tensor.

        Returns:
            SiameseOutput.
        """
        y = self.encoder(x, **kwargs)
        z = self.projector(y)
        q = self.predictor(z)

        if self.normalize_y:
            y = _normalize(y)

        if self.normalize_z_q:
            z = _normalize(z)
            q = _normalize(q)

        return SiameseOutput(y, z, q)


@argbind.bind()
@dataclass(frozen=True)
class SLAPConfig:
    """Configuration for the SLAP model.

    Args:
        embed_dim: Dimension of the shared projection/prediction spaces.
        pred_hidden_dim: Hidden dimension of the predictor MLPs.
        normalize_y: Whether to normalize latent representations (y).
        normalize_z_q: Whether to normalize projections/predictions (z and q).
        projector_dropout: Dropout rate for projector MLP.
        predictor_dropout: Dropout rate for predictor MLP.
        predictor_batchnorm_momentum: BatchNorm momentum in predictor's MLP layers.
    """

    embed_dim: int = 512
    pred_hidden_dim: int = 4096
    normalize_y: bool = False
    normalize_z_q: bool = True
    projector_dropout: float = 0.0
    predictor_dropout: float = 0.1
    predictor_batchnorm_momentum: float = 0.9


class SLAP(SaveLoadModule):
    """SLAP model for multimodal representation learning without negative samples.

    Learns joint representations between two modalities (e.g., audio and
    synthesizer parameters) without requiring negative samples.
    """

    def __init__(
        self,
        config: SLAPConfig,
        encoder1: Module,
        encoder2: Module,
        rngs: Rngs = None,
    ):
        """Initialize the SLAP model.

        Args:
            config: SLAPConfig with architecture parameters.
            encoder1: A Module instance for the first modality's encoder.
            encoder2: A Module instance for the second modality's encoder.
            rngs: Random number generators.
        """
        self.normalize_z_q = config.normalize_z_q

        # Create Siamese arms for each modality
        self.arm1 = SiameseArm(
            encoder=encoder1,
            embed_dim=config.embed_dim,
            pred_hidden_dim=config.pred_hidden_dim,
            normalize_y=config.normalize_y,
            normalize_z_q=config.normalize_z_q,
            projector_dropout=config.projector_dropout,
            predictor_dropout=config.predictor_dropout,
            predictor_batchnorm_momentum=config.predictor_batchnorm_momentum,
            rngs=rngs,
        )

        self.arm2 = SiameseArm(
            encoder=encoder2,
            embed_dim=config.embed_dim,
            pred_hidden_dim=config.pred_hidden_dim,
            normalize_y=config.normalize_y,
            normalize_z_q=config.normalize_z_q,
            projector_dropout=config.projector_dropout,
            predictor_dropout=config.predictor_dropout,
            predictor_batchnorm_momentum=config.predictor_batchnorm_momentum,
            rngs=rngs,
        )

    def _forward(
        self,
        batch: Any,
        arm1_kwargs: dict[str, Any] = None,
        arm2_kwargs: dict[str, Any] = None,
    ) -> tuple[SiameseOutput, SiameseOutput]:
        if arm1_kwargs is None:
            arm1_kwargs = {}
        if arm2_kwargs is None:
            arm2_kwargs = {}
        out_arm1 = self.arm1(batch, **arm1_kwargs)
        out_arm2 = self.arm2(batch, **arm2_kwargs)

        return out_arm1, out_arm2

    def __call__(
        self,
        batch: Any,
        arm1_kwargs: dict[str, Any] = None,
        arm2_kwargs: dict[str, Any] = None,
    ) -> tuple[SiameseOutput, SiameseOutput]:
        """Forward pass through the SLAP model.

        Args:
            batch: Batch that will go to both Siamese arms.
            arm1_kwargs: Keyword arguments passed to the first SiameseArm.
            arm2_kwargs: Keyword arguments passed to the second SiameseArm.

        Returns:
            Tuple of (SiameseOutput, SiameseOutput).
        """
        return self._forward(batch, arm1_kwargs, arm2_kwargs)

    def forward_batches(
        self,
        batch: Any,
        mini_batch_size: int,
        arm1_kwargs: dict[str, Any] = None,
        arm2_kwargs: dict[str, Any] = None,
    ) -> tuple[SiameseOutput, SiameseOutput]:
        """A forward pass that uses nnx.scan to reduce memory consumption.

        Accepts any pytree batch (AudioTree, dict, etc.) and processes it in
        mini-batches. Subclasses can wrap the result in a typed output.
        """
        reshaped = tree_map(
            lambda x: (
                x.reshape(-1, mini_batch_size, *x.shape[1:])
                if hasattr(x, "shape")
                else x
            ),
            batch,
        )
        scan_fn = nnx.scan(
            SLAP._forward, in_axes=(None, 0, None, None), out_axes=(0, 0)
        )
        out1, out2 = scan_fn(self, reshaped, arm1_kwargs, arm2_kwargs)
        flatten = lambda x: tree_map(lambda a: a.reshape(-1, *a.shape[2:]), x)
        return flatten(out1), flatten(out2)
