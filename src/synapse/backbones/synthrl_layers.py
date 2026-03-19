"""SynthRL layers: CNN backbone, transformer layers, and position encodings.

This is a JAX/Flax NNX port of SynthRL, originally implemented in PyTorch.

Original repository: https://github.com/argaaw/SynthRL
License: MIT, Copyright (c) 2025 Wonchul Shin

Reference:
    Shin, W., & Lee, K. (2025). Cross-domain Synthesizer Sound Matching via
    Reinforcement Learning. In Proceedings of the International Joint
    Conference on Artificial Intelligence (IJCAI).
"""

import math
from functools import partial

import jax
from einops import rearrange
from flax import nnx
from flax.nnx import BatchNorm, Dropout, LayerNorm, leaky_relu
from jax import numpy as jnp
from jax.typing import ArrayLike

from synapse.layers.torch_nn import Conv, Linear

_gelu = partial(nnx.gelu, approximate=False)


class CNNBackbone(nnx.Module):
    """CNN feature extractor for the backbone of Transformer encoder.

    5-layer CNN that progressively downsamples spectrograms:
    1 → 16 → 32 → 64 → 128 → 256 features with stride 2 at each layer.

    Reference: SynthRL/model/network.py:10-48
    """

    def __init__(
        self,
        in_features: int = 1,
        out_features: int = 256,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize CNN backbone.

        Args:
            in_features: Number of input features (1 for mel-spectrogram).
            out_features: Number of output features (d_model for transformer).
            dropout: Dropout rate applied after each activation+batchnorm block.
            rngs: Random number generators.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = Dropout(rate=dropout, rngs=rngs)

        # Build conv layers: 1 → 16 → 32 → 64 → 128 → 256
        feature_sizes = [
            in_features,
            out_features // 16,  # 16
            out_features // 8,  # 32
            out_features // 4,  # 64
            out_features // 2,  # 128
            out_features,  # 256
        ]

        # First layer: kernel=5, no batchnorm
        # Rest: kernel=4, with batchnorm
        self.conv0 = Conv(
            in_features=feature_sizes[0],
            out_features=feature_sizes[1],
            kernel_size=(5, 5),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            rngs=rngs,
        )

        self.conv1 = Conv(
            in_features=feature_sizes[1],
            out_features=feature_sizes[2],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            rngs=rngs,
        )
        self.bn1 = BatchNorm(
            num_features=feature_sizes[2],
            momentum=0.1,
            epsilon=1e-5,
            rngs=rngs,
        )

        self.conv2 = Conv(
            in_features=feature_sizes[2],
            out_features=feature_sizes[3],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            rngs=rngs,
        )
        self.bn2 = BatchNorm(
            num_features=feature_sizes[3],
            momentum=0.1,
            epsilon=1e-5,
            rngs=rngs,
        )

        self.conv3 = Conv(
            in_features=feature_sizes[3],
            out_features=feature_sizes[4],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            rngs=rngs,
        )
        self.bn3 = BatchNorm(
            num_features=feature_sizes[4],
            momentum=0.1,
            epsilon=1e-5,
            rngs=rngs,
        )

        self.conv4 = Conv(
            in_features=feature_sizes[4],
            out_features=feature_sizes[5],
            kernel_size=(4, 4),
            strides=(2, 2),
            padding=((2, 2), (2, 2)),
            rngs=rngs,
        )
        self.bn4 = BatchNorm(
            num_features=feature_sizes[5],
            momentum=0.1,
            epsilon=1e-5,
            rngs=rngs,
        )

    def __call__(self, x: ArrayLike) -> jax.Array:
        """Forward pass through CNN backbone.

        Args:
            x: Input mel-spectrogram of shape (B, H, W, C).

        Returns:
            Feature map of shape (B, H', W', out_features).
        """
        # Layer 0: conv + leaky_relu + dropout (no batchnorm)
        x = self.conv0(x)
        x = leaky_relu(x, negative_slope=0.1)
        x = self.dropout(x)

        # Layer 1: conv + leaky_relu + batchnorm + dropout
        x = self.conv1(x)
        x = leaky_relu(x, negative_slope=0.1)
        x = self.bn1(x)
        x = self.dropout(x)

        # Layer 2
        x = self.conv2(x)
        x = leaky_relu(x, negative_slope=0.1)
        x = self.bn2(x)
        x = self.dropout(x)

        # Layer 3
        x = self.conv3(x)
        x = leaky_relu(x, negative_slope=0.1)
        x = self.bn3(x)
        x = self.dropout(x)

        # Layer 4
        x = self.conv4(x)
        x = leaky_relu(x, negative_slope=0.1)
        x = self.bn4(x)
        x = self.dropout(x)

        return x


def create_2d_sin_embedding(
    num_pos_feats: int,
    height: int,
    width: int,
    temperature: float = 10000.0,
    normalize: bool = True,
    scale: float = 2 * math.pi,
) -> jax.Array:
    """Create 2D sinusoidal positional embeddings.

    Creates positional embeddings for 2D feature maps, with separate
    sinusoidal encodings for height and width dimensions.

    Reference: SynthRL/model/position_encoding.py:10-48 (PositionEmbeddingSine)

    Args:
        num_pos_feats: Number of position features (half of embedding dim).
        height: Height of the feature map.
        width: Width of the feature map.
        temperature: Temperature for sinusoidal encoding.
        normalize: Whether to normalize positions to [0, scale].
        scale: Scale factor for normalization.

    Returns:
        Position embedding of shape (1, height, width, num_pos_feats * 2).
    """
    # Create position grids
    y_embed = jnp.arange(1, height + 1, dtype=jnp.float32)[:, None]
    x_embed = jnp.arange(1, width + 1, dtype=jnp.float32)[None, :]

    # Broadcast to full grid
    y_embed = jnp.broadcast_to(y_embed, (height, width))
    x_embed = jnp.broadcast_to(x_embed, (height, width))

    if normalize:
        eps = 1e-6
        y_embed = y_embed / (height + eps) * scale
        x_embed = x_embed / (width + eps) * scale

    # Compute sinusoidal embeddings
    dim_t = jnp.arange(num_pos_feats, dtype=jnp.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)
    pos_y = y_embed[:, :, None] / dim_t  # (H, W, num_pos_feats)

    # Interleave sin and cos
    pos_x_sin = jnp.sin(pos_x[:, :, 0::2])
    pos_x_cos = jnp.cos(pos_x[:, :, 1::2])
    pos_x = rearrange(
        jnp.stack([pos_x_sin, pos_x_cos], axis=-1), "h w n two -> h w (n two)"
    )

    pos_y_sin = jnp.sin(pos_y[:, :, 0::2])
    pos_y_cos = jnp.cos(pos_y[:, :, 1::2])
    pos_y = rearrange(
        jnp.stack([pos_y_sin, pos_y_cos], axis=-1), "h w n two -> h w (n two)"
    )

    # Concatenate y and x embeddings
    pos = jnp.concatenate([pos_y, pos_x], axis=-1)  # (H, W, num_pos_feats * 2)

    return pos[None, :, :, :]  # Add batch dimension


def create_1d_sin_embedding(
    d_model: int,
    max_len: int,
) -> jax.Array:
    """Create 1D sinusoidal positional embeddings.

    Reference: SynthRL/model/position_encoding.py:51-83 (PositionalEncoding1D)

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length (number of queries).

    Returns:
        Position embedding of shape (max_len, d_model).
    """
    pos = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]

    _2i = jnp.arange(0, d_model, step=2, dtype=jnp.float32)

    encoding = jnp.zeros((max_len, d_model))
    encoding = encoding.at[:, 0::2].set(jnp.sin(pos / (10000 ** (_2i / d_model))))
    encoding = encoding.at[:, 1::2].set(jnp.cos(pos / (10000 ** (_2i / d_model))))

    return encoding


class TransformerEncoderLayer(nnx.Module):
    """Transformer encoder layer with pre-norm architecture.

    Reference: SynthRL/model/layer.py:143-200 (TransformerEncoderLayer)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize transformer encoder layer.

        Args:
            d_model: Model dimension.
            nhead: Number of attention heads.
            dim_feedforward: FFN hidden dimension.
            dropout: Dropout rate.
            rngs: Random number generators.
        """
        self.d_model = d_model
        self.nhead = nhead

        # Self-attention
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=nhead,
            in_features=d_model,
            decode=False,
            dropout_rate=dropout,
            deterministic=False,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )

        # FFN
        self.linear1 = Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = Linear(dim_feedforward, d_model, rngs=rngs)

        # Normalization (pre-norm)
        self.norm1 = LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = LayerNorm(num_features=d_model, rngs=rngs)

        # Dropout
        self.dropout1 = Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = Dropout(rate=dropout, rngs=rngs)
        self.dropout_ffn = Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        src: ArrayLike,
        pos: ArrayLike | None = None,
        mask: ArrayLike | None = None,
    ) -> jax.Array:
        """Forward pass with pre-norm architecture.

        Args:
            src: Input tensor of shape (B, T, C).
            pos: Position embedding of shape (B, T, C) or (T, C).
            mask: Optional attention mask.

        Returns:
            Output tensor of shape (B, T, C).
        """
        # Pre-norm self-attention
        src2 = self.norm1(src)
        q = k = src2 if pos is None else src2 + pos
        # MultiHeadAttention: query, key, value
        src2 = self.self_attn(q, k, src2, mask=mask)
        src = src + self.dropout1(src2)

        # Pre-norm FFN
        src2 = self.norm2(src)
        src2 = self.linear1(src2)
        src2 = _gelu(src2)
        src2 = self.dropout_ffn(src2)
        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)

        return src


class TransformerDecoderLayer(nnx.Module):
    """Transformer decoder layer with pre-norm architecture.

    Has self-attention, cross-attention, and FFN sublayers.

    Reference: SynthRL/model/layer.py:203-286 (TransformerDecoderLayer)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize transformer decoder layer.

        Args:
            d_model: Model dimension.
            num_heads: Number of attention heads.
            dim_feedforward: FFN hidden dimension.
            dropout: Dropout rate.
            rngs: Random number generators.
        """
        self.d_model = d_model
        self.nhead = num_heads

        # Self-attention
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            decode=False,
            dropout_rate=dropout,
            deterministic=False,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )

        # Cross-attention
        self.cross_attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_model,
            decode=False,
            dropout_rate=dropout,
            deterministic=False,
            kernel_init=nnx.initializers.xavier_uniform(),
            rngs=rngs,
        )

        # FFN
        self.linear1 = Linear(d_model, dim_feedforward, rngs=rngs)
        self.linear2 = Linear(dim_feedforward, d_model, rngs=rngs)

        # Normalization (pre-norm)
        self.norm1 = LayerNorm(num_features=d_model, rngs=rngs)
        self.norm2 = LayerNorm(num_features=d_model, rngs=rngs)
        self.norm3 = LayerNorm(num_features=d_model, rngs=rngs)

        # Dropout
        self.dropout1 = Dropout(rate=dropout, rngs=rngs)
        self.dropout2 = Dropout(rate=dropout, rngs=rngs)
        self.dropout3 = Dropout(rate=dropout, rngs=rngs)
        self.dropout_ffn = Dropout(rate=dropout, rngs=rngs)

    def __call__(
        self,
        tgt: ArrayLike,
        memory: ArrayLike,
        pos: ArrayLike | None = None,
        query_pos: ArrayLike | None = None,
        tgt_mask: ArrayLike | None = None,
        memory_mask: ArrayLike | None = None,
    ) -> jax.Array:
        """Forward pass with pre-norm architecture.

        Args:
            tgt: Target tensor of shape (B, T_q, C).
            memory: Encoder output of shape (B, T_k, C).
            pos: Encoder position embedding.
            query_pos: Decoder query position embedding.
            tgt_mask: Optional target mask.
            memory_mask: Optional memory mask.

        Returns:
            Output tensor of shape (B, T_q, C).
        """
        # Pre-norm self-attention on queries
        tgt2 = self.norm1(tgt)
        q = k = tgt2 if query_pos is None else tgt2 + query_pos
        tgt2 = self.self_attn(q, k, tgt2, mask=tgt_mask)
        tgt = tgt + self.dropout1(tgt2)

        # Pre-norm cross-attention
        tgt2 = self.norm2(tgt)
        q = tgt2 if query_pos is None else tgt2 + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.cross_attn(q, k, memory, mask=memory_mask)
        tgt = tgt + self.dropout2(tgt2)

        # Pre-norm FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear1(tgt2)
        tgt2 = _gelu(tgt2)
        tgt2 = self.dropout_ffn(tgt2)
        tgt2 = self.linear2(tgt2)
        tgt = tgt + self.dropout3(tgt2)

        return tgt
