# Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
# Under MIT License
"""PANNs convolutional building blocks used by :class:`Cnn14`."""

from functools import partial

from flax import nnx
from flax.nnx import BatchNorm, Module, Rngs
from flax.nnx.module import first_from
from flax.nnx.nn import dtypes
from flax.typing import Dtype, PromoteDtypeFn
from jax import numpy as jnp
from jax.typing import ArrayLike

Linear = partial(nnx.Linear, kernel_init=nnx.initializers.xavier_uniform())
Conv = partial(nnx.Conv, kernel_init=nnx.initializers.xavier_uniform())


class ConvBlock(Module):
    """A convolutional block with two conv layers and optional batch normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        pool_size=(2, 2),
        pool_type: str = "avg",
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: Rngs = None,
    ):
        self.use_batchnorm = use_batchnorm
        self.pool_size = pool_size
        self.pool_type = pool_type
        self.dtype = dtype
        self.promote_dtype = promote_dtype

        self.conv1 = Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.conv2 = Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        if use_batchnorm:
            self.bn1 = BatchNorm(
                num_features=out_channels,
                momentum=0.9,
                epsilon=1e-5,
                axis=-1,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.bn2 = BatchNorm(
                num_features=out_channels,
                momentum=0.9,
                epsilon=1e-5,
                axis=-1,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

    def __call__(
        self,
        x: ArrayLike,
        pool_size: list[int] = None,
        pool_type: str = None,
    ):
        pool_size = first_from(
            self.pool_size, pool_size, error_msg="`pool_size` must be provided"
        )
        pool_type = first_from(
            self.pool_type, pool_type, error_msg="`pool_type` must be provided"
        )

        # Promote dtype if needed
        (x,) = self.promote_dtype((x,), dtype=self.dtype)

        # First conv + bn + relu
        x = self.conv1(x)

        if self.use_batchnorm:
            x = self.bn1(x)

        x = nnx.relu(x)

        # Second conv + bn + relu
        x = self.conv2(x)

        if self.use_batchnorm:
            x = self.bn2(x)

        x = nnx.relu(x)

        # Pooling
        if pool_type == "max":
            x = nnx.max_pool(x, window_shape=pool_size, strides=pool_size)
        elif pool_type == "avg":
            x = nnx.avg_pool(x, window_shape=pool_size, strides=pool_size)
        elif pool_type == "avg+max":
            x1 = nnx.avg_pool(x, window_shape=pool_size, strides=pool_size)
            x2 = nnx.max_pool(x, window_shape=pool_size, strides=pool_size)
            x = x1 + x2
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}")

        return x
