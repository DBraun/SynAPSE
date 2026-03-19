import typing as tp

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import (
    ConvGeneralDilatedT,
    Dtype,
    Initializer,
    PaddingLike,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import lax

Array = jax.Array


def make_initializer(
    in_channels: int, out_channels: int, kernel_size, groups, mode="fan_in"
):
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    if mode == "fan_in":
        c = in_channels
    elif mode == "fan_out":
        c = out_channels
    else:
        raise ValueError(f"Unexpected mode: {mode}")
    k = groups / (c * jnp.prod(jnp.array(kernel_size)))
    scale = jnp.sqrt(k)
    return lambda key, shape, dtype: jax.random.uniform(
        key, shape, minval=-scale, maxval=scale, dtype=dtype
    )


class Conv(nnx.Conv):
    """Conv implementation with the same initialization as PyTorch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: tp.Union[None, int, tp.Sequence[int]] = 1,
        *,
        padding: PaddingLike = "SAME",
        input_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        kernel_dilation: tp.Union[None, int, tp.Sequence[int]] = 1,
        feature_group_count: int = 1,
        use_bias: bool = True,
        mask: tp.Optional[Array] = None,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: tp.Optional[Initializer] = None,
        bias_init: tp.Optional[Initializer] = None,
        conv_general_dilated: ConvGeneralDilatedT = lax.conv_general_dilated,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        if kernel_init is None:
            kernel_init = make_initializer(
                in_features,
                out_features,
                kernel_size,
                feature_group_count,
                mode="fan_in",
            )

        if use_bias and bias_init is None:
            bias_init = make_initializer(
                in_features,
                out_features,
                kernel_size,
                feature_group_count,
                mode="fan_in",
            )

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            input_dilation=input_dilation,
            kernel_dilation=kernel_dilation,
            feature_group_count=feature_group_count,
            use_bias=use_bias,
            mask=mask,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            conv_general_dilated=conv_general_dilated,
            promote_dtype=promote_dtype,
            rngs=rngs,
        )


class ConvTranspose1d(nnx.ConvTranspose):
    """ConvTranspose1d implementation with the same initialization as PyTorch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | tp.Sequence[int],
        strides: int | tp.Sequence[int] | None = None,
        *,
        padding: PaddingLike = "SAME",
        kernel_dilation: int | tp.Sequence[int] | None = None,
        use_bias: bool = True,
        mask: Array | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike | None = None,
        kernel_init: tp.Optional[Initializer] = None,
        bias_init: tp.Optional[Initializer] = None,
        transpose_kernel: bool = True,  # note: keep this True to help load weights from PyTorch
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
    ):
        groups = 1
        if kernel_init is None:
            kernel_init = make_initializer(
                in_features,
                out_features,
                kernel_size,
                groups,
                mode="fan_out",
            )

        if use_bias and bias_init is None:
            bias_init = make_initializer(
                in_features,
                out_features,
                kernel_size,
                groups,
                mode="fan_out",
            )
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_dilation=kernel_dilation,
            use_bias=use_bias,
            mask=mask,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=kernel_init,
            bias_init=bias_init,
            transpose_kernel=transpose_kernel,
            promote_dtype=promote_dtype,
            rngs=rngs,
        )
