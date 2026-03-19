import typing as tp

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.nn import dtypes
from flax.typing import (
    DotGeneralT,
    Dtype,
    Initializer,
    Optional,
    PrecisionLike,
    PromoteDtypeFn,
)
from jax import lax
from jax import numpy as jnp
from jax import random


def make_initializer_linear(in_channels: int):
    # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    k = 1 / in_channels
    scale = jnp.sqrt(k)
    return lambda key, shape, dtype: random.uniform(
        key, shape, minval=-scale, maxval=scale, dtype=dtype
    )


def Linear(
    in_features: int,
    out_features: int,
    *,
    use_bias: bool = True,
    dtype: tp.Optional[Dtype] = None,
    param_dtype: Dtype = jnp.float32,
    precision: PrecisionLike = None,
    kernel_init: Optional[Initializer] = None,
    bias_init: Optional[Initializer] = None,
    dot_general: DotGeneralT = lax.dot_general,
    promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    rngs: rnglib.Rngs,
):
    """A Linear layer with default kernel_init and bias_init that match PyTorch."""

    if kernel_init is None:
        kernel_init = make_initializer_linear(in_features)

    if use_bias:
        if bias_init is None:
            bias_init = make_initializer_linear(in_features)
    else:
        bias_init = None

    return nnx.Linear(
        in_features,
        out_features,
        use_bias=use_bias,
        dtype=dtype,
        param_dtype=param_dtype,
        precision=precision,
        kernel_init=kernel_init,
        bias_init=bias_init,
        dot_general=dot_general,
        promote_dtype=promote_dtype,
        rngs=rngs,
    )
