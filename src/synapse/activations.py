"""Activation-name resolution for configurable Flax NNX modules.

Turns a plain ``activation: str`` hyperparameter (configurable from YAML or the
command line) into the corresponding Flax NNX callable, so a module can expose
``activation="gelu_tanh"`` and resolve it to a function at construction time.
"""

from collections.abc import Callable
from functools import partial

import jax
from flax import nnx

# The two GELU variants, kept as distinct callables so a config can pick either.
# ``approximate=False`` is the exact erf-based GELU (PyTorch/HuggingFace ``"gelu"``
# default); ``approximate=True`` is the tanh approximation (HuggingFace
# ``"gelu_new"`` / ``"gelu_pytorch_tanh"``, GPT-2). Note ``nnx.gelu``/``jax.nn.gelu``
# default to the *approximate* form, so we make the exactness explicit here rather
# than relying on that default.
_gelu_exact: Callable[[jax.Array], jax.Array] = partial(nnx.gelu, approximate=False)
_gelu_tanh: Callable[[jax.Array], jax.Array] = partial(nnx.gelu, approximate=True)

# String name -> zero-configuration elementwise activation callable from Flax NNX.
# Every value is a ``Callable[[jax.Array], jax.Array]`` usable directly as a layer
# inside ``nnx.Sequential``. Aliases (``swish``/``silu``, ``hard_swish``/``hard_silu``,
# ``linear``/``none``/``identity``, and the GELU variant names) resolve to the same
# function.
_ACTIVATIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": nnx.relu,
    "relu6": nnx.relu6,
    "gelu": _gelu_exact,
    "gelu_exact": _gelu_exact,
    "gelu_erf": _gelu_exact,
    "gelu_tanh": _gelu_tanh,
    "gelu_approx": _gelu_tanh,
    "gelu_new": _gelu_tanh,
    "gelu_pytorch_tanh": _gelu_tanh,
    "tanh": nnx.tanh,
    "sigmoid": nnx.sigmoid,
    "silu": nnx.silu,
    "swish": nnx.swish,
    "softplus": nnx.softplus,
    "soft_sign": nnx.soft_sign,
    "elu": nnx.elu,
    "celu": nnx.celu,
    "selu": nnx.selu,
    "leaky_relu": nnx.leaky_relu,
    "hard_tanh": nnx.hard_tanh,
    "hard_sigmoid": nnx.hard_sigmoid,
    "hard_silu": nnx.hard_silu,
    "hard_swish": nnx.hard_swish,
    "log_sigmoid": nnx.log_sigmoid,
    "identity": nnx.identity,
    "linear": nnx.identity,
    "none": nnx.identity,
}


def parse_activation(name: str, **kwargs) -> Callable[[jax.Array], jax.Array]:
    """Resolve an activation name to its Flax NNX callable.

    This lets a module expose a plain ``activation: str = "relu"`` hyperparameter
    (configurable from YAML or the command line) and turn it into the
    corresponding function, e.g. ``parse_activation("relu") is nnx.relu``. The
    returned callable is a stateless elementwise op that can be dropped straight
    into an ``nnx.Sequential``.

    Extra keyword arguments are bound onto the returned callable for parametric
    activations, e.g. ``parse_activation("leaky_relu", negative_slope=0.2)`` or
    ``parse_activation("elu", alpha=1.0)``. With no kwargs the canonical
    registered callable is returned unchanged (so ``parse_activation("relu") is
    nnx.relu``).

    Args:
        name: Case-insensitive activation name, e.g. ``"relu"``, ``"gelu"``,
            ``"leaky_relu"``. ``"linear"``/``"none"`` map to the identity.
        **kwargs: Optional keyword arguments bound onto the activation (e.g.
            ``alpha`` for ``elu``/``celu``, ``negative_slope`` for ``leaky_relu``).

    Returns:
        The activation function ``Callable[[jax.Array], jax.Array]``.

    Raises:
        ValueError: If ``name`` is not a known activation.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}."
        )
    fn = _ACTIVATIONS[key]
    return partial(fn, **kwargs) if kwargs else fn
