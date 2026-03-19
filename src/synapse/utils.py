"""Small standalone utilities: dtype resolution and the cache directory."""

import os
from pathlib import Path

from jax import numpy as jnp


def parse_dtype(dtype):
    """Resolve a dtype spec to the matching jax.numpy scalar type (e.g. ``jnp.float32``).

    Args:
        dtype: A string naming a jax.numpy dtype, with or without a ``jnp.`` / ``jax.numpy.`` /
            ``np.`` / ``numpy.`` prefix (``"float32"``, ``"jnp.bfloat16"``, ``"int32"``, ...); an
            already-resolved scalar type such as ``jnp.float32``; or a ``jax.numpy.dtype`` /
            ``numpy.dtype`` object. Non-string dtypes are normalized to the jax.numpy scalar type.

    Returns:
        The matching jax.numpy scalar type, e.g. ``jnp.float32``.

    Raises:
        TypeError: If ``dtype`` is neither a string nor a dtype-like value.
        ValueError: If ``dtype`` does not name a jax.numpy dtype.
    """
    if isinstance(dtype, str):
        name = dtype.strip()
        for prefix in ("jax.numpy.", "jnp.", "numpy.", "np."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
    elif isinstance(dtype, jnp.dtype):
        name = dtype.name
    elif isinstance(dtype, type):
        name = dtype.__name__
    else:
        raise TypeError(
            f"parse_dtype expects a dtype string or a dtype, got "
            f"{type(dtype).__name__}: {dtype!r}"
        )

    resolved = getattr(jnp, name, None)
    if isinstance(resolved, type) and (
        jnp.issubdtype(resolved, jnp.number) or jnp.issubdtype(resolved, jnp.bool_)
    ):
        return resolved
    raise ValueError(
        f"{dtype!r} does not name a jax.numpy dtype "
        "(e.g. 'float32', 'jnp.bfloat16', 'int32', 'bool')."
    )


def get_cache_dir() -> Path:
    """Return the on-disk cache directory for downloaded/converted checkpoints.

    Resolution order: ``$SYNAPSE_CACHE`` if it is an absolute path, else
    ``$XDG_CACHE_HOME/synapse`` if that is an absolute path, else
    ``~/.cache/synapse``. Both env vars are ignored when relative, since a
    relative cache root would follow the working directory. The directory is
    created if it does not exist.

    This is separate from the HuggingFace cache on purpose: it holds artifacts
    that do not come from the Hub (the SynthRL checkpoint is a GitHub release,
    plus its converted safetensors), and ``huggingface_hub`` is only the
    optional ``hf`` extra. Models loaded with ``from_pretrained`` from a repo id
    go to the HF cache, managed by ``huggingface_hub`` itself.

    Returns:
        The cache directory path.
    """
    env = os.environ.get("SYNAPSE_CACHE", "").strip()
    if env and os.path.isabs(env):
        cache_home = Path(env)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
        base = Path(xdg) if xdg and os.path.isabs(xdg) else Path.home() / ".cache"
        cache_home = base / "synapse"
    cache_home.mkdir(parents=True, exist_ok=True)
    return cache_home
