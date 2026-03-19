"""Weight save/load utilities backing the :class:`SaveLoadModule` mixin.

Two formats are supported:

- **Orbax** (training): a ``CheckpointManager`` directory, matching what most JAX
  trainers write.
- **Safetensors** (distribution): a single file with the argbind config optionally
  embedded as JSON metadata, so the file is self-contained.

Downstream code should prefer the :class:`SaveLoadModule` methods
(``from_checkpoint`` / ``from_pretrained`` / ``save_safetensors`` / ...) over
calling these helpers directly.
"""

import dataclasses
import json
import os
from pathlib import Path

import jax
import numpy as np
import yaml
from flax import nnx
from jax import numpy as jnp
from orbax import checkpoint as ocp


class DatasetStat(nnx.Variable):
    """Variable wrapper for dataset statistics (constant/frozen stats).

    Its own Variable type so dataset-derived constants — e.g. the log-mel
    mean/std :class:`~synapse.backbones.Cnn14` uses for input normalization —
    are both saved and restored. A bare ``nnx.Variable`` would be saved but
    silently skipped on restore.
    """


# Only these Variable types come from a checkpoint; RNG state and inference caches
# are rebuilt fresh and deliberately not restored.
_LOADABLE = nnx.Any(nnx.Param, nnx.BatchStat, DatasetStat)


def materialize_abstract_rngs(model: nnx.Module, seed: int = 0) -> None:
    """Replace abstract (``eval_shape``) RNG state with concrete keys/counts, in place.

    An ``eval_shape``-based loader leaves RNG state abstract — it isn't in the
    checkpoint and only ``Param``/``BatchStat`` are restored. Concretize it so the
    returned model is a valid concrete pytree (required before ``nnx.jit`` or any
    inference that draws noise).

    Args:
        model: A model whose RNG state may still be abstract after loading.
        seed: Seed used to build the concrete keys.
    """
    _, rng_state, _ = nnx.split(model, nnx.RngState, ...)

    def _concretize(leaf):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            return leaf
        if jax.dtypes.issubdtype(leaf.dtype, jax.dtypes.prng_key):
            if leaf.shape == ():
                return jax.random.key(seed)
            return jax.random.split(jax.random.key(seed), leaf.shape)
        return jnp.zeros(leaf.shape, leaf.dtype)  # RNG counts

    rng_state = jax.tree.map(
        _concretize, rng_state, is_leaf=lambda x: isinstance(x, jax.ShapeDtypeStruct)
    )
    nnx.update(model, rng_state)


def restore_model_weights(
    model: nnx.Module,
    checkpoint_dir: str | Path,
    step: int | None = None,
) -> None:
    """Restore only a model's weights in-place from an Orbax checkpoint.

    Targets just the loadable weights (``Param``/``BatchStat``), leaving RNG state
    and inference caches as freshly built — robust to models whose tree structure an
    ``nnx.eval_shape`` rebuild can't otherwise reproduce.

    Args:
        model: An nnx.Module (typically created via ``nnx.eval_shape`` for zero memory).
        checkpoint_dir: Path to the checkpoint directory managed by CheckpointManager.
        step: Checkpoint step. If None, uses the latest step.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    checkpoint_manager = ocp.CheckpointManager(directory=checkpoint_dir)
    step = step or checkpoint_manager.latest_step()

    weights = nnx.state(model, _LOADABLE)
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray), weights
    )
    restored = checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeRestore(
                weights, restore_args=restore_args, partial_restore=True
            )
        ),
    )
    nnx.update(model, restored.model)


# ---------------------------------------------------------------------------
# argbind config (embedded in / alongside the weights)
# ---------------------------------------------------------------------------


def _load_argbind_config(path: Path) -> dict:
    """Load an argbind config from a checkpoint directory or safetensors file.

    Args:
        path: Checkpoint directory (containing ``argbind_params.yml``) or a
            ``.safetensors`` file.

    Returns:
        Dictionary of argbind configuration parameters.
    """
    if path.suffix == ".safetensors":
        return _load_argbind_config_from_safetensors(path)

    with open(path / "argbind_params.yml") as f:
        return yaml.load(f, Loader=yaml.Loader)


def _load_argbind_config_from_safetensors(path: Path) -> dict:
    """Load an argbind config from safetensors file metadata.

    Args:
        path: Path to the ``.safetensors`` file.

    Returns:
        Dictionary of argbind configuration parameters.
    """
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        metadata = f.metadata()

    return json.loads(metadata["argbind_config"])


def dump_args(args: dict, output_path: str | Path) -> None:
    """Dump an argbind args dict to a YAML file (``argbind_params.yml`` layout).

    Args:
        args: Dictionary of arguments to dump.
        output_path: Path where the arguments should be saved.
    """
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        yaml.Dumper.ignore_aliases = lambda *a: True

        args = {
            k: v
            for k, v in args.items()
            if not isinstance(v, dataclasses._HAS_DEFAULT_FACTORY_CLASS)
        }

        x = yaml.dump(args, Dumper=yaml.Dumper)
        prev_line = None
        output = []
        for line in x.split("\n"):
            cur_line = line.split(".")[0].strip()
            if not cur_line.startswith("-"):
                if cur_line != prev_line and prev_line:
                    line = f"\n{line}"
                prev_line = line.split(".")[0].strip()
            output.append(line)
        f.write("\n".join(output))


# ---------------------------------------------------------------------------
# Safetensors (distribution format)
# ---------------------------------------------------------------------------


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict to dot-separated keys with numpy array values.

    Args:
        d: Nested dictionary (from ``nnx.to_pure_dict``).
        prefix: Key prefix for recursion.

    Returns:
        Flat dictionary mapping ``"a.b.c"`` keys to numpy arrays.
    """
    out: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = np.asarray(v)
    return out


def _unflatten_dict(flat: dict[str, np.ndarray]) -> dict:
    """Unflatten dot-separated keys back to a nested dict.

    All-digit path components are converted back to ``int`` so that list and
    ``nnx.Sequential`` indices round-trip (nnx uses integer keys for sequences,
    and module attribute names are never all-digit).

    Args:
        flat: Flat dictionary mapping ``"a.b.c"`` keys to arrays.

    Returns:
        Nested dictionary suitable for ``nnx.update``.
    """

    def _key(part: str) -> str | int:
        return int(part) if part.isdigit() else part

    out: dict = {}
    for key, value in flat.items():
        parts = key.split(".")
        d = out
        for part in parts[:-1]:
            d = d.setdefault(_key(part), {})
        d[_key(parts[-1])] = value
    return out


def save_safetensors(
    model: nnx.Module,
    path: str | Path,
    argbind_config: dict | None = None,
) -> None:
    """Export model weights to a single safetensors file.

    Embeds the argbind config as JSON metadata in the safetensors header so the
    file is fully self-contained for distribution.

    Args:
        model: An nnx.Module with trained weights.
        path: Output ``.safetensors`` file path.
        argbind_config: Argbind configuration dict to embed as metadata. If None,
            no config is embedded.
    """
    from safetensors.numpy import save_file

    # Persist only the loadable weights (Param/BatchStat). RNG state and inference
    # caches are rebuilt fresh on load and cannot be serialized to numpy (PRNGKey
    # dtype), so they are intentionally excluded — matching what load restores.
    state = nnx.state(model, _LOADABLE)
    pure_dict = nnx.to_pure_dict(state)
    flat = _flatten_dict(pure_dict)

    metadata = {}
    if argbind_config is not None:
        metadata["argbind_config"] = json.dumps(argbind_config)

    save_file(flat, str(path), metadata=metadata or None)


def load_safetensors(model: nnx.Module, path: str | Path) -> None:
    """Restore model weights in-place from a safetensors file.

    Args:
        model: An nnx.Module (can be created via ``nnx.eval_shape``).
        path: Path to the ``.safetensors`` file.
    """
    from safetensors.numpy import load_file

    flat = load_file(str(path))
    nested = _unflatten_dict(flat)
    nnx.update(model, nested)
