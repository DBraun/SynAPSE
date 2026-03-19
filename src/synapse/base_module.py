"""Uniform save/load mixin for the DX7 GNN encoder.

``SaveLoadModule`` gives a top-level ``@argbind.bind()`` model one interface for
checkpointing and distribution, so callers never touch the checkpoint helpers
directly. It is the standalone counterpart of a project's model base class,
carrying only the weight save/load surface — no audio-specific helpers.

Loading is driven by argbind: the model is rebuilt by replaying the
``argbind_params.yml`` (or the config embedded in a ``.safetensors``) under
``argbind.scope``, so a flat ``@argbind.bind()`` model round-trips with no
separate config object. A subclass ``__init__`` must therefore be **purely
constructive** (it is traced with ``nnx.eval_shape``): no branching on array
shapes/values, no ``.item()``, no numpy conversion of parameters. It must also
build with no required positional args — default ``rngs=None`` and fall back to
``nnx.Rngs(0)`` so ``cls()`` works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
from audiotree import AudioTree
from flax import nnx

from .checkpoint import (
    _load_argbind_config,
    dump_args,
    load_safetensors,
    materialize_abstract_rngs,
    restore_model_weights,
    save_safetensors,
)

#: argbind scopes that hold dataset/training-loop config rather than model architecture.
#: Dropped by ``save_pretrained`` by default so distributed configs don't leak local paths.
DEFAULT_DROP_SCOPES = ("train", "val", "test", "gen")


class SaveLoadModule(nnx.Module):
    """Mixin base class giving a model a uniform save/load interface.

    Provides the ``from_checkpoint`` / ``from_pretrained`` constructors, the
    ``save_orbax`` / ``save_safetensors`` / ``save_pretrained`` savers, and the
    audio-input helper ``require_waveform``. Subclass it instead of ``nnx.Module``
    for any model intended to be checkpointed, loaded for evaluation, or distributed.
    """

    # ------------------------------------------------------------------
    # Audio-input helpers
    # ------------------------------------------------------------------

    def gen(self, **attributes):
        self.eval(**attributes)

    def require_waveform(self, audio: Any, *, mono: bool = False) -> jax.Array:
        """Return ``audio``'s waveform resampled to this model's ``sample_rate``.

        Enforces the audio-input contract: an audio model accepts an
        :class:`audiotree.AudioTree` — which carries its ``sample_rate`` — rather
        than a bare waveform array whose rate would have to be guessed. Wrap raw
        audio with ``AudioTree(waveform=w, sample_rate=sr)``.

        Args:
            audio: An :class:`audiotree.AudioTree`.
            mono: If True, mix channels down to a single ``[B, 1, T]`` channel.

        Returns:
            The ``[B, C, T]`` (or ``[B, 1, T]`` when ``mono``) waveform at
            ``self.sample_rate``.
        """
        assert hasattr(
            self, "sample_rate"
        ), "require_waveform requires self.sample_rate to be defined"
        if not isinstance(audio, AudioTree):
            raise TypeError(
                f"{type(self).__name__} expects an audiotree.AudioTree (which carries "
                f"its sample_rate); got {type(audio).__name__}. Wrap raw audio with "
                f"AudioTree(waveform=w, sample_rate=sr)."
            )
        audio = audio.resample(self.sample_rate)
        if mono:
            audio = audio.to_mono()
        return audio.waveform

    # ------------------------------------------------------------------
    # Constructors (build + load weights)
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, *, step: int | None = None
    ) -> SaveLoadModule:
        """Load a trained model from an Orbax checkpoint or argbind ``.safetensors``.

        Reconstructs the model from the ``argbind_params.yml`` saved alongside the
        checkpoint (or embedded in the safetensors metadata), then fills the weights
        once via ``nnx.eval_shape`` (no double allocation).

        Args:
            path: An Orbax checkpoint directory (containing ``argbind_params.yml``) or
                a ``.safetensors`` file with an embedded argbind config.
            step: Orbax checkpoint step. If None, uses the latest step.

        Returns:
            The model in eval mode with restored weights.
        """
        import argbind

        path = Path(path)
        args = _load_argbind_config(path)

        # Loaders build on the default device; disable eager sharding so params
        # carrying partitioning annotations don't require an active mesh here.
        with argbind.scope(args), nnx.use_eager_sharding(False):
            model = nnx.eval_shape(lambda: cls())

        if path.suffix == ".safetensors":
            model.load_safetensors(path)
        else:
            model.load_orbax(path, step=step)

        # eval_shape leaves RNG state abstract; concretize so the model is a valid
        # concrete pytree.
        materialize_abstract_rngs(model)
        model.eval()
        return model

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        *,
        step: int | None = None,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> SaveLoadModule:
        """Load a distributed model from a directory or HuggingFace repo.

        The expected layout (produced by :meth:`save_pretrained`) is an
        ``argbind_params.yml`` plus a ``model.safetensors``. A bare ``.safetensors``
        file with an embedded config also works.

        Args:
            name_or_path: A local directory, a local ``.safetensors`` file, or a
                HuggingFace repo id such as ``"DBraun/dx7-gnn"``.
            step: Orbax step, used only if the resolved path is a training checkpoint.
            subfolder: Directory within the repo (or local directory) holding this
                model, for repos that publish several models side by side.
            cache_dir: Optional HuggingFace download cache directory.
            **kwargs: Forwarded to the HuggingFace downloader.

        Returns:
            The model in eval mode with restored weights.
        """
        import argbind

        path = Path(name_or_path)
        if path.suffix == ".safetensors" and path.exists():
            return cls.from_checkpoint(path)

        directory = cls._resolve_pretrained(
            name_or_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs
        )
        safetensors = directory / "model.safetensors"
        if not safetensors.exists():
            # Not a distribution dir; treat it as an Orbax training checkpoint.
            return cls.from_checkpoint(directory, step=step)

        if (directory / "argbind_params.yml").exists():
            args = _load_argbind_config(directory)
        else:
            args = _load_argbind_config(safetensors)

        with argbind.scope(args), nnx.use_eager_sharding(False):
            model = nnx.eval_shape(lambda: cls())
        model.load_safetensors(safetensors)

        materialize_abstract_rngs(model)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Savers / in-place loaders
    # ------------------------------------------------------------------

    def save_orbax(
        self,
        directory: str | Path,
        *,
        step: int = 0,
        optimizer: nnx.Optimizer | None = None,
    ) -> None:
        """Save model (and optionally optimizer) weights as an Orbax checkpoint.

        Produces a ``CheckpointManager``-style directory readable by ``load_orbax`` /
        ``from_checkpoint``.

        Args:
            directory: Output checkpoint directory.
            step: Checkpoint step to write under.
            optimizer: Optional optimizer whose state is saved alongside the model.
        """
        import jax
        from orbax import checkpoint as ocp

        directory = Path(directory).resolve()
        save_kwargs = {"model": ocp.args.PyTreeSave(jax.device_get(nnx.state(self)))}
        if optimizer is not None:
            save_kwargs["optimizer"] = ocp.args.PyTreeSave(
                jax.device_get(nnx.state(optimizer))
            )

        with ocp.CheckpointManager(directory) as manager:
            manager.save(step, args=ocp.args.Composite(**save_kwargs))
            manager.wait_until_finished()

    def load_orbax(self, directory: str | Path, *, step: int | None = None) -> None:
        """Restore model weights in-place from an Orbax checkpoint directory.

        Args:
            directory: Checkpoint directory managed by ``CheckpointManager``.
            step: Checkpoint step. If None, uses the latest step.
        """
        restore_model_weights(self, directory, step)

    def save_safetensors(
        self, path: str | Path, *, argbind_config: dict | None = None
    ) -> None:
        """Save model weights to a single ``.safetensors`` file.

        Args:
            path: Output ``.safetensors`` path.
            argbind_config: Optional argbind config dict embedded as JSON metadata so
                the file is self-contained for ``from_checkpoint`` / ``from_pretrained``.
        """
        save_safetensors(self, path, argbind_config=argbind_config)

    def load_safetensors(self, path: str | Path) -> None:
        """Restore model weights in-place from a ``.safetensors`` file.

        Args:
            path: Path to the ``.safetensors`` file.
        """
        load_safetensors(self, path)

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        argbind_config: dict,
        drop_scopes: tuple[str, ...] = DEFAULT_DROP_SCOPES,
    ) -> None:
        """Write a distributable model: ``argbind_params.yml`` + ``model.safetensors``.

        The resulting directory is loadable with ``from_pretrained`` and can be pushed
        to the HuggingFace Hub. The config is the model's argbind args dict (e.g. the
        ``argbind_params.yml`` from training) with ``drop_scopes`` removed, so local-only
        keys such as dataset paths are not published. For full manual control, pass an
        already-filtered ``argbind_config`` and ``drop_scopes=()``.

        Args:
            directory: Output directory.
            argbind_config: The argbind args dict needed to reconstruct the model
                (e.g. the ``argbind_params.yml`` written during training).
            drop_scopes: Scope prefixes (``"train"`` → ``"train/..."``) to strip before
                writing. Defaults to the dataset/training-loop scopes.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        config = self._clean_config(argbind_config, drop_scopes)
        dump_args(config, directory / "argbind_params.yml")
        self.save_safetensors(directory / "model.safetensors", argbind_config=config)

        readme = directory / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# {type(self).__name__}\n\n"
                "Load with:\n\n"
                "```python\n"
                f"from synapse import {type(self).__name__}\n"
                f'model = {type(self).__name__}.from_pretrained("<path-or-repo>")\n'
                "```\n"
            )

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_config(
        argbind_config: dict, drop_scopes: tuple[str, ...] = DEFAULT_DROP_SCOPES
    ) -> dict:
        """Strip scoped (dataset/training-loop) keys from an argbind config dict.

        Args:
            argbind_config: Full argbind args dict.
            drop_scopes: Scope names whose ``"<scope>/..."`` keys are removed.

        Returns:
            A new dict with the scoped keys dropped.
        """
        if not drop_scopes:
            return dict(argbind_config)
        prefixes = tuple(f"{scope}/" for scope in drop_scopes)
        return {
            key: value
            for key, value in argbind_config.items()
            if not key.startswith(prefixes)
        }

    @staticmethod
    def _resolve_pretrained(
        name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> Path:
        """Resolve a name-or-path to a local directory, downloading from HF if needed.

        Args:
            name_or_path: Local directory or HuggingFace repo id.
            subfolder: Directory within the repo (or local directory) holding the
                model. Only that subtree is downloaded.
            cache_dir: Optional HuggingFace download cache directory.
            **kwargs: Forwarded to ``huggingface_hub.snapshot_download``.

        Returns:
            A local directory containing ``argbind_params.yml`` and ``model.safetensors``.
        """
        path = Path(name_or_path)
        if path.exists():
            return path / subfolder if subfolder else path

        from huggingface_hub import snapshot_download

        if subfolder:
            kwargs.setdefault("allow_patterns", f"{subfolder}/*")

        snapshot = Path(
            snapshot_download(repo_id=str(name_or_path), cache_dir=cache_dir, **kwargs)
        )
        return snapshot / subfolder if subfolder else snapshot
