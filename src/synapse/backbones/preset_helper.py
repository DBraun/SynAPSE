"""PresetIndexesHelper for SynthRL parameter mapping.

This is a JAX/Flax NNX port of the helper SynthRL carries from Preset-Gen-VAE,
originally implemented in PyTorch.

Original repositories:
    https://github.com/argaaw/SynthRL (MIT, Copyright (c) 2025 Wonchul Shin)
    https://github.com/gwendal-lv/preset-gen-vae (MIT, Copyright (c) 2026
    Gwendal Le Vaillant) — SynthRL's NOTICE attributes these portions here
License: MIT

Reference:
    Shin, W., & Lee, K. (2025). Cross-domain Synthesizer Sound Matching via
    Reinforcement Learning. In Proceedings of the International Joint
    Conference on Artificial Intelligence (IJCAI).

Handles mapping between:
- Full VST preset parameters (all synth parameters)
- Learnable parameters (subset used for prediction)
- Categorical vs numerical parameter types
"""

from collections.abc import Sequence
from enum import Enum


class _Synth(Enum):
    """Supported synthesizer types."""

    GENERIC = 0  # Undefined synth - numerical params only, all learnable
    DEXED = 1
    DIVA = 2


class PresetIndexesHelper:
    """Helper for converting between full-preset and learnable parameter indices.

    This class maps between:
    - Full VSTi preset indices (all parameters, VSTi-compatible)
    - Learnable parameter indices (used by neural network)

    Parameters can be:
    - Non-learnable (None): Fixed/ignored parameters
    - Numerical ('num'): Single regression value
    - Categorical ('cat'): One-hot encoded classes

    Reference: SynthRL/data/preset.py:23-290
    """

    def __init__(
        self,
        dataset=None,
        nb_params: int | None = None,
        synth_name: str = "generic_synth",
    ):
        """Initialize preset index helper.

        Args:
            dataset: Optional dataset with parameter information.
                If None, creates an identity translator for nb_params.
            nb_params: Number of parameters for identity translator.
            synth_name: Name of the synthesizer.
        """
        self._full_to_learnable: list[int | list[int] | None] = []
        self._learnable_to_full: list[int] = []

        # Identity default translator (all params are learnable numerical)
        if dataset is None:
            assert nb_params is not None, "Must provide nb_params if dataset is None"
            self._full_to_learnable = list(range(nb_params))
            self._learnable_to_full = list(range(nb_params))
            self._param_names = [f"param_{i}" for i in range(nb_params)]
            self._vst_param_learnable_model = ["num" for _ in range(nb_params)]
            self._param_cardinals = [-1 for _ in range(nb_params)]
            self._numerical_vst_params = list(range(nb_params))
            self._categorical_vst_params: list[int] = []
            self._learnable_preset_size = nb_params
            self.synth_name = synth_name
            self._synth = _Synth.GENERIC
        else:
            # Actual construction based on dataset
            self.synth_name = dataset.synth_name
            if self.synth_name.lower() == "dexed":
                self._synth = _Synth.DEXED
            elif self.synth_name.lower() == "diva":
                self._synth = _Synth.DIVA
            else:
                self._synth = _Synth.GENERIC

            self._param_names = dataset.preset_param_names
            self._vst_param_learnable_model = dataset.vst_param_learnable_model
            self._param_cardinals = [
                dataset.get_preset_param_cardinality(
                    vst_idx, learnable_representation=True
                )
                for vst_idx in range(dataset.total_nb_params)
            ]

            # Build index mappings
            current_learnable_idx = 0
            for vst_idx in range(dataset.total_nb_params):
                learn_model = dataset.vst_param_learnable_model[vst_idx]

                if learn_model is None:
                    self._full_to_learnable.append(None)
                elif learn_model == "num":
                    self._learnable_to_full.append(vst_idx)
                    self._full_to_learnable.append(current_learnable_idx)
                    current_learnable_idx += 1
                elif learn_model == "cat":
                    learnable_indexes = []
                    for _ in range(self._param_cardinals[vst_idx]):
                        self._learnable_to_full.append(vst_idx)
                        learnable_indexes.append(current_learnable_idx)
                        current_learnable_idx += 1
                    self._full_to_learnable.append(learnable_indexes)
                else:
                    raise ValueError(f"Unknown param learning model '{learn_model}'")

            self._learnable_preset_size = current_learnable_idx
            self._numerical_vst_params = dataset.numerical_vst_params
            self._categorical_vst_params = dataset.categorical_vst_params

        # Pre-compute index dictionaries for efficient lookup
        self._cat_idx_learned_as_num: dict[int, int] = {}
        self._cat_idx_learned_as_cat: dict[int, list[int]] = {}
        self._num_idx_learned_as_num: dict[int, int] = {}
        self._num_idx_learned_as_cat: dict[int, list[int]] = {}

        for vst_idx in self.categorical_vst_params:
            learn_model = self.vst_param_learnable_model[vst_idx]
            if learn_model is not None:
                if learn_model == "num":
                    idx = self.full_to_learnable[vst_idx]
                    assert isinstance(idx, int)
                    self._cat_idx_learned_as_num[vst_idx] = idx
                elif learn_model == "cat":
                    idx = self.full_to_learnable[vst_idx]
                    assert isinstance(idx, list)
                    self._cat_idx_learned_as_cat[vst_idx] = idx

        for vst_idx in self.numerical_vst_params:
            learn_model = self.vst_param_learnable_model[vst_idx]
            if learn_model is not None:
                if learn_model == "num":
                    idx = self.full_to_learnable[vst_idx]
                    assert isinstance(idx, int)
                    self._num_idx_learned_as_num[vst_idx] = idx
                elif learn_model == "cat":
                    idx = self.full_to_learnable[vst_idx]
                    assert isinstance(idx, list)
                    self._num_idx_learned_as_cat[vst_idx] = idx

    def __str__(self) -> str:
        learnable_count = sum(
            1
            for learn_model in self._vst_param_learnable_model
            if learn_model is not None
        )
        params_str = (
            f"[PresetIndexesHelper] {learnable_count} learnable VSTi parameters:\n"
        )
        for i, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model is not None:
                params_str += f"    - {i}.{self._param_names[i]}: {learn_model} ({self._full_to_learnable[i]})\n"
        return params_str

    @property
    def short_description(self) -> str:
        """Short description of the helper."""
        vsti_learnable_count = sum(
            1
            for learn_model in self._vst_param_learnable_model
            if learn_model is not None
        )
        tensor_learnable_size = 0
        for learnable_indexes in self._full_to_learnable:
            if isinstance(learnable_indexes, Sequence) and not isinstance(
                learnable_indexes, str
            ):
                tensor_learnable_size += len(learnable_indexes)
            elif isinstance(learnable_indexes, int):
                tensor_learnable_size += 1
        return (
            f"[PresetIndexesHelper] {vsti_learnable_count} learnable VSTi parameters, "
            f"learnable tensor representation size: {tensor_learnable_size}"
        )

    # Properties about VSTi (full-preset) parameters
    @property
    def full_preset_size(self) -> int:
        """Size of a full VSTi preset (learnable and non-learnable parameters)."""
        return len(self._full_to_learnable)

    @property
    def vst_param_names(self) -> list[str]:
        """Names of VSTi parameters."""
        return self._param_names

    @property
    def numerical_vst_params(self) -> list[int]:
        """VSTi-indexes of numerical synth parameters."""
        return self._numerical_vst_params

    @property
    def categorical_vst_params(self) -> list[int]:
        """VSTi-indexes of categorical synth parameters."""
        return self._categorical_vst_params

    @property
    def vst_param_learnable_model(self) -> list[str | None]:
        """None, 'num' or 'cat' for each full-preset parameter."""
        return self._vst_param_learnable_model

    @property
    def vst_param_cardinals(self) -> list[int]:
        """Cardinality of each parameter (-1 for continuous)."""
        return self._param_cardinals

    @property
    def full_to_learnable(self) -> list[int | list[int] | None]:
        """Map from full-preset index to learnable index(es).

        Returns None for non-learnable params, int for numerical,
        or list of ints for categorical.
        """
        return self._full_to_learnable

    # Pre-computed dictionaries for efficient lookup
    @property
    def cat_idx_learned_as_num(self) -> dict[int, int]:
        """Categorical VST params learned as numerical."""
        return self._cat_idx_learned_as_num

    @property
    def cat_idx_learned_as_cat(self) -> dict[int, list[int]]:
        """Categorical VST params learned as categorical."""
        return self._cat_idx_learned_as_cat

    @property
    def num_idx_learned_as_num(self) -> dict[int, int]:
        """Numerical VST params learned as numerical."""
        return self._num_idx_learned_as_num

    @property
    def num_idx_learned_as_cat(self) -> dict[int, list[int]]:
        """Numerical VST params learned as categorical."""
        return self._num_idx_learned_as_cat

    # Properties about learnable parameters
    @property
    def learnable_preset_size(self) -> int:
        """Size of the learnable representation of a preset."""
        return self._learnable_preset_size

    @property
    def learnable_to_full(self) -> list[int]:
        """Map from learnable index to full-preset VSTi index."""
        return self._learnable_to_full

    def get_numerical_learnable_indexes(self) -> list[int]:
        """Get indices of numerical parameters in learnable tensor."""
        numerical_indexes = []
        for vst_idx, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model == "num":
                idx = self._full_to_learnable[vst_idx]
                assert isinstance(idx, int)
                numerical_indexes.append(idx)
        return numerical_indexes

    def get_categorical_learnable_indexes(self) -> list[list[int]]:
        """Get indices of categorical parameters in learnable tensor."""
        categorical_indexes = []
        for vst_idx, learn_model in enumerate(self._vst_param_learnable_model):
            if learn_model == "cat":
                idx = self._full_to_learnable[vst_idx]
                assert isinstance(idx, list)
                categorical_indexes.append(idx)
        return categorical_indexes
