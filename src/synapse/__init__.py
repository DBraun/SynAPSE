"""SynAPSE: FM-synthesizer audio-parameter shared embeddings (JAX / Flax NNX).

A multimodal BYOL-style (SLAP) model that learns a joint embedding space for DX7
audio and DX7 synthesizer presets. Given an audio query, retrieve presets with
high cosine similarity.

Top-level model: :class:`SynAPSE`. It pairs an audio encoder (``SynthRLWrapper``
or ``PANNsWrapper``) with a DX7 parameter encoder (``DX7GNNEncoder``,
``FlatParamEncoder``, or ``TransformerParamEncoder``) inside a SLAP siamese setup.
"""

from .audio_encoders import PANNsWrapper, SynthRLWrapper
from .backbones import Cnn14, PresetIndexesHelper, SynthRL
from .base_module import SaveLoadModule
from .layers import AttentionPool
from .model import SynAPSE, SynAPSEOutput
from .param_encoders import (
    DX7GNNEncoder,
    FlatParamEncoder,
    TransformerParamEncoder,
)
from .slap import (
    MLP,
    SLAP,
    SiameseArm,
    SiameseOutput,
    SLAPConfig,
)

__all__ = [
    # Top-level model
    "SynAPSE",
    "SynAPSEOutput",
    # SLAP core
    "SLAP",
    "SLAPConfig",
    "SiameseArm",
    "SiameseOutput",
    "MLP",
    "SaveLoadModule",
    # Audio encoders + backbones
    "PANNsWrapper",
    "SynthRLWrapper",
    "Cnn14",
    "SynthRL",
    "PresetIndexesHelper",
    "AttentionPool",
    # Parameter encoders
    "DX7GNNEncoder",
    "FlatParamEncoder",
    "TransformerParamEncoder",
]

__version__ = "0.1.0"
