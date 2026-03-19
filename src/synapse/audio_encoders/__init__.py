"""Audio-side encoders for SynAPSE (wrap the audio backbones for SLAP)."""

from .panns_wrapper import PANNsWrapper
from .synthrl_wrapper import SynthRLWrapper

__all__ = ["PANNsWrapper", "SynthRLWrapper"]
