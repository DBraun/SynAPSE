"""SynAPSE: FM Synthesizer Audio-Parameter Shared Embeddings.

A SLAP (BYOL-style, non-contrastive) model with two encoders — an audio encoder
and a DX7 synthesizer-parameter encoder — trained to produce a shared embedding
space. Given an audio query, retrieve presets with high cosine similarity.
"""

import argbind
from absl import logging
from flax import nnx
from flax.struct import dataclass

from .audio_encoders import PANNsWrapper, SynthRLWrapper
from .param_encoders import (
    DX7GNNEncoder,
    FlatParamEncoder,
    TransformerParamEncoder,
)
from .slap import SLAP, SiameseOutput, SLAPConfig


@dataclass
class SynAPSEOutput:
    """Output from a SynAPSE forward pass.

    This is a flax.struct.dataclass, automatically registered as a JAX pytree.
    """

    audio_output: SiameseOutput
    parameter_output: SiameseOutput


@argbind.bind()
class SynAPSE(SLAP):
    """Synthesizer Audio-Parameter Shared Embeddings."""

    def __init__(
        self,
        encoder: str = "gnn",
        audio_encoder: str = "panns",
        rngs: nnx.Rngs = None,
    ):
        """Initialize SynAPSE.

        Args:
            encoder: Parameter encoder type — "gnn", "flat", or "transformer".
            audio_encoder: Audio encoder type — "panns" or "synthrl". The
                default matches the released checkpoints, and needs no torch:
                "synthrl" loads pretrained weights via the optional
                ``synthrl-pretrained`` extra unless its mode is "scratch".
            rngs: Random number generators. Defaults to ``nnx.Rngs(0)`` so the
                model can be rebuilt with ``cls()`` under ``nnx.eval_shape``.
        """
        if rngs is None:
            rngs = nnx.Rngs(0)

        # Create audio encoder
        audio_encoder_type = audio_encoder.lower()
        if audio_encoder_type == "synthrl":
            audio_enc = SynthRLWrapper(rngs=rngs)
        elif audio_encoder_type == "panns":
            audio_enc = PANNsWrapper(rngs=rngs)
        else:
            raise ValueError(
                f"Unknown audio_encoder: {audio_encoder_type}. "
                f"Options: 'synthrl', 'panns'"
            )

        # Create synthesizer parameter encoder
        encoder_type = encoder.lower()

        if encoder_type == "gnn":
            logging.info("Using GNN encoder for DX7")
            param_enc = DX7GNNEncoder(rngs=rngs)
        elif encoder_type == "flat":
            logging.info("Using FlatParamEncoder for DX7")
            param_enc = FlatParamEncoder(rngs=rngs)
        elif encoder_type == "transformer":
            logging.info("Using TransformerParamEncoder for DX7")
            param_enc = TransformerParamEncoder(rngs=rngs)
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. Options: 'flat', 'gnn', 'transformer'"
            )

        super().__init__(
            config=SLAPConfig(),
            encoder1=audio_enc,
            encoder2=param_enc,
            rngs=rngs,
        )

    def __call__(
        self, *args, audio_kwargs=None, param_kwargs=None, **kwargs
    ) -> SynAPSEOutput:
        return SynAPSEOutput(
            *super().__call__(
                *args,
                arm1_kwargs=audio_kwargs,
                arm2_kwargs=param_kwargs,
                **kwargs,
            )
        )

    def forward_batches(
        self, batch, mini_batch_size, audio_kwargs=None, param_kwargs=None
    ) -> SynAPSEOutput:
        return SynAPSEOutput(
            *super().forward_batches(batch, mini_batch_size, audio_kwargs, param_kwargs)
        )
