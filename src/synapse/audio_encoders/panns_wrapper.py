"""PANNs CNN14 wrapper for SynAPSE audio encoding.

Wraps :class:`Cnn14` to produce audio embeddings compatible with SLAP's
SiameseArm. Cnn14 handles its own mel spectrogram extraction and global pooling
internally.
"""

import argbind
import jax
from audiotree import AudioTree
from flax import nnx
from flax.typing import Dtype
from jax import numpy as jnp

from synapse.backbones import Cnn14
from synapse.utils import parse_dtype


@argbind.bind()
class PANNsWrapper(nnx.Module):
    """PANNs audio encoder wrapper for SynAPSE.

    Wraps Cnn14 and extracts the mid-channel embedding. Cnn14 internally
    computes mel spectrograms, runs 6 conv blocks, applies max+mean time
    pooling, and projects to embed_dim.

    Args:
        model_name: PANNs model variant (only "Cnn14" is supported).
        output_dim: Output embedding dimension (sets Cnn14's embed_dim).
        sample_rate: Audio sample rate.
        window_size: STFT window size.
        hop_size: STFT hop size.
        mel_bins: Number of mel frequency bins.
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank (None for Nyquist).
        use_batchnorm: Whether to use batch normalization in conv blocks.
        use_specaug: Whether to apply SpecAugmentation during training.
        dropout_rate: Dropout rate in conv blocks.
        dropout_rate_fc: Dropout rate in FC layers.
        use_fc: Whether to use FC layer before final projection.
        input_norm: Input normalization type ("batchnorm1d", "batchnorm2d", "none").
        dtype: Computation dtype.
        rngs: Random number generators.
    """

    def __init__(
        self,
        model_name: str = "Cnn14",
        output_dim: int = 768,
        sample_rate: int = 22050,
        window_size: int = 2048,
        hop_size: int = 1024,
        mel_bins: int = 128,
        fmin: float = 0.0,
        fmax: float | None = None,
        use_batchnorm: bool = True,
        use_specaug: bool = True,
        dropout_rate: float = 0.2,
        dropout_rate_fc: float = 0.5,
        use_fc: bool = True,
        input_norm: str = "batchnorm1d",
        dtype: Dtype | None = jnp.float32,
        rngs: nnx.Rngs | None = None,
    ):
        self.out_features = output_dim
        self.sample_rate = sample_rate

        dtype = parse_dtype(dtype)

        if model_name.lower() != "cnn14":
            raise ValueError(f"Unknown PANNs model name: {model_name}")

        self.panns = Cnn14(
            embed_dim=output_dim,
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            use_batchnorm=use_batchnorm,
            use_specaug=use_specaug,
            dropout_rate=dropout_rate,
            dropout_rate_fc=dropout_rate_fc,
            use_fc=use_fc,
            input_norm=input_norm,
            dtype=dtype,
            param_dtype=dtype,
            rngs=rngs,
        )

        self.panns.train()

    def __call__(self, audio_tree: AudioTree, **kwargs) -> jax.Array:
        """Encode audio to a fixed-dim embedding.

        The tree is passed to :class:`~synapse.backbones.Cnn14` untouched, which
        resamples and mixes channels itself. If it carries ``extras["mel"]``,
        Cnn14 uses that and skips feature extraction — the fast path for
        pre-rendered data.

        Args:
            audio_tree: AudioTree with a ``(B, C, T)`` waveform, or a precomputed
                ``extras["mel"]`` of shape ``(B, C, time_steps, mel_bins)`` as
                produced by ``Cnn14.waveform2mel`` — one channel for mono, two
                for stereo (channel 0 = mid, channel 1 = side). The mel must come
                from these same mel settings: a mel computed for another encoder,
                a SynthRL-format one say, silently encodes the wrong features, so
                a pipeline serving several audio encoders must key its cached mel
                to the encoder that computed it.

        Returns:
            Audio embeddings of shape (B, output_dim).
        """
        mid_embed, _side_embed = self.panns(audio_tree)
        return mid_embed
