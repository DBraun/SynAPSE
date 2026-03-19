"""SynthRL model: CNN + Transformer for synthesizer parameter prediction.

This is a JAX/Flax NNX port of SynthRL, originally implemented in PyTorch.

Original repository: https://github.com/argaaw/SynthRL
License: MIT, Copyright (c) 2025 Wonchul Shin

Reference:
    Shin, W., & Lee, K. (2025). Cross-domain Synthesizer Sound Matching via
    Reinforcement Learning. In Proceedings of the International Joint
    Conference on Artificial Intelligence (IJCAI).

SynthRL predicts synthesizer parameters from mel-spectrograms using:
1. CNN backbone to extract features from spectrograms
2. Transformer encoder to process feature maps with positional encoding
3. Transformer decoder with learnable queries for parameter prediction
4. Per-parameter output heads (numerical or categorical)
"""

import argbind
import jax
import librosax
import librosax.feature
from audiotree import AudioTree
from einops import rearrange
from flax import nnx
from jax import numpy as jnp

from synapse.base_module import SaveLoadModule
from synapse.layers.torch_nn import Linear

from .preset_helper import PresetIndexesHelper
from .synthrl_layers import (
    CNNBackbone,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    create_1d_sin_embedding,
    create_2d_sin_embedding,
)


@argbind.bind()
class SynthRL(SaveLoadModule):
    """SynthRL: CNN + Transformer for synthesizer parameter prediction.

    A JAX/Flax NNX port of SynthRL (Shin & Lee, "Cross-domain Synthesizer
    Sound Matching via Reinforcement Learning", IJCAI 2025). It predicts a
    synthesizer preset from a recording so the synth re-creates that sound.

    Architecture:

    1. ``CNNBackbone`` extracts features from a mel-spectrogram.
    2. A Transformer encoder processes those features with 2D positional
       encoding.
    3. A Transformer decoder attends from ``n_queries`` learnable queries to
       the encoder memory.
    4. Per-query projection heads emit one value per parameter (numerical
       heads use ``tanh``; categorical heads emit class logits).

    Conditioning:
        A mono audio recording as an :class:`audiotree.AudioTree` (preprocessed
        to a mel-spectrogram), or a precomputed mel-spectrogram carried in
        ``AudioTree.extras["mel"]``.

    Output:
        :meth:`__call__` returns a ``(B, out_dim)`` parameter vector in
        ``[0, 1]`` (see :class:`PresetIndexesHelper` for the layout).

    Example:

        >>> from synapse.backbones import SynthRL
        >>> from audiotree import AudioTree
        >>> from jax import numpy as jnp
        >>> model = SynthRL.from_variant("in-domain-dexed")
        >>> audio = AudioTree(waveform=jnp.zeros((1, 1, 44100)), sample_rate=44100)
        >>> params = model(audio)   # (B, out_dim) in [0, 1]

    Reference: ``SynthRL/model/network.py:50-126``.
    """

    def __init__(
        self,
        preset_helper: PresetIndexesHelper,
        *,
        rngs: nnx.Rngs,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        cnn_dropout: float = 0.0,
        n_queries: int = 144,
        in_features: int = 1,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        encoder_only: bool = False,
    ):
        """Initialize SynthRL model.

        Args:
            preset_helper: Helper for mapping parameters to output heads.
            rngs: Random number generators.
            d_model: Model dimension / CNN output features.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            num_decoder_layers: Number of transformer decoder layers.
            dim_feedforward: FFN hidden dimension in transformer.
            dropout: Dropout rate for transformer layers.
            cnn_dropout: Dropout rate for CNN backbone (after each
                activation+batchnorm block). Defaults to 0.0 (no dropout).
            n_queries: Number of learnable queries (= number of parameters).
            in_features: Number of input features for mel-spectrogram.
            sample_rate: Audio sample rate for preprocessing.
            n_fft: FFT window size for mel spectrogram.
            hop_length: Hop length for mel spectrogram.
            n_mels: Number of mel frequency bins.
            encoder_only: If True, only build the CNN backbone and transformer
                encoder. Skips decoder, learnable queries, and projection heads.
                Use when you only need encode() (e.g., as a feature extractor).
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.n_queries = n_queries
        self.in_features = in_features
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.encoder_only = encoder_only

        self.preset_helper = preset_helper
        self.out_dim = preset_helper.learnable_preset_size

        # Get parameter indices
        self.cat_idx, self.num_idx = self._get_learnable_idx(preset_helper)

        # CNN backbone
        self.backbone = CNNBackbone(
            in_features=in_features,
            out_features=d_model,
            dropout=cnn_dropout,
            rngs=rngs,
        )

        # Transformer encoder layers
        self.encoder_layers = nnx.List(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rngs=rngs,
                )
                for _ in range(num_encoder_layers)
            ]
        )
        self.encoder_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        if encoder_only:
            return

        # Transformer decoder layers
        self.decoder_layers = nnx.List(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    rngs=rngs,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.decoder_norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

        # Learnable query embedding (initialized with normal distribution)
        self.tgt = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(
                rngs.params(),
                (n_queries, d_model),
                jnp.float32,
            )
        )

        # Per-query projection heads (matches PyTorch nn.ModuleList)
        proj_heads: list = []
        for i in range(len(self.cat_idx)):
            proj_heads.append(Linear(d_model, len(self.cat_idx[i]), rngs=rngs))
        for i in range(len(self.num_idx)):
            proj_heads.append(Linear(d_model, 1, rngs=rngs))
        self.proj_heads = nnx.List(proj_heads)

        # Dropout for projection heads
        self.proj_dropout = nnx.Dropout(rate=0.3, rngs=rngs)

    def encode(self, audio: AudioTree) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        """Run the CNN backbone and Transformer encoder.

        Args:
            audio: An :class:`audiotree.AudioTree` (mono), preprocessed to a
                mel-spectrogram and resampled to the model's ``sample_rate``. A
                precomputed mel-spectrogram may instead be supplied via
                ``audio.extras["mel"]`` of shape ``(B, H, W, C)`` (``H`` =
                frequency bins, ``W`` = time frames, ``C`` = channels, usually
                1; ``int16`` is dequantized to ``float32``), which bypasses the
                mel front-end.

        Returns:
            A tuple ``(feature_shape, spectrogram, memory)`` where:

            - ``feature_shape`` is the ``(B, H, W, C)`` shape of the CNN
              backbone output.
            - ``spectrogram`` is the mel-spectrogram of shape
              ``(B, n_mels, n_frames, 1)``.
            - ``memory`` is the encoder output of shape ``(B, H*W, d_model)``.
        """
        spectrogram = self.preprocess_audio(audio)

        # CNN backbone: (B, H, W, C) -> (B, H', W', d_model)
        features = self.backbone(spectrogram)
        B, H, W, C = features.shape

        # Create 2D positional embedding for encoder
        enc_pos_embed = create_2d_sin_embedding(
            num_pos_feats=self.d_model // 2,
            height=H,
            width=W,
        )  # (1, H, W, d_model)

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        features_flat = rearrange(features, "b h w c -> b (h w) c")
        enc_pos_flat = rearrange(enc_pos_embed, "b h w c -> b (h w) c")

        # Transformer encoder
        memory = features_flat
        for layer in self.encoder_layers:
            memory = layer(memory, pos=enc_pos_flat)
        memory = self.encoder_norm(memory)

        return (B, H, W, C), spectrogram, memory

    def __call__(self, audio: AudioTree) -> jax.Array:
        """Predict synthesizer parameters from audio.

        Runs :meth:`encode` (CNN backbone + encoder), then the Transformer
        decoder over ``n_queries`` learnable queries, then the per-query
        projection heads, producing one value per synth parameter.

        Args:
            audio: An :class:`audiotree.AudioTree` (mono), preprocessed to a
                mel-spectrogram and resampled to the model's ``sample_rate``. A
                precomputed mel-spectrogram may instead be supplied via
                ``audio.extras["mel"]`` of shape ``(B, H, W, C)`` (usually
                ``C == 1``; ``int16`` is dequantized to ``float32``).

        Returns:
            Predicted parameters of shape ``(B, out_dim)`` with values in
            ``[0, 1]`` (``out_dim == preset_helper.learnable_preset_size``).

        Raises:
            RuntimeError: If the model was created with ``encoder_only=True``.
        """
        if self.encoder_only:
            raise RuntimeError(
                "Cannot call forward() on an encoder_only model. Use encode() instead."
            )
        (B, H, W, C), spectrogram, memory = self.encode(audio)

        # Create 2D positional embedding for encoder
        enc_pos_embed = create_2d_sin_embedding(
            num_pos_feats=self.d_model // 2,
            height=H,
            width=W,
        )  # (1, H, W, d_model)

        # Flatten spatial dimensions: (B, H, W, C) -> (B, H*W, C)
        enc_pos_flat = rearrange(enc_pos_embed, "b h w c -> b (h w) c")

        # Create 1D positional embedding for decoder queries
        query_pos_embed = create_1d_sin_embedding(
            d_model=self.d_model,
            max_len=self.n_queries,
        )  # (n_queries, d_model)

        # Expand tgt to batch: (n_queries, d_model) -> (B, n_queries, d_model)
        tgt = jnp.broadcast_to(
            self.tgt[None, :, :],
            (B, self.n_queries, self.d_model),
        )

        # Transformer decoder
        dec_out = tgt
        for layer in self.decoder_layers:
            dec_out = layer(
                dec_out, memory, pos=enc_pos_flat, query_pos=query_pos_embed
            )
        dec_out = self.decoder_norm(dec_out)

        # Apply dropout to decoder output
        dec_out = self.proj_dropout(dec_out)

        # Per-query projection heads
        n_cat = len(self.cat_idx)
        out = jnp.zeros((B, self.out_dim), dtype=dec_out.dtype)
        for i in range(n_cat):
            out = out.at[:, jnp.array(self.cat_idx[i])].set(
                self.proj_heads[i](dec_out[:, i, :])
            )
        for i in range(len(self.num_idx)):
            out = out.at[:, self.num_idx[i]].set(
                self.proj_heads[n_cat + i](dec_out[:, n_cat + i, :]).squeeze(-1)
            )

        # Output activation: tanh scaled to [0, 1]
        out = jnp.tanh(out)
        out = 0.5 * (out + 1.0)

        return out

    def preprocess_audio(self, audio: AudioTree) -> jax.Array:
        """Preprocess an ``AudioTree`` to a mel-spectrogram for model input.

        Args:
            audio: An :class:`audiotree.AudioTree` with a ``(B, C, T)`` waveform
                where ``C`` must be 1 (mono); it is resampled to the model's
                ``sample_rate``. A precomputed mel-spectrogram carried in
                ``audio.extras["mel"]`` (``int16`` dequantized to ``float32``)
                is returned directly, bypassing the front-end.

        Returns:
            Mel-spectrogram of shape (B, n_mels, n_frames, 1) in log scale.

        Raises:
            ValueError: If audio is not mono (C != 1).
        """
        if audio.extras and "mel" in audio.extras:
            mel = audio.extras["mel"]
            if mel.dtype == jnp.int16:
                mel = mel.astype(jnp.float32) / 32767.0
            return mel

        # Resample to the model's sample rate; keep channels so a non-mono clip errors
        # loudly rather than being silently mixed down.
        waveform = self.require_waveform(audio, mono=False)  # (B, C, T)
        if waveform.shape[1] != 1:
            raise ValueError(
                f"SynthRL requires mono audio, got {waveform.shape[1]} channels. "
                "Convert to mono before passing to the model."
            )
        waveform = waveform[:, 0, :]  # (B, C, T) -> (B, T)

        # Compute mel-spectrogram (supports batch axis)
        # Parameters match torchaudio.transforms.MelSpectrogram defaults
        mel_specs = librosax.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            norm="slaney",  # Original code explicitly sets this (also librosax default)
            pad_mode="reflect",  # torchaudio default (librosax defaults to 'constant')
            htk=True,  # torchaudio default (librosax defaults to False/Slaney)
        )
        # Convert to log scale and normalize (matches original PyTorch preprocessing)
        # Original: torch.clip(torch.log(spec + 1e-5) / 12., -1, 1)
        mel_specs = jnp.clip(jnp.log(mel_specs + 1e-5) / 12.0, -1.0, 1.0)

        # Add channel dimension: (B, n_mels, n_frames) -> (B, n_mels, n_frames, 1)
        return mel_specs[:, :, :, None]

    def _get_learnable_idx(
        self, preset_helper: PresetIndexesHelper
    ) -> tuple[list[list[int]], list[int]]:
        """Extract categorical and numerical parameter indices.

        Args:
            preset_helper: Helper with full_to_learnable mapping.

        Returns:
            Tuple of (cat_idx, num_idx) where:
            - cat_idx: List of lists of indices for categorical parameters
            - num_idx: List of indices for numerical parameters
        """
        full_idx = preset_helper.full_to_learnable
        cat_idx, num_idx = [], []

        for idx in full_idx:
            if isinstance(idx, list):
                cat_idx.append(idx)
            elif isinstance(idx, int):
                num_idx.append(idx)

        return cat_idx, num_idx

    @classmethod
    def from_variant(cls, *args, **kwargs) -> "SynthRL":
        """Load this model with its upstream pretrained weights.

        Distinct from :meth:`~synapse.base_module.SaveLoadModule.from_pretrained`,
        which loads a model *we* published from a path or HuggingFace repo id.
        Delegates to the package-level :func:`load_model`. Its first argument is
        a variant key — ``"in-domain-dexed"`` (Dexed, in-domain) or
        ``"out-of-domain-surge"`` (Surge, out-of-domain) — or a path to a local
        checkpoint; remaining keyword arguments are forwarded.

        Args:
            *args: Positional arguments forwarded to :func:`load_model`.
            **kwargs: Keyword arguments forwarded to :func:`load_model`.

        Returns:
            A :class:`SynthRL` with pretrained weights, in eval mode.

        Note:
            Requires the optional ``synthrl-pretrained`` extra (torch, for the
            one-time PyTorch -> safetensors conversion).
        """
        # Lazy import: the pretrained loader lives in the optional `synapse.pretrained`
        # subpackage (torch/download), kept out of the base import path.
        from synapse.pretrained.synthrl import load_model

        return load_model(*args, **kwargs)
