# Adapted from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
# and https://github.com/csteinmetz1/st-ito/blob/main/st_ito/models/panns.py
# Under MIT License

import os.path

import jax
import numpy as np
from absl import logging
from audiotree import AudioTree
from einops import rearrange
from flax import nnx
from flax.nnx import BatchNorm, Dropout, Rngs
from flax.nnx.nn import dtypes
from flax.typing import Dtype, PromoteDtypeFn
from jax import numpy as jnp
from jax.typing import ArrayLike
from librosax.layers import LogMelFilterBank, SpecAugmentation, Spectrogram

from synapse.base_module import SaveLoadModule
from synapse.checkpoint import DatasetStat

from .panns_blocks import ConvBlock, Linear


class Cnn14(SaveLoadModule):
    """CNN14 audio tagging encoder with mid/side channel processing (PANNs).

    A 14-layer log-mel CNN from Kong et al. "PANNs: Large-Scale Pretrained
    Audio Neural Networks for Audio Pattern Recognition" (2020), in the ST-ITO
    stereo variant. The waveform is converted to a log-mel spectrogram,
    normalized, and passed through six ``ConvBlock`` stages (channels
    ``64 -> 128 -> 256 -> 512 -> 1024 -> 2048``) with global pooling, then two
    ``2048 -> embed_dim`` heads for the mid and side channels.

    Sample rate, mel settings, and ``embed_dim`` are constructor arguments
    (there is no single fixed configuration). Input is an ``AudioTree`` (which
    carries its sample rate and is resampled to ``self.sample_rate``), with
    ``C`` either 1 (mono) or 2 (stereo, decomposed into mid ``(L + R) / 2`` and
    side ``(L - R) / 2``). Calling the model returns a ``(mid_embed,
    side_embed)`` tuple, each of shape ``[B, embed_dim]`` (for mono the two are
    identical). In SynAPSE this backbone is trained from scratch.

    Example:

    .. code-block:: python

        from audiotree import AudioTree
        from jax import numpy as jnp
        from flax import nnx
        from synapse.backbones import Cnn14

        model = Cnn14(embed_dim=512, sample_rate=22050, window_size=2048,
                      hop_size=1024, mel_bins=128, rngs=nnx.Rngs(0))
        audio = AudioTree(jnp.zeros((1, 1, 44100)), sample_rate=44100)
        mid, side = model(audio)  # mid/side: [B, 512]
    """

    def __init__(
        self,
        embed_dim: int,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: float = 0.0,
        fmax: float | None = None,
        use_batchnorm: bool = True,
        use_specaug: bool = True,
        dropout_rate: float = 0.2,
        use_fc: bool = True,
        dropout_rate_fc: float = 0.5,
        input_norm: str = "batchnorm1d",
        input_norm_path: str | None = None,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: Rngs | None = None,
    ):
        self.sample_rate = sample_rate
        self.out_features = embed_dim
        assert input_norm in ["batchnorm1d", "batchnorm2d", "npz", "none"]
        self.input_norm = input_norm
        self.use_specaug = use_specaug
        self.dtype = dtype
        self.promote_dtype = promote_dtype
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            freeze_parameters=True,
        )
        self.n_mels = mel_bins
        self.logmel_extractor = LogMelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            top_db=None,
            freeze_parameters=True,
        )

        if self.use_specaug:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=64,
                time_stripes_num=2,
                freq_drop_width=8,
                freq_stripes_num=2,
                rngs=rngs,
            )

        if input_norm == "npz":
            # ``input_mean``/``input_std`` are DatasetStat so they are saved in the
            # checkpoint AND restored by ``from_checkpoint`` (which loads
            # Param/BatchStat/DatasetStat). A bare ``nnx.Variable`` would be saved but
            # silently skipped on restore, leaving the placeholders below and disabling
            # input normalization. The npz path only seeds the stats at construction;
            # a loaded checkpoint overwrites them, so the file is optional at load time.
            num_features = 2 * mel_bins  # Same as batchnorm2d
            if input_norm_path is not None and os.path.exists(input_norm_path):
                logging.info(f"Cnn14 will use {input_norm_path} as input_norm.")
                stats = np.load(input_norm_path)
                self.input_mean = DatasetStat(jnp.array(stats["mean"]))
                self.input_std = DatasetStat(jnp.array(stats["std"]))
            else:
                # Placeholders; a restored checkpoint supplies the trained stats.
                logging.info(
                    "Cnn14 input_norm='npz' with no stats file; using placeholder "
                    "mean/std (a restored checkpoint supplies the trained values)."
                )
                self.input_mean = DatasetStat(jnp.zeros(num_features))
                self.input_std = DatasetStat(jnp.ones(num_features))
        elif self.input_norm.startswith("batchnorm"):
            logging.info("Cnn14 will use batchnorm as input_norm.")
            self.bn0 = BatchNorm(
                num_features=mel_bins if input_norm == "batchnorm1d" else 2 * mel_bins,
                momentum=0.99,
                epsilon=1e-5,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )

        # Apply convolution blocks
        out_channels_arr = [64, 128, 256, 512, 1024, 2048]
        seq_layers = []
        in_ch = 1
        for i, out_ch in enumerate(out_channels_arr):
            seq_layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    use_batchnorm=use_batchnorm,
                    pool_size=(2, 2) if i != len(out_channels_arr) - 1 else (1, 1),
                    dtype=dtype,
                    param_dtype=param_dtype,
                    promote_dtype=promote_dtype,
                    rngs=rngs,
                )
            )
            in_ch = out_ch
            seq_layers.append(Dropout(rate=dropout_rate, rngs=rngs))
        self.conv_blocks = nnx.Sequential(*seq_layers)

        if use_fc:
            self.fc1 = Linear(
                in_features=2048,
                out_features=2048,
                dtype=dtype,
                param_dtype=param_dtype,
                rngs=rngs,
            )
            self.dropout_fc1 = Dropout(rate=dropout_rate_fc, rngs=rngs)
            self.dropout_fc2 = Dropout(rate=dropout_rate_fc, rngs=rngs)
        else:
            self.fc1 = None

        self.fc_mid = Linear(
            in_features=2048,
            out_features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.fc_side = Linear(
            in_features=2048,
            out_features=embed_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def waveform2mel(self, x: ArrayLike) -> jax.Array:
        """Convert waveform to mel spectrogram whose values have been remapped and clamped to [-1,1].

        Args:
            x: Input tensor with shape (batch_size, channels, seq_len).

        Returns:
            array of shape (batch_size, channels, time_steps, mel_bins)
        """
        # Handle both (B, C, T) and (B, T) inputs
        if x.ndim == 2:
            x = x[:, None, :]  # Add channel dimension
        B, C, seq_len = x.shape

        # Compute mid and side signals
        if C == 1:
            # Single channel - no mid/side processing
            pass
        elif C == 2:
            # Stereo - compute mid and side
            x_mid = (x[:, 0, :] + x[:, 1, :]) / 2
            x_side = (x[:, 0, :] - x[:, 1, :]) / 2
            # Stack along channel dim
            x = jnp.stack([x_mid, x_side], axis=1)  # [B, 2, T]
        else:
            raise ValueError(f"Invalid number of channels: {C}")

        # Extract logmel features
        x = self.spectrogram_extractor(x)
        # x is now (B, C, time_steps, mel_bins)
        x = self.logmel_extractor(x)
        # x is still (B, C, time_steps, mel_bins)

        x = jnp.clip(x, -80.0, 40.0)  # Clamp logmels
        x = (x + 80) / 120  # Normalize between 0 and 1
        x = (x * 2) - 1  # Normalize between -1 and 1

        return x

    def __call__(
        self,
        x: AudioTree,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute mid and side embeddings from input audio.

        Args:
            x: An ``AudioTree`` with ``C`` in ``{1, 2}`` channels. It is
                resampled to ``self.sample_rate`` before its log-mel features
                are computed, unless it carries a precomputed ``extras["mel"]``
                of shape ``[B, 2, time_steps, mel_bins]`` (channel 0 = mid,
                channel 1 = side; ``int16`` is dequantized to ``float32``), which
                bypasses feature extraction.

        Returns:
            A ``(mid_embed, side_embed)`` tuple, each of shape
            ``[B, embed_dim]``. For mono input the two arrays are identical.
        """
        input_norm = self.input_norm
        # AudioTree-only. A precomputed log-mel in extras["mel"] (shape
        # (B, 2, time_steps, mel_bins), channel 0 = mid=(L+R)/2, channel 1 =
        # side=(L-R)/2, matching waveform2mel()) bypasses feature extraction;
        # otherwise the log-mel is computed from the resampled waveform.
        if isinstance(x, AudioTree) and x.extras and "mel" in x.extras:
            x = x.extras["mel"]
            # Convert int16 to float32 if needed (storage format → computation format)
            if x.dtype == jnp.int16:
                logging.info("Converting mel from jnp.int16 to jnp.float32")
                x = x.astype(jnp.float32) / 32767.0
        else:
            x = self.waveform2mel(self.require_waveform(x))

        B, C, t_steps, n_mels = x.shape
        assert self.n_mels == n_mels

        # Promote dtype after feature extraction
        (x,) = self.promote_dtype((x,), dtype=self.dtype)

        # Apply normalization
        if input_norm == "batchnorm1d":
            x = self.bn0(x)
        elif input_norm == "batchnorm2d":
            # perform batch norm on both the mid/side separately.
            B, C, T, F = x.shape
            x = rearrange(x, "B C T F -> B T (C F)")
            x = self.bn0(x)
            x = rearrange(x, "B T (C F) -> B C T F", C=C, F=F)
        elif input_norm == "npz":
            # Fixed normalization from pre-computed statistics
            B, C, T, F = x.shape
            x = rearrange(x, "B C T F -> B T (C F)")
            x = (x - self.input_mean[...]) / self.input_std[...]
            x = rearrange(x, "B T (C F) -> B C T F", C=C, F=F)
        elif input_norm == "none":
            pass
        else:
            raise ValueError(f"Invalid input_norm: {input_norm}")

        if self.use_specaug:
            x = self.spec_augmenter(x)

        x = rearrange(
            x,
            "B C t_steps n_mels -> (B C) t_steps n_mels 1",
            t_steps=t_steps,
            n_mels=n_mels,
        )

        # Apply convolution blocks
        x = self.conv_blocks(x)

        # Global pooling
        x = jnp.mean(x, axis=2)  # Mean over frequency dimension

        # Combination of max and mean pooling over time
        x1 = jnp.max(x, axis=1)  # Max over time
        x2 = jnp.mean(x, axis=1)  # Mean over time
        x = x1 + x2  # [B*C, 2048]

        # Apply fc1 with dropout and relu
        if self.fc1:
            x = self.dropout_fc1(x)
            x = nnx.relu(self.fc1(x))
            x = self.dropout_fc2(x)

        # Reshape back to separate batch and channel dimensions
        x = rearrange(x, "(B C) D -> B C D", B=B, C=C, D=2048)  # [B, C, 2048]

        # Apply final projections
        if C == 1:
            x_mid = x[:, 0, :]
            mid_embed = self.fc_mid(x_mid)
            side_embed = mid_embed
        elif C == 2:
            x_mid = x[:, 0, :]
            x_side = x[:, 1, :]
            mid_embed = self.fc_mid(x_mid)
            side_embed = self.fc_side(x_side)
        else:
            raise ValueError(f"Invalid number of channels: {C}")

        return mid_embed, side_embed
