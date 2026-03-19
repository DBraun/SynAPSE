"""SynthRL encoder wrapper with frozen/fine_tune/scratch modes for SynAPSE.

In frozen mode, reads precomputed encoder memory from extras.
In fine_tune mode, runs the SynthRL encoder from pretrained weights.
In scratch mode, builds a fresh SynthRL encoder (no pretrained weights, no torch).
All modes use a trainable AttentionPool to reduce the sequence to a fixed-dim vector.
"""

import argbind
import jax
from audiotree import AudioTree
from flax import nnx
from jax import lax
from jax import numpy as jnp

from synapse.backbones import PresetIndexesHelper, SynthRL
from synapse.layers.pooling import AttentionPool


@argbind.bind()
class SynthRLWrapper(nnx.Module):
    """SynthRL encoder wrapper with frozen/fine_tune modes.

    Args:
        mode: "frozen" reads precomputed encoder memory from extras;
            "fine_tune" runs the full SynthRL encoder from pretrained weights.
        out_features: Output dimensionality after attention pooling.
        checkpoint_path: Path to SynthRL PyTorch checkpoint. Defaults to
            the bundled in-domain-dexed checkpoint.
        rngs: Random number generators.
    """

    def __init__(
        self,
        mode: str = "fine_tune",
        out_features: int = 512,
        checkpoint_path: str = "in-domain-dexed",
        pool: str = "attention",
        cnn_dropout: float = 0.0,
        rngs: nnx.Rngs = None,
    ):
        self.mode = mode
        self.pool = pool
        self.out_features = out_features

        if mode == "fine_tune":
            # Lazy import: the pretrained loader lives in the optional
            # `synapse.pretrained` subpackage (torch/download for the one-time
            # PyTorch -> safetensors conversion), kept out of the base import path.
            from synapse.pretrained.synthrl import load_model

            self.synthrl = load_model(
                checkpoint_path, encoder_only=True, cnn_dropout=cnn_dropout
            )
            synth_rl_features = self.synthrl.d_model
        elif mode == "scratch":
            self.synthrl = SynthRL(
                PresetIndexesHelper(nb_params=144),
                rngs=rngs,
                encoder_only=True,
                cnn_dropout=cnn_dropout,
            )
            synth_rl_features = self.synthrl.d_model
        elif mode == "frozen":
            # do nothing, assume the memory latents are already in the AudioTree passed to __call__
            synth_rl_features = 512
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Options are: 'frozen', 'fine_tune', 'scratch'"
            )

        if pool == "attention":
            # Attention pool is always trainable (both modes)
            self.attn_pool = AttentionPool(
                in_features=synth_rl_features, out_features=out_features, rngs=rngs
            )
        elif pool == "mean":
            self.out_linear = nnx.Linear(
                in_features=synth_rl_features, out_features=out_features, rngs=rngs
            )
        else:
            raise ValueError(f"Unknown pool: {pool}")

    def __call__(self, audio_tree: AudioTree, **kwargs) -> jax.Array:
        """Encode audio to a fixed-dim vector.

        Args:
            audio_tree: AudioTree batch.
                - frozen mode: requires extras["synthrl_memory"] (B, seq_len, 512)
                - fine_tune mode: uses extras["mel"] (B, n_mels, n_frames, 1)
                  if available, otherwise computes mel from raw audio via
                  SynthRL's preprocess_audio (for inference/ITO use cases).

        Returns:
            (B, out_features) embedding.
        """
        if self.mode == "frozen":
            memory = audio_tree.extras["synthrl_memory"]  # (B, H*W, 512)
            if memory.dtype == jnp.float16:
                memory = memory.astype(jnp.float32)
            elif memory.dtype == jnp.int16:
                raise ValueError("Memory dtype must be float16 or float32.")

            memory = lax.stop_gradient(memory)
        else:
            # Uses audio_tree.extras["mel"] or computes mel on the fly via SynthRL's preprocess_audio
            _, _, memory = self.synthrl.encode(audio_tree)

        if self.pool == "attention":
            out = self.attn_pool(memory)  # (B, out_features)
        elif self.pool == "mean":
            out = jnp.mean(memory, axis=-2)  # (B, synthrl_features)
            out = self.out_linear(out)  # (B, out_features)
        else:
            raise RuntimeError(f"Unknown pool: {self.pool}")

        return out
