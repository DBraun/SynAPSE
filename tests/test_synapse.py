"""Tests for the full SynAPSE model and its audio/param encoders.

All paths here are torch-free: the PANNs backbone is trained from scratch and the
SynthRL backbone runs in ``scratch`` mode (the pretrained ``fine_tune`` path needs
the optional ``synthrl-pretrained`` extra and network access, so it is not tested
here).
"""

import argbind
import numpy as np
import pytest
from audiotree import AudioTree
from flax import nnx
from jax import numpy as jnp

from synapse import SynAPSE, SynAPSEOutput
from synapse.audio_encoders import SynthRLWrapper
from synapse.slap import SLAP

# A small SynAPSE (via argbind scope) so tests stay fast. SynthRL runs in scratch
# mode; every dimension is shrunk from its default.
SMALL_SCOPE = {
    "SynthRLWrapper.mode": "scratch",
    "SynthRLWrapper.out_features": 48,
    "SynthRL.d_model": 32,
    "SynthRL.nhead": 4,
    "SynthRL.num_encoder_layers": 2,
    "SynthRL.dim_feedforward": 64,
    "PANNsWrapper.output_dim": 48,
    "DX7GNNEncoder.hidden_dim": 32,
    "DX7GNNEncoder.out_features": 48,
    "DX7GNNEncoder.num_layers": 2,
    "DX7GNNEncoder.input_ratios": [1],
    "DX7GNNEncoder.output_ratios": [1],
    "FMMessagePassing.hidden_dim": 32,
    "FlatParamEncoder.hidden_dims": [64, 48],
    "TransformerParamEncoder.hidden_dim": 32,
    "TransformerParamEncoder.out_features": 48,
    "TransformerParamEncoder.num_layers": 2,
    "TransformerParamEncoder.num_heads": 4,
    "SLAPConfig.embed_dim": 32,
    "SLAPConfig.pred_hidden_dim": 64,
}

# Waveform long enough to survive the backbones' time-downsampling.
N_SAMPLES = 96000


def _audio_tree(batch_size: int = 2) -> AudioTree:
    rng = np.random.RandomState(0)
    wav = jnp.asarray(rng.randn(batch_size, 1, N_SAMPLES).astype("float32")) * 0.1
    return AudioTree(
        wav,
        sample_rate=44100,
        extras={
            "params": jnp.asarray(rng.rand(batch_size, 145).astype("float32")),
            "algorithm": jnp.arange(batch_size, dtype=jnp.int32) % 32,
        },
    )


def _build_synapse(**kwargs) -> SynAPSE:
    with argbind.scope(SMALL_SCOPE):
        model = SynAPSE(rngs=nnx.Rngs(0), **kwargs)
    model.eval()
    return model


@pytest.mark.parametrize("param_encoder", ["gnn", "flat", "transformer"])
def test_synapse_synthrl_scratch_forward(param_encoder):
    """SynAPSE with the (scratch) SynthRL backbone and each param encoder."""
    model = _build_synapse(audio_encoder="synthrl", encoder=param_encoder)
    at = _audio_tree()
    out = model(at)
    assert isinstance(out, SynAPSEOutput)
    B = at.waveform.shape[0]
    for arm in (out.audio_output, out.parameter_output):
        assert arm.latent.shape[0] == B
        assert arm.projection.shape == (B, SMALL_SCOPE["SLAPConfig.embed_dim"])
        assert arm.prediction.shape == (B, SMALL_SCOPE["SLAPConfig.embed_dim"])
    assert jnp.all(jnp.isfinite(out.parameter_output.projection))


def test_synapse_panns_forward():
    """SynAPSE with the PANNs (Cnn14) backbone, trained from scratch."""
    model = _build_synapse(audio_encoder="panns", encoder="gnn")
    out = model(_audio_tree())
    assert out.audio_output.projection.shape == (2, SMALL_SCOPE["SLAPConfig.embed_dim"])
    assert jnp.all(jnp.isfinite(out.audio_output.projection))


def test_synapse_forward_batches():
    """forward_batches scans mini-batches and matches a single-shot forward pass."""
    at = _audio_tree(4)
    model = _build_synapse(audio_encoder="synthrl", encoder="gnn")
    ref = model(at)

    out = model.forward_batches(at, mini_batch_size=2)
    assert isinstance(out, SynAPSEOutput)
    B = at.waveform.shape[0]
    for arm in (out.audio_output, out.parameter_output):
        assert arm.latent.shape[0] == B
        assert arm.projection.shape == (B, SMALL_SCOPE["SLAPConfig.embed_dim"])
        assert arm.prediction.shape == (B, SMALL_SCOPE["SLAPConfig.embed_dim"])
        assert jnp.all(jnp.isfinite(arm.projection))

    # In eval mode (no dropout, BatchNorm on running stats) the scan is exact.
    for scanned, whole in (
        (out.audio_output, ref.audio_output),
        (out.parameter_output, ref.parameter_output),
    ):
        np.testing.assert_allclose(
            np.asarray(scanned.projection), np.asarray(whole.projection), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(scanned.prediction), np.asarray(whole.prediction), atol=1e-6
        )


def test_slap_forward_batches_arm_kwargs():
    """The underlying SLAP.forward_batches takes per-arm kwargs positionally or by name."""
    at = _audio_tree(4)
    model = _build_synapse(audio_encoder="synthrl", encoder="gnn")

    by_name = SLAP.forward_batches(model, at, 2, arm1_kwargs={}, arm2_kwargs={})
    positional = SLAP.forward_batches(model, at, 2, {}, {})
    for named, pos in zip(by_name, positional):
        assert named.projection.shape == (4, SMALL_SCOPE["SLAPConfig.embed_dim"])
        np.testing.assert_allclose(
            np.asarray(named.projection), np.asarray(pos.projection), atol=1e-6
        )

    # SynAPSE forwards its audio_kwargs/param_kwargs down to the two arms.
    out = model.forward_batches(at, 2, audio_kwargs={}, param_kwargs={})
    np.testing.assert_allclose(
        np.asarray(out.audio_output.projection),
        np.asarray(by_name[0].projection),
        atol=1e-6,
    )


def test_synapse_safetensors_roundtrip(tmp_path):
    """save_safetensors -> load into a fresh SynAPSE reproduces outputs."""
    at = _audio_tree()
    model = _build_synapse(audio_encoder="synthrl", encoder="gnn")
    ref = model(at)

    path = tmp_path / "synapse.safetensors"
    model.save_safetensors(path)

    with argbind.scope(SMALL_SCOPE):
        reloaded = SynAPSE(audio_encoder="synthrl", encoder="gnn", rngs=nnx.Rngs(123))
    reloaded.load_safetensors(path)
    reloaded.eval()
    out = reloaded(at)

    np.testing.assert_allclose(
        np.asarray(out.parameter_output.projection),
        np.asarray(ref.parameter_output.projection),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(out.audio_output.projection),
        np.asarray(ref.audio_output.projection),
        atol=1e-6,
    )


def test_synthrl_wrapper_scratch_shapes():
    """SynthRLWrapper in scratch mode pools the encoder memory to out_features."""
    with argbind.scope(SMALL_SCOPE):
        wrap = SynthRLWrapper(rngs=nnx.Rngs(0))
    wrap.eval()
    out = wrap(_audio_tree())
    assert out.shape == (2, SMALL_SCOPE["SynthRLWrapper.out_features"])


def test_unknown_encoder_raises():
    """An unknown encoder/audio_encoder name is rejected."""
    with pytest.raises(ValueError):
        SynAPSE(encoder="nope", rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        SynAPSE(audio_encoder="nope", rngs=nnx.Rngs(0))
