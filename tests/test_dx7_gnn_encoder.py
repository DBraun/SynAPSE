"""Tests for the standalone DX7 GNN encoder.

These exercise the full forward path (import, construction, output shapes), the
core output-level gating inductive bias, scan/loop equivalence, and the feature
extraction helpers.
"""

import jax
import numpy as np
import pytest
from audiotree import AudioTree
from flax import nnx
from jax import numpy as jnp

from synapse import DX7GNNEncoder
from synapse.param_encoders.dx7_features import (
    NODE_INPUT_DIM,
    NODE_INPUT_DIM_NO_OL,
    extract_node_features,
    extract_node_features_no_ol,
    extract_output_levels,
)
from synapse.param_encoders.dx7gnn.build_graph_edges import build_graph_edges

OUT_FEATURES = 32
HIDDEN_DIM = 16
NUM_LAYERS = 3

# argbind config that reconstructs the same small architecture as _make_encoder,
# used to test the argbind-config-driven load path (from_checkpoint / from_pretrained).
SMALL_CONFIG = {
    "DX7GNNEncoder.hidden_dim": HIDDEN_DIM,
    "DX7GNNEncoder.out_features": OUT_FEATURES,
    "DX7GNNEncoder.num_layers": NUM_LAYERS,
    "DX7GNNEncoder.input_ratios": [1],
    "DX7GNNEncoder.output_ratios": [1],
}


def _make_audio_tree(
    params: jax.Array,
    algorithm: jax.Array,
    num_channels: int = 1,
    n_samples: int = 512,
) -> AudioTree:
    """Build an AudioTree carrying DX7 params/algorithm in its extras."""
    batch_size = params.shape[0]
    waveform = jnp.zeros((batch_size, num_channels, n_samples))
    return AudioTree(
        waveform,
        sample_rate=44100,
        extras={"params": params, "algorithm": algorithm},
    )


def _make_encoder(rngs: nnx.Rngs, **kwargs) -> DX7GNNEncoder:
    """Small encoder for fast tests; eval mode so dropout/masking are off."""
    defaults = dict(
        hidden_dim=HIDDEN_DIM,
        out_features=OUT_FEATURES,
        num_layers=NUM_LAYERS,
        input_ratios=[1],
        output_ratios=[1],
    )
    defaults.update(kwargs)
    encoder = DX7GNNEncoder(rngs=rngs, **defaults)
    encoder.eval()
    return encoder


def test_construct_and_forward_single_channel():
    """A mono AudioTree yields one [B, out_features] embedding per item."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs)

    batch_size = 4
    params = jnp.full((batch_size, 145), 0.5)
    algorithm = jnp.arange(batch_size, dtype=jnp.int32) % 32
    audio_tree = _make_audio_tree(params, algorithm, num_channels=1)

    out = encoder(audio_tree)
    assert out.shape == (batch_size, OUT_FEATURES)
    assert jnp.all(jnp.isfinite(out))


def test_forward_multi_channel_returns_seven_tokens():
    """A multi-channel AudioTree returns the 6 operators + carrier-sum token."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs)

    batch_size = 2
    params = jnp.full((batch_size, 145), 0.5)
    algorithm = jnp.zeros(batch_size, dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm, num_channels=2)

    out = encoder(audio_tree)
    assert out.shape == (batch_size * 7, OUT_FEATURES)
    assert jnp.all(jnp.isfinite(out))


def test_output_level_gating_makes_other_params_irrelevant():
    """The core inductive bias: an operator whose output level is 0 contributes
    nothing, so changing its *other* parameters must not change the embedding."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs)

    op = 1  # operator to silence
    algorithm = jnp.zeros(1, dtype=jnp.int32)

    base = jnp.full((1, 145), 0.5)
    # Force operator `op`'s output level (scalar field 0, stride-6 layout) to 0.
    base = base.at[0, 63 + op].set(0.0)

    # A second preset: identical, but perturb operator `op`'s *non-OL* params
    # (its env rates/levels and other scalar fields).
    perturbed = base
    perturbed = perturbed.at[0, 15 + op * 4 : 15 + op * 4 + 4].set(0.9)  # env rates
    perturbed = perturbed.at[0, 39 + op * 4 : 39 + op * 4 + 4].set(0.1)  # env levels
    for field in range(1, 10):  # other operator scalar fields
        perturbed = perturbed.at[0, 63 + field * 6 + op].set(0.2)

    out_base = encoder(_make_audio_tree(base, algorithm))
    out_perturbed = encoder(_make_audio_tree(perturbed, algorithm))

    np.testing.assert_allclose(
        np.asarray(out_base), np.asarray(out_perturbed), atol=1e-6
    )


def test_scan_matches_loop():
    """use_scan is a memory optimization and must match the plain loop exactly."""
    params = jnp.full((3, 145), 0.5)
    algorithm = jnp.array([0, 5, 17], dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm)

    enc_loop = _make_encoder(nnx.Rngs(7), use_scan=False)
    enc_scan = _make_encoder(nnx.Rngs(7), use_scan=True)

    out_loop = enc_loop(audio_tree)
    out_scan = enc_scan(audio_tree)

    np.testing.assert_allclose(np.asarray(out_loop), np.asarray(out_scan), atol=1e-6)


def test_per_layer_weights_forward():
    """share_gnn_weights=False (independent per-layer params) also runs."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs, share_gnn_weights=False, num_layers=2)

    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([3, 9], dtype=jnp.int32)
    out = encoder(_make_audio_tree(params, algorithm))
    assert out.shape == (2, OUT_FEATURES)


@pytest.mark.parametrize("feedback_scale_mode", ["auto", "half", "zero"])
def test_feedback_scale_modes(feedback_scale_mode):
    """All feedback scaling modes construct and run."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs, feedback_scale_mode=feedback_scale_mode)

    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([0, 4], dtype=jnp.int32)  # algo 4 has cross-op feedback
    out = encoder(_make_audio_tree(params, algorithm))
    assert out.shape == (2, OUT_FEATURES)


def test_fully_connected_graph_mode():
    """The fully-connected ablation graph runs and gives finite output."""
    rngs = nnx.Rngs(0)
    encoder = _make_encoder(rngs, graph_mode="fully_connected")

    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([0, 31], dtype=jnp.int32)
    out = encoder(_make_audio_tree(params, algorithm))
    assert out.shape == (2, OUT_FEATURES)
    assert jnp.all(jnp.isfinite(out))


def test_feature_extraction_shapes_and_range():
    """Feature extractors produce the documented dims, centered in [-1, 1]."""
    params = jax.random.uniform(jax.random.key(0), (5, 145))

    feats = extract_node_features(params)
    assert feats.shape == (5, 6, NODE_INPUT_DIM)  # 49

    feats_no_ol = extract_node_features_no_ol(params)
    assert feats_no_ol.shape == (5, 6, NODE_INPUT_DIM_NO_OL)  # 48

    feats_no_fb = extract_node_features_no_ol(params, include_feedback=False)
    assert feats_no_fb.shape == (5, 6, NODE_INPUT_DIM_NO_OL - 1)  # 47

    # Inputs in [0, 1] map to [-1, 1].
    assert jnp.all(feats >= -1.0 - 1e-6) and jnp.all(feats <= 1.0 + 1e-6)

    ols = extract_output_levels(params)
    assert ols.shape == (5, 6)


def test_safetensors_roundtrip_in_place(tmp_path):
    """save_safetensors → load_safetensors into a fresh model reproduces outputs."""
    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([0, 12], dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm)

    enc = _make_encoder(nnx.Rngs(0))
    ref = enc(audio_tree)

    path = tmp_path / "model.safetensors"
    enc.save_safetensors(path)

    # A differently-initialized model of the same shape, then load the weights.
    enc2 = _make_encoder(nnx.Rngs(999))
    assert float(jnp.max(jnp.abs(enc2(audio_tree) - ref))) > 1e-4  # differs before load
    enc2.load_safetensors(path)
    enc2.eval()

    np.testing.assert_allclose(
        np.asarray(enc2(audio_tree)), np.asarray(ref), atol=1e-6
    )


def test_orbax_roundtrip_in_place(tmp_path):
    """save_orbax → load_orbax into a fresh model reproduces outputs."""
    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([1, 20], dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm)

    enc = _make_encoder(nnx.Rngs(0))
    ref = enc(audio_tree)

    ckpt_dir = tmp_path / "ckpt"
    enc.save_orbax(ckpt_dir)

    enc2 = _make_encoder(nnx.Rngs(999))
    enc2.load_orbax(ckpt_dir)
    enc2.eval()

    np.testing.assert_allclose(
        np.asarray(enc2(audio_tree)), np.asarray(ref), atol=1e-6
    )


def test_from_checkpoint_safetensors_embedded_config(tmp_path):
    """A .safetensors with an embedded argbind config rebuilds via from_checkpoint."""
    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([0, 7], dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm)

    enc = _make_encoder(nnx.Rngs(0))
    ref = enc(audio_tree)

    path = tmp_path / "model.safetensors"
    enc.save_safetensors(path, argbind_config=SMALL_CONFIG)

    loaded = DX7GNNEncoder.from_checkpoint(path)  # rebuilds architecture + loads weights
    np.testing.assert_allclose(
        np.asarray(loaded(audio_tree)), np.asarray(ref), atol=1e-6
    )


def test_save_and_from_pretrained_roundtrip(tmp_path):
    """save_pretrained writes a loadable dir; from_pretrained restores it."""
    params = jnp.full((2, 145), 0.5)
    algorithm = jnp.array([3, 30], dtype=jnp.int32)
    audio_tree = _make_audio_tree(params, algorithm)

    enc = _make_encoder(nnx.Rngs(0))
    ref = enc(audio_tree)

    out_dir = tmp_path / "pretrained"
    enc.save_pretrained(out_dir, argbind_config=SMALL_CONFIG)
    assert (out_dir / "argbind_params.yml").exists()
    assert (out_dir / "model.safetensors").exists()

    loaded = DX7GNNEncoder.from_pretrained(out_dir)
    np.testing.assert_allclose(
        np.asarray(loaded(audio_tree)), np.asarray(ref), atol=1e-6
    )


def test_build_graph_edges_shapes():
    """Algorithm-mode edges have the expected packed shapes and carrier mask."""
    algorithm = jnp.array([0, 15, 31], dtype=jnp.int32)
    feedback = jnp.full((3,), 0.5)

    edge_index, edge_weights, edge_mask, carrier_mask = build_graph_edges(
        algorithm, feedback
    )

    max_edges = 6
    assert edge_index.shape == (2, 3 * max_edges)
    assert edge_weights.shape == (3 * max_edges,)
    assert edge_mask.shape == (3 * max_edges,)
    assert carrier_mask.shape == (3, 6)
