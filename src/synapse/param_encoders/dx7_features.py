"""Shared DX7 preset feature extraction for parameter encoders.

Extracts node features from DX7 preset flat arrays (145 dims from
``dexed.Preset.to_array()``). Used by the GNN, Transformer, and flat encoders.

Preset flat array layout:
  [0:7]     global continuous scalars (7): feedback, transpose, pitch_mod_sens,
            lfo_speed, lfo_delay, lfo_pmd, lfo_amd
  [7:15]    global continuous arrays (8): pitch_env_rates(4), pitch_env_levels(4)
  [15:63]   operator envelopes (48): op_env_rates(6,4), op_env_levels(6,4)
  [63:123]  operator scalars (60): 10 fields x (6,) each (output_level, freq_coarse,
            freq_fine, detune, vel_sens, amp_mod_sens, rate_scaling, breakpoint,
            left_depth, right_depth)
  [123:127] global integers (4): osc_key_sync, lfo_sync, algorithm, lfo_wave
  [127:145] operator integers (18): op_frequency_mode(6), op_left_curve(6), op_right_curve(6)

Transpose (index 1) is excluded from input (always centered during prerender).
Algorithm (index 125) is used for graph topology / embedding, not as a node feature.
"""

import jax
from jax import numpy as jnp
from jax.typing import ArrayLike

# Feature dimensions
GLOBAL_CONTINUOUS_DIM = 14  # 15 minus transpose
GLOBAL_BINARY_DIM = 2  # osc_key_sync, lfo_sync
GLOBAL_ONEHOT_DIM = 6  # lfo_wave (6 choices)
GLOBAL_DIM = GLOBAL_CONTINUOUS_DIM + GLOBAL_BINARY_DIM + GLOBAL_ONEHOT_DIM  # = 22

OP_CONTINUOUS_DIM = 18
OP_BINARY_DIM = 1  # freq_mode
OP_ONEHOT_DIM = 8  # left_curve (4) + right_curve (4)
OP_DIM = OP_CONTINUOUS_DIM + OP_BINARY_DIM + OP_ONEHOT_DIM  # = 27

NODE_INPUT_DIM = GLOBAL_DIM + OP_DIM  # = 49

# Dimensions with output_level removed (for GNN with explicit OL gating)
OP_CONTINUOUS_DIM_NO_OL = 17  # 18 - 1 (output_level removed)
OP_DIM_NO_OL = OP_CONTINUOUS_DIM_NO_OL + OP_BINARY_DIM + OP_ONEHOT_DIM  # = 26
NODE_INPUT_DIM_NO_OL = GLOBAL_DIM + OP_DIM_NO_OL  # = 48


def extract_global_features(
    flat_params: ArrayLike, include_feedback: bool = True
) -> jax.Array:
    """Extract global features from DX7 preset params.

    All features are mapped to [-1, 1]: continuous via 2x-1, binary {0,1} -> {-1,1},
    one-hot {0,1} -> {-1,1}. This centers all features around 0.

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().
        include_feedback: If False, drop the feedback scalar (index 0) from the global
            features (22 -> 21 dims). Used by the GNN "remove feedback global param" ablation.

    Returns:
        Global features [batch_size, 22] (14 continuous + 2 binary + 6 one-hot LFO wave),
        or [batch_size, 21] when ``include_feedback`` is False.
    """
    # Global continuous scalars [0:7], skip transpose (index 1); optionally drop feedback (index 0).
    if include_feedback:
        global_cont = jnp.concatenate(
            [
                flat_params[:, 0:1],  # feedback
                flat_params[:, 2:7],  # pitch_mod_sensitivity, lfo_speed..amp_mod_depth
            ],
            axis=-1,
        )  # [B, 6]
    else:
        global_cont = flat_params[:, 2:7]  # [B, 5]  (feedback dropped)
    # pitch_env_rates [7:11] and pitch_env_levels [11:15]
    global_env = flat_params[:, 7:15]  # [B, 8]

    # Global binary ints
    osc_key_sync = flat_params[:, 123:124]  # [B, 1]
    lfo_sync = flat_params[:, 124:125]  # [B, 1]
    # algorithm (index 125) is used for graph topology, not as node feature
    lfo_wave_idx = jnp.round(flat_params[:, 126]).astype(jnp.int32)
    lfo_wave_onehot = jax.nn.one_hot(lfo_wave_idx, 6)  # [B, 6]

    global_features = jnp.concatenate(
        [global_cont, global_env, osc_key_sync, lfo_sync, lfo_wave_onehot],
        axis=-1,
    )  # [B, 6 + 8 + 1 + 1 + 6 = 22]

    return 2 * global_features - 1


def extract_operator_features(flat_params: ArrayLike) -> jax.Array:
    """Extract per-operator features from DX7 preset params.

    All features are mapped to [-1, 1]: continuous via 2x-1, binary {0,1} -> {-1,1},
    one-hot {0,1} -> {-1,1}. This centers all features around 0.

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().

    Returns:
        Per-operator features [batch_size, 6, 27] (18 continuous + 1 binary + 8 one-hot curves).
    """
    # to_array() stores per-op params grouped by field, not by operator.
    # Each (6,K) field is flattened row-major, so op i's values are at
    # base + i*K : base + i*K + K for (6,K) fields, or base + i for (6,) fields.

    # Continuous (6, 4) fields
    op_env_rates = flat_params[:, 15:39].reshape(-1, 6, 4)  # [B, 6, 4]
    op_env_levels = flat_params[:, 39:63].reshape(-1, 6, 4)  # [B, 6, 4]

    # Continuous (6,) fields — 10 fields from indices 63..123
    op_scalars = jnp.stack(
        [flat_params[:, 63 + i * 6 : 63 + (i + 1) * 6] for i in range(10)], axis=-1
    )  # [B, 6, 10]

    op_cont = jnp.concatenate(
        [op_env_rates, op_env_levels, op_scalars], axis=-1
    )  # [B, 6, 18]

    # Integer (6,) fields
    freq_mode = flat_params[:, 127:133].reshape(-1, 6, 1)  # [B, 6, 1]
    left_curve_idx = jnp.round(flat_params[:, 133:139]).astype(jnp.int32)  # [B, 6]
    right_curve_idx = jnp.round(flat_params[:, 139:145]).astype(jnp.int32)  # [B, 6]
    left_curve_onehot = jax.nn.one_hot(left_curve_idx, 4)  # [B, 6, 4]
    right_curve_onehot = jax.nn.one_hot(right_curve_idx, 4)  # [B, 6, 4]

    op_features = jnp.concatenate(
        [
            op_cont,
            freq_mode,
            left_curve_onehot,
            right_curve_onehot,
        ],
        axis=-1,
    )  # [B, 6, 18 + 1 + 4 + 4 = 27]

    return 2 * op_features - 1


def extract_node_features(flat_params: ArrayLike) -> jax.Array:
    """Extract combined node features (global broadcast + per-operator).

    Each of the 6 operators receives the same 22-dim global features
    concatenated with its 27-dim operator-specific features, yielding
    49-dim node features.

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().

    Returns:
        Node features [batch_size, 6, 49].
    """
    global_features = extract_global_features(flat_params)  # [B, 22]
    op_features = extract_operator_features(flat_params)  # [B, 6, 27]

    # Broadcast global features to [B, 6, 22]
    global_broadcast = jnp.broadcast_to(
        global_features[:, None, :],
        (flat_params.shape[0], 6, global_features.shape[-1]),
    )

    node_features = jnp.concatenate(
        [global_broadcast, op_features], axis=-1
    )  # [B, 6, 22 + 27 = 49]

    return node_features


def extract_output_levels(flat_params: ArrayLike) -> jax.Array:
    """Extract raw output levels for each operator.

    Output level is field 0 of the 10 operator scalar fields, stored at
    flat_params[:, 63:69] (stride-6 layout, field 0 x 6 operators).

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().

    Returns:
        Output levels [batch_size, 6] in [0, 1].
    """
    return flat_params[:, 63:69]  # [B, 6]


def extract_operator_features_no_ol(flat_params: ArrayLike) -> jax.Array:
    """Extract per-operator features without output_level (for GNN with explicit OL gating).

    Same as extract_operator_features but skips field 0 (output_level) in the
    10 operator scalars, keeping only fields 1-9. Result: 17 continuous + 1 binary + 8 one-hot = 26.

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().

    Returns:
        Per-operator features [batch_size, 6, 26].
    """
    # Continuous (6, 4) fields
    op_env_rates = flat_params[:, 15:39].reshape(-1, 6, 4)  # [B, 6, 4]
    op_env_levels = flat_params[:, 39:63].reshape(-1, 6, 4)  # [B, 6, 4]

    # Continuous (6,) fields — skip field 0 (output_level), keep fields 1-9
    op_scalars_no_ol = jnp.stack(
        [flat_params[:, 63 + i * 6 : 63 + (i + 1) * 6] for i in range(1, 10)], axis=-1
    )  # [B, 6, 9]

    op_cont = jnp.concatenate(
        [op_env_rates, op_env_levels, op_scalars_no_ol], axis=-1
    )  # [B, 6, 17]

    # Integer (6,) fields (same as extract_operator_features)
    freq_mode = flat_params[:, 127:133].reshape(-1, 6, 1)  # [B, 6, 1]
    left_curve_idx = jnp.round(flat_params[:, 133:139]).astype(jnp.int32)  # [B, 6]
    right_curve_idx = jnp.round(flat_params[:, 139:145]).astype(jnp.int32)  # [B, 6]
    left_curve_onehot = jax.nn.one_hot(left_curve_idx, 4)  # [B, 6, 4]
    right_curve_onehot = jax.nn.one_hot(right_curve_idx, 4)  # [B, 6, 4]

    op_features = jnp.concatenate(
        [
            op_cont,
            freq_mode,
            left_curve_onehot,
            right_curve_onehot,
        ],
        axis=-1,
    )  # [B, 6, 17 + 1 + 4 + 4 = 26]

    return 2 * op_features - 1


def extract_node_features_no_ol(
    flat_params: ArrayLike, include_feedback: bool = True
) -> jax.Array:
    """Extract combined node features without output_level (for GNN with explicit OL gating).

    Each of the 6 operators receives the same 22-dim global features
    concatenated with its 26-dim operator-specific features (no output_level),
    yielding 48-dim node features.

    Args:
        flat_params: Flat parameter array [batch_size, 145] from Preset.to_array().
        include_feedback: If False, drop the feedback scalar from the global features,
            yielding 47-dim node features (GNN "remove feedback global param" ablation).

    Returns:
        Node features [batch_size, 6, 48], or [batch_size, 6, 47] when
        ``include_feedback`` is False.
    """
    global_features = extract_global_features(
        flat_params, include_feedback=include_feedback
    )  # [B, 22 or 21]
    op_features = extract_operator_features_no_ol(flat_params)  # [B, 6, 26]

    # Broadcast global features to [B, 6, 22]
    global_broadcast = jnp.broadcast_to(
        global_features[:, None, :],
        (flat_params.shape[0], 6, global_features.shape[-1]),
    )

    node_features = jnp.concatenate(
        [global_broadcast, op_features], axis=-1
    )  # [B, 6, 22 + 26 = 48]

    return node_features
