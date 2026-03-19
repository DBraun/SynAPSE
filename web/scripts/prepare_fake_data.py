#!/usr/bin/env python3
"""
SynAPSE - Synthesizer Audio Preset Embeddings
Data Preparation Pipeline

This script generates mock embedding data for development and testing.

Usage (run from the repository root):
    python scripts/prepare_fake_data.py --mode mock --output public/data/embeddings.json
"""

import argparse
import json
import random
from pathlib import Path
import numpy as np

# DX7 parameter scales (from dexed-py Preset._SCALES)
SCALES = {
    "feedback": 7,
    "transpose": 48,
    "pitch_mod_sensitivity": 7,
    "lfo_speed": 99,
    "lfo_delay": 99,
    "lfo_pitch_mod_depth": 99,
    "lfo_amp_mod_depth": 99,
    "pitch_env_rates": 99,
    "pitch_env_levels": 99,
    "op_env_rates": 99,
    "op_env_levels": 99,
    "op_output_level": 99,
    "op_frequency_coarse": 31,
    "op_frequency_fine": 99,
    "op_detune": 14,
    "op_velocity_sensitivity": 7,
    "op_amp_mod_sensitivity": 3,
    "op_rate_scaling": 7,
    "op_breakpoint": 99,
    "op_left_depth": 99,
    "op_right_depth": 99,
}

# DX7 algorithm carrier definitions (0-indexed)
ALGORITHM_CARRIERS = {
    0: [0, 2], 1: [0, 2], 2: [0, 3], 3: [0, 3], 4: [0, 2, 4],
    5: [0, 2, 4], 6: [0, 2], 7: [0, 2], 8: [0, 2], 9: [0, 3],
    10: [0, 3], 11: [0, 2], 12: [0, 2], 13: [0, 2], 14: [0, 2],
    15: [0], 16: [0], 17: [0], 18: [0, 3, 4], 19: [0, 1, 3],
    20: [0, 1, 3, 4], 21: [0, 2, 3, 4], 22: [0, 1, 3, 4],
    23: [0, 1, 2, 3, 4], 24: [0, 1, 2, 3, 4], 25: [0, 1, 3],
    26: [0, 1, 3], 27: [0, 2, 5], 28: [0, 1, 2, 4], 29: [0, 1, 2, 5],
    30: [0, 1, 2, 3, 4], 31: [0, 1, 2, 3, 4, 5],
}

DX7_PRESET_NAMES = [
    "BRASS   1", "BRASS   2", "BRASS   3", "E.PIANO 1", "E.PIANO 2",
    "E.PIANO 3", "STRINGS 1", "STRINGS 2", "ORGAN   1", "ORGAN   2",
    "CLAV    1", "HARPSICH", "MARIMBA 1", "BELLS   1", "BELLS   2",
    "FLUTE   1", "FLUTE   2", "OBOE    1", "CLARINET", "BASSOON 1",
    "SAX     1", "SAX     2", "TRUMPET 1", "TROMBONE", "TUBA    1",
    "CELESTE 1", "VIBES   1", "GUITAR  1", "GUITAR  2", "BASS    1",
    "BASS    2", "SYNBASS 1", "SYNBASS 2", "SYNLEAD 1", "SYNLEAD 2",
    "SYNPAD  1", "SYNPAD  2", "CHOIR   1", "CHOIR   2", "VOX     1",
    "KALIMBA 1", "STEEL DR", "TOM     1", "WHISTLE 1", "SFX     1",
    "INIT VOI", "PLUCK   1", "PLUCK   2", "MALLET  1", "PERC    1",
]


def generate_dx7_preset(algorithm: int | None = None) -> list[float]:
    """Generate a random but valid 145-element DX7 preset array.

    Format matches dexed-py Preset.to_array():
    - Indices 0-122: continuous params normalized [0,1]
    - Indices 123-144: integer params as float
    """
    preset = [0.0] * 145

    # Global continuous (indices 0-14)
    preset[0] = random.random()  # feedback
    preset[1] = 0.5  # transpose (center = C3)
    preset[2] = random.random()  # pitch_mod_sensitivity
    preset[3] = random.uniform(0.1, 0.7)  # lfo_speed
    preset[4] = random.uniform(0, 0.5)  # lfo_delay
    preset[5] = random.uniform(0, 0.3)  # lfo_pitch_mod_depth
    preset[6] = random.uniform(0, 0.3)  # lfo_amp_mod_depth

    # Pitch envelope rates/levels (indices 7-14)
    for i in range(4):
        preset[7 + i] = random.uniform(0.5, 1.0)  # rates
        preset[11 + i] = 0.5  # levels (center)

    # Per-operator continuous (6 ops x 18 params each, indices 15-122)
    algo = algorithm if algorithm is not None else random.randint(0, 31)
    carriers = ALGORITHM_CARRIERS.get(algo, [0])

    for op in range(6):
        base = 15 + op * 18
        # Envelope rates
        preset[base + 0] = random.uniform(0.7, 1.0)  # attack
        preset[base + 1] = random.uniform(0.3, 0.8)  # decay
        preset[base + 2] = random.uniform(0.2, 0.7)  # sustain
        preset[base + 3] = random.uniform(0.5, 1.0)  # release
        # Envelope levels
        preset[base + 4] = random.uniform(0.8, 1.0)  # L1
        preset[base + 5] = random.uniform(0.5, 1.0)  # L2
        preset[base + 6] = random.uniform(0.3, 0.9)  # L3
        preset[base + 7] = 0.0  # L4 (release level)
        # Output level - carriers louder
        if op in carriers:
            preset[base + 8] = random.uniform(0.6, 1.0)
        else:
            preset[base + 8] = random.uniform(0.3, 1.0)
        # Frequency coarse
        preset[base + 9] = random.choice([1/31, 2/31, 3/31, 4/31, 5/31, 6/31, 7/31, 8/31])
        # Frequency fine
        preset[base + 10] = random.uniform(0, 0.1)
        # Detune (0.5 = center = 0 detune)
        preset[base + 11] = random.gauss(0.5, 0.1)
        preset[base + 11] = max(0, min(1, preset[base + 11]))
        # Other params
        preset[base + 12] = random.uniform(0, 0.5)  # velocity sens
        preset[base + 13] = random.uniform(0, 0.3)  # amp mod sens
        preset[base + 14] = 0.0  # rate scaling
        preset[base + 15] = 39.0 / 99.0  # breakpoint
        preset[base + 16] = 0.0  # left depth
        preset[base + 17] = 0.0  # right depth

    # Integer params (indices 123-144)
    preset[123] = 1.0  # osc_key_sync
    preset[124] = 0.0  # lfo_sync
    preset[125] = float(algo)  # algorithm (0-31)
    preset[126] = float(random.randint(0, 5))  # lfo_wave

    # Per-operator integers (indices 127-144)
    for op in range(6):
        int_base = 127 + op * 3
        preset[int_base + 0] = 0.0  # freq_mode (ratio)
        preset[int_base + 1] = 0.0  # left_curve
        preset[int_base + 2] = 0.0  # right_curve

    return preset


def extract_dx7_tags(preset: list[float]) -> list[str]:
    """Extract categorization tags from a DX7 preset."""
    tags = []

    algo = int(round(preset[125]))
    tags.append(f"algorithm_{algo + 1}")

    carriers = ALGORITHM_CARRIERS.get(algo, [0])
    tags.append(f"carriers_{len(carriers)}")

    feedback = round(preset[0] * 7)
    if feedback == 0:
        tags.append("no_feedback")
    elif feedback <= 3:
        tags.append("low_feedback")
    else:
        tags.append("high_feedback")

    # Check for fixed frequency operators
    has_fixed = False
    for op in range(6):
        if round(preset[127 + op * 3]) == 1:
            has_fixed = True
            break
    if has_fixed:
        tags.append("has_fixed_freq")

    # LFO characteristics
    lfo_speed = round(preset[3] * 99)
    lfo_pmd = round(preset[5] * 99)
    lfo_amd = round(preset[6] * 99)
    if lfo_speed == 0 and lfo_pmd == 0 and lfo_amd == 0:
        tags.append("no_lfo")
    elif lfo_speed < 30:
        tags.append("slow_lfo")
    elif lfo_speed > 60:
        tags.append("fast_lfo")

    return tags


def generate_mock_embeddings(n_pairs: int = 2500, embedding_dim: int = 128) -> list[dict]:
    """Generate mock embedding data with DX7 presets."""
    data = []

    # Generate cluster centers (one per algorithm group)
    n_clusters = 32
    cluster_centers = np.random.randn(n_clusters, embedding_dim) * 3

    for i in range(n_pairs):
        pair_id = f"pair_{i:05d}"

        # Pick algorithm (cluster by algorithm)
        algorithm = i % n_clusters
        center = cluster_centers[algorithm]

        # Generate DX7 preset
        preset = generate_dx7_preset(algorithm)
        tags = extract_dx7_tags(preset)
        preset_name = random.choice(DX7_PRESET_NAMES)

        # Generate embeddings
        shared_signal = np.random.randn(embedding_dim)

        audio_embedding = (
            center +
            shared_signal * 0.5 +
            np.random.randn(embedding_dim) * 1.2
        )

        preset_embedding = (
            center +
            shared_signal * 0.5 +
            np.random.randn(embedding_dim) * 1.2
        )

        # 2D positions (cluster by algorithm)
        cluster_2d = np.random.randn(2) * 5
        cluster_2d[0] += (algorithm % 8) * 15
        cluster_2d[1] += (algorithm // 8) * 15

        audio_2d = cluster_2d + np.random.randn(2) * 2
        preset_2d = cluster_2d + np.random.randn(2) * 2

        # Audio point
        data.append({
            "id": f"audio_{i:05d}",
            "modality": "audio",
            "dx7_preset": preset,
            "preset_name": preset_name,
            "embedding": audio_embedding.tolist(),
            "x": float(audio_2d[0]),
            "y": float(audio_2d[1]),
            "pair_id": pair_id,
            "tags": tags,
        })

        # Preset point
        data.append({
            "id": f"preset_{i:05d}",
            "modality": "preset",
            "dx7_preset": preset,
            "preset_name": preset_name,
            "embedding": preset_embedding.tolist(),
            "x": float(preset_2d[0]),
            "y": float(preset_2d[1]),
            "pair_id": pair_id,
            "tags": tags,
        })

    return data


def main():
    parser = argparse.ArgumentParser(description="Prepare embedding data for SynAPSE")
    parser.add_argument("--mode", choices=["mock"], default="mock",
                       help="Generate mock data")
    parser.add_argument("--output", type=Path, default=Path("public/data/embeddings.json"),
                       help="Output JSON file (relative to the repository root)")
    parser.add_argument("--n-pairs", type=int, default=2500,
                       help="Number of audio-preset pairs")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Embedding dimension")

    args = parser.parse_args()

    print(f"Generating {args.n_pairs} mock audio-preset pairs...")
    data = generate_mock_embeddings(args.n_pairs, args.embedding_dim)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(data, f)

    print(f"Wrote {len(data)} points to {args.output}")

    modalities = {}
    all_tags = set()
    for point in data:
        modalities[point["modality"]] = modalities.get(point["modality"], 0) + 1
        all_tags.update(point["tags"])

    print(f"Modalities: {modalities}")
    print(f"Unique tags: {len(all_tags)}")
    print(f"Sample tags: {list(all_tags)[:10]}")


if __name__ == "__main__":
    main()
