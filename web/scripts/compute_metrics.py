#!/usr/bin/env python3
"""SynAPSE - Precomputed retrieval metrics.

Reads an ``embeddings.json`` file and writes a ``metrics.json`` file containing
cross-modal recall@k for the full (unfiltered) dataset. The web app loads this
file to display retrieval metrics instantly on page load, without waiting for
the in-browser distance matrix or the metrics worker.

The recall definition here mirrors ``src/workers/metricsWorker.ts`` exactly:
for each query, candidates of the opposite modality are ranked by ascending
cross-modal cosine distance (ties broken by dataset order, matching a stable
argsort), and the 1-indexed position of the ground-truth pair is the rank.

Usage (run from the repository root):
    python scripts/compute_metrics.py \
        --input public/data/embeddings.json \
        --output public/data/metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

RECALL_KS = (1, 5, 10, 20)


def _unit_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize each row so that a dot product yields cosine similarity.

    Args:
        embeddings: Array of shape ``(n, dim)``.

    Returns:
        Row-normalized array of the same shape.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def _direction_metrics(
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    true_candidate_index: np.ndarray,
) -> dict[str, float | int]:
    """Compute recall@k for one retrieval direction.

    Args:
        query_embeddings: Normalized query embeddings, shape ``(n_query, dim)``.
        candidate_embeddings: Normalized candidate embeddings, shape
            ``(n_candidate, dim)``.
        true_candidate_index: For each query, the column index of its
            ground-truth pair in ``candidate_embeddings``, or ``-1`` when the
            query has no pair among the candidates (such queries are excluded).

    Returns:
        A dict with ``recall_at_1/5/10/20`` (floats in ``[0, 1]``) and
        ``sample_size`` (the number of valid queries).
    """
    valid = true_candidate_index >= 0
    valid_queries = query_embeddings[valid]
    valid_true_index = true_candidate_index[valid]

    # Cross-modal cosine distance = 1 - cosine similarity.
    distances = 1.0 - valid_queries @ candidate_embeddings.T

    # Stable argsort puts ties in candidate order, matching the worker's sort.
    # The inverse permutation gives each candidate's rank position directly.
    order = np.argsort(distances, axis=1, kind="stable")
    inverse = np.argsort(order, axis=1, kind="stable")
    ranks = inverse[np.arange(valid_true_index.shape[0]), valid_true_index] + 1

    sample_size = int(ranks.shape[0])
    metrics: dict[str, float | int] = {"sample_size": sample_size}
    for k in RECALL_KS:
        hits = int(np.count_nonzero(ranks <= k))
        metrics[f"recall_at_{k}"] = hits / sample_size if sample_size > 0 else 0.0
    return metrics


def compute_metrics(points: list[dict]) -> dict[str, dict]:
    """Compute both cross-modal retrieval directions for the full dataset.

    Args:
        points: Parsed ``embeddings.json`` contents.

    Returns:
        A dict with ``audio_to_preset`` and ``preset_to_audio`` metric blocks.
    """
    audio_points = [p for p in points if p["modality"] == "audio"]
    preset_points = [p for p in points if p["modality"] == "preset"]

    audio_embeddings = _unit_normalize(
        np.asarray([p["embedding"] for p in audio_points], dtype=np.float32)
    )
    preset_embeddings = _unit_normalize(
        np.asarray([p["embedding"] for p in preset_points], dtype=np.float32)
    )

    # First candidate column seen for each pair_id, matching the worker's
    # ``Array.find`` for the paired point.
    preset_col_by_pair: dict[str, int] = {}
    for col, p in enumerate(preset_points):
        preset_col_by_pair.setdefault(p["pair_id"], col)
    audio_col_by_pair: dict[str, int] = {}
    for col, p in enumerate(audio_points):
        audio_col_by_pair.setdefault(p["pair_id"], col)

    audio_true_preset = np.asarray(
        [preset_col_by_pair.get(p["pair_id"], -1) for p in audio_points]
    )
    preset_true_audio = np.asarray(
        [audio_col_by_pair.get(p["pair_id"], -1) for p in preset_points]
    )

    return {
        "audio_to_preset": _direction_metrics(
            audio_embeddings, preset_embeddings, audio_true_preset
        ),
        "preset_to_audio": _direction_metrics(
            preset_embeddings, audio_embeddings, preset_true_audio
        ),
    }


def main() -> None:
    """Parse arguments, compute metrics, and write ``metrics.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("public/data/embeddings.json"),
        help="Path to the input embeddings.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/data/metrics.json"),
        help="Path to the output metrics.json file.",
    )
    args = parser.parse_args()

    points = json.loads(args.input.read_text())
    metrics = compute_metrics(points)
    args.output.write_text(json.dumps(metrics, indent=2))

    for direction, block in metrics.items():
        print(
            f"{direction}: "
            f"R@1={block['recall_at_1']:.3f} "
            f"R@5={block['recall_at_5']:.3f} "
            f"R@10={block['recall_at_10']:.3f} "
            f"R@20={block['recall_at_20']:.3f} "
            f"(n={block['sample_size']})"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
