"""Download pretrained SynthRL models from GitHub releases.

This is a JAX/Flax NNX port of SynthRL, originally implemented in PyTorch.

Original repository: https://github.com/argaaw/SynthRL
License: MIT, Copyright (c) 2025 Wonchul Shin — the released checkpoints this
    module downloads are MIT as well, per the upstream README.

Reference:
    Shin, W., & Lee, K. (2025). Cross-domain Synthesizer Sound Matching via
    Reinforcement Learning. In Proceedings of the International Joint
    Conference on Artificial Intelligence (IJCAI).

SynthRL checkpoints are hosted at:
https://github.com/argaaw/SynthRL/releases/tag/v1.0.0

Note: The release files are named .tar but are actually PyTorch checkpoints (zip format).
"""

import urllib.request
from pathlib import Path

from absl import logging

from synapse.utils import get_cache_dir as _get_cache_root

# GitHub release URL for SynthRL checkpoints
GITHUB_RELEASE_BASE = "https://github.com/argaaw/SynthRL/releases/download/v1.0.0"

# Pretrained model configurations
# Note: Files are named .tar but are actually PyTorch checkpoints
PRETRAINED_MODELS = {
    "in-domain-dexed": {
        "remote_filename": "in-domain-dexed.tar",
        "local_filename": "in-domain-dexed.pt",
        "description": "SynthRL trained on Dexed synth (in-domain evaluation)",
    },
    "out-of-domain-surge": {
        "remote_filename": "out-of-domain-surge.tar",
        "local_filename": "out-of-domain-surge.pt",
        "description": "SynthRL trained on Surge synth (out-of-domain evaluation)",
    },
}


def get_cache_dir() -> Path:
    """Get cache directory for SynthRL models."""
    cache_dir = _get_cache_root() / "synthrl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, output_path: Path) -> Path:
    """Download a file from URL.

    Args:
        url: URL to download from
        output_path: Where to save the file

    Returns:
        Path to downloaded file
    """
    logging.info(f"Downloading {url}")
    logging.info(f"  -> {output_path}")

    # Download with progress
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\r  Progress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook)
    print()  # Newline after progress

    return output_path


def download_checkpoint(model_key: str, force: bool = False) -> Path:
    """Download a SynthRL checkpoint.

    Args:
        model_key: Key from PRETRAINED_MODELS (e.g., "in-domain-dexed")
        force: If True, re-download even if cached

    Returns:
        Path to checkpoint file

    Raises:
        ValueError: If model_key is unknown
    """
    if model_key not in PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown model: {model_key}. "
            f"Available: {list(PRETRAINED_MODELS.keys())}"
        )

    model_info = PRETRAINED_MODELS[model_key]
    remote_filename = model_info["remote_filename"]
    local_filename = model_info["local_filename"]

    cache_dir = get_cache_dir()
    output_path = cache_dir / local_filename
    safetensors_path = output_path.with_suffix(".safetensors")

    # Cached, converted weights (torch-free load).
    if safetensors_path.exists() and not force:
        logging.info(f"Model already cached at {safetensors_path}")
        return safetensors_path

    # Download the raw PyTorch checkpoint (remote is .tar but it's a PyTorch checkpoint).
    if not output_path.exists() or force:
        url = f"{GITHUB_RELEASE_BASE}/{remote_filename}"
        download_file(url, output_path)

    # Convert once to a flat float32 NumPy safetensors so loads need no torch.
    import torch

    from .torch_convert import dump_state_dict_to_safetensors

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    dump_state_dict_to_safetensors(state_dict, safetensors_path)
    # Keep the native checkpoint alongside so parity tests can load the reference SynthRL model.
    return safetensors_path


def download_all_checkpoints(force: bool = False) -> dict:
    """Download all SynthRL checkpoints.

    Args:
        force: If True, re-download even if cached

    Returns:
        Dictionary mapping model keys to checkpoint paths
    """
    logging.info("Downloading all SynthRL checkpoints...")
    paths = {}

    for model_key in PRETRAINED_MODELS:
        try:
            paths[model_key] = download_checkpoint(model_key, force=force)
        except Exception as e:
            logging.warning(f"Failed to download {model_key}: {e}")

    logging.info(f"\nDownloaded {len(paths)} / {len(PRETRAINED_MODELS)} checkpoints")
    return paths


if __name__ == "__main__":
    import sys

    logging.set_verbosity(logging.INFO)

    if len(sys.argv) > 1:
        model_key = sys.argv[1]
        if model_key == "all":
            paths = download_all_checkpoints()
            print("\nDownloaded checkpoints:")
            for key, path in paths.items():
                print(f"  {key}: {path}")
        else:
            path = download_checkpoint(model_key)
            print(f"\nSuccessfully downloaded to: {path}")
    else:
        logging.info(
            "Usage: python -m synapse.pretrained.synthrl_download <model_key|all>"
        )
        logging.info(f"Available models: {list(PRETRAINED_MODELS.keys())}")
