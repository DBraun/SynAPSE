"""Minimal PyTorch -> NumPy weight-conversion helpers for the SynthRL loader.

Only the pieces the SynthRL pretrained path needs: a state-dict normalizer, the
conv-kernel transpose, and a flat-``.safetensors`` dumper used at download time so
subsequent loads are torch-free.
"""

import json

import numpy as np


def _to_np(value) -> np.ndarray:
    """Normalize a state-dict value (``torch.Tensor`` or array-like) to a NumPy array.

    Torch tensors are moved to CPU and cast to float32; values that are already
    arrays are passed through unchanged.
    """
    if hasattr(value, "cpu"):  # torch.Tensor
        return value.cpu().float().numpy()
    return np.asarray(value)


def transpose_conv_weight(weight: np.ndarray) -> np.ndarray:
    """Transpose a PyTorch conv kernel to JAX layout (Conv1d or Conv2d).

    Args:
        weight: PyTorch weight — Conv2d ``(out, in, h, w)`` or Conv1d ``(out, in, k)``.

    Returns:
        JAX weight — Conv2d ``(h, w, in, out)`` or Conv1d ``(k, in, out)``.
    """
    if weight.ndim == 4:
        return np.transpose(weight, (2, 3, 1, 0))
    if weight.ndim == 3:
        return np.transpose(weight, (2, 1, 0))
    raise ValueError(
        f"Unsupported conv kernel ndim {weight.ndim}; expected 3 (Conv1d) or 4 (Conv2d)."
    )


def dump_state_dict_to_safetensors(
    state_dict: dict, path, *, metadata: dict[str, str] | None = None
) -> None:
    """Flatten a (torch or array) state dict to float32 NumPy and write a ``.safetensors``.

    Called at download time to cache PyTorch weights as plain arrays, so subsequent
    loads read them with ``safetensors.numpy.load_file`` (fast, no torch import).
    Non-array entries are kept only if they expose a shape; everything else is skipped.

    Args:
        state_dict: Mapping of name -> tensor/array.
        path: Output ``.safetensors`` path.
        metadata: Optional string->string map stored in the file header.
    """
    from safetensors.numpy import save_file

    flat: dict[str, np.ndarray] = {}
    zero_dim: list[str] = []
    for key, value in state_dict.items():
        if not (hasattr(value, "cpu") or hasattr(value, "shape")):
            continue
        arr = _to_np(value)
        if arr.ndim == 0:  # safetensors can't store 0-dim; record for faithful restore
            zero_dim.append(key)
        arr = np.ascontiguousarray(arr)  # contiguous; also promotes 0-dim to shape (1,)
        flat[key] = arr

    meta = dict(metadata or {})
    if zero_dim:
        meta["__zero_dim__"] = json.dumps(zero_dim)
    save_file(flat, str(path), metadata=meta or None)
