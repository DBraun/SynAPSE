"""Pretrained weight loading for SynthRL.

This is a JAX/Flax NNX port of SynthRL, originally implemented in PyTorch.

Original repository: https://github.com/argaaw/SynthRL
License: MIT, Copyright (c) 2025 Wonchul Shin

Reference:
    Shin, W., & Lee, K. (2025). Cross-domain Synthesizer Sound Matching via
    Reinforcement Learning. In Proceedings of the International Joint
    Conference on Artificial Intelligence (IJCAI).

Converts PyTorch checkpoint weights to NNX format.
"""

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from safetensors.numpy import load_file

from synapse.backbones import PresetIndexesHelper, SynthRL

from .synthrl_download import PRETRAINED_MODELS, download_checkpoint
from .torch_convert import transpose_conv_weight


def torch_to_jax_linear(weight: np.ndarray) -> jax.Array:
    """Convert PyTorch Linear weight to JAX format.

    PyTorch: (out_features, in_features)
    JAX:     (in_features, out_features)
    """
    return jnp.transpose(weight, (1, 0))


def torch_to_jax_batchnorm(
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
) -> dict[str, jax.Array]:
    """Convert PyTorch BatchNorm parameters to JAX format.

    Returns dict with scale, bias, mean, var.
    """
    return {
        "scale": jnp.array(weight),
        "bias": jnp.array(bias),
        "mean": jnp.array(running_mean),
        "var": jnp.array(running_var),
    }


def torch_to_jax_multihead_attention(
    in_proj_weight: np.ndarray,
    in_proj_bias: np.ndarray,
    out_proj_weight: np.ndarray,
    out_proj_bias: np.ndarray,
    d_model: int,
    num_heads: int,
) -> dict[str, dict[str, jax.Array]]:
    """Convert PyTorch MultiheadAttention to NNX format.

    PyTorch uses fused in_proj for Q, K, V, we split and reshape them.

    PyTorch shapes:
        in_proj_weight: (3*d_model, d_model)
        in_proj_bias: (3*d_model,)
        out_proj_weight: (d_model, d_model)
        out_proj_bias: (d_model,)

    NNX shapes:
        query/key/value kernel: (d_model, num_heads, head_dim)
        query/key/value bias: (num_heads, head_dim)
        out kernel: (num_heads, head_dim, d_model)
        out bias: (d_model,)
    """
    head_dim = d_model // num_heads

    # Split in_proj into Q, K, V and reshape
    # PyTorch: (3*d, d) -> transpose to (d, 3*d) -> split -> reshape to (d, num_heads, head_dim)
    q_w = in_proj_weight[:d_model, :].T.reshape(d_model, num_heads, head_dim)
    k_w = in_proj_weight[d_model : 2 * d_model, :].T.reshape(
        d_model, num_heads, head_dim
    )
    v_w = in_proj_weight[2 * d_model :, :].T.reshape(d_model, num_heads, head_dim)

    q_b = in_proj_bias[:d_model].reshape(num_heads, head_dim)
    k_b = in_proj_bias[d_model : 2 * d_model].reshape(num_heads, head_dim)
    v_b = in_proj_bias[2 * d_model :].reshape(num_heads, head_dim)

    # Output projection: (d, d) -> transpose -> reshape to (num_heads, head_dim, d)
    out_w = out_proj_weight.T.reshape(num_heads, head_dim, d_model)

    return {
        "query": {
            "kernel": jnp.array(q_w),
            "bias": jnp.array(q_b),
        },
        "key": {
            "kernel": jnp.array(k_w),
            "bias": jnp.array(k_b),
        },
        "value": {
            "kernel": jnp.array(v_w),
            "bias": jnp.array(v_b),
        },
        "out": {
            "kernel": jnp.array(out_w),
            "bias": jnp.array(out_proj_bias),
        },
    }


def convert_pytorch_checkpoint(
    checkpoint: dict[str, Any],
    model: SynthRL,
) -> dict[str, Any]:
    """Convert PyTorch SynthRL checkpoint to NNX parameters.

    Args:
        checkpoint: PyTorch state_dict.
        model: SynthRL model (used for architecture dimensions).

    Returns:
        Dictionary of NNX parameters.
    """
    params = {}

    # Helper to get numpy array from torch tensor
    def to_numpy(key: str) -> np.ndarray:
        val = checkpoint[key]
        if hasattr(val, "numpy"):
            return val.numpy()
        return np.array(val)

    # CNN Backbone
    # Layer 0: conv only (no batchnorm)
    params["backbone"] = {
        "conv0": {
            "kernel": jnp.array(
                transpose_conv_weight(to_numpy("backbone.conv.0.0.weight"))
            ),
            "bias": jnp.array(to_numpy("backbone.conv.0.0.bias")),
        },
        # Layer 1: conv + batchnorm
        "conv1": {
            "kernel": jnp.array(
                transpose_conv_weight(to_numpy("backbone.conv.1.0.weight"))
            ),
            "bias": jnp.array(to_numpy("backbone.conv.1.0.bias")),
        },
        "bn1": torch_to_jax_batchnorm(
            to_numpy("backbone.conv.1.2.weight"),
            to_numpy("backbone.conv.1.2.bias"),
            to_numpy("backbone.conv.1.2.running_mean"),
            to_numpy("backbone.conv.1.2.running_var"),
        ),
        # Layer 2
        "conv2": {
            "kernel": jnp.array(
                transpose_conv_weight(to_numpy("backbone.conv.2.0.weight"))
            ),
            "bias": jnp.array(to_numpy("backbone.conv.2.0.bias")),
        },
        "bn2": torch_to_jax_batchnorm(
            to_numpy("backbone.conv.2.2.weight"),
            to_numpy("backbone.conv.2.2.bias"),
            to_numpy("backbone.conv.2.2.running_mean"),
            to_numpy("backbone.conv.2.2.running_var"),
        ),
        # Layer 3
        "conv3": {
            "kernel": jnp.array(
                transpose_conv_weight(to_numpy("backbone.conv.3.0.weight"))
            ),
            "bias": jnp.array(to_numpy("backbone.conv.3.0.bias")),
        },
        "bn3": torch_to_jax_batchnorm(
            to_numpy("backbone.conv.3.2.weight"),
            to_numpy("backbone.conv.3.2.bias"),
            to_numpy("backbone.conv.3.2.running_mean"),
            to_numpy("backbone.conv.3.2.running_var"),
        ),
        # Layer 4
        "conv4": {
            "kernel": jnp.array(
                transpose_conv_weight(to_numpy("backbone.conv.4.0.weight"))
            ),
            "bias": jnp.array(to_numpy("backbone.conv.4.0.bias")),
        },
        "bn4": torch_to_jax_batchnorm(
            to_numpy("backbone.conv.4.2.weight"),
            to_numpy("backbone.conv.4.2.bias"),
            to_numpy("backbone.conv.4.2.running_mean"),
            to_numpy("backbone.conv.4.2.running_var"),
        ),
    }

    # Transformer tgt (learnable queries)
    params["tgt"] = jnp.array(
        to_numpy("transformer.tgt").squeeze(1)
    )  # Remove middle dim

    # Encoder layers
    encoder_layers = []
    for i in range(model.num_encoder_layers):
        prefix = f"transformer.encoder.layers.{i}"
        layer_params = {
            "self_attn": torch_to_jax_multihead_attention(
                to_numpy(f"{prefix}.self_attn.in_proj_weight"),
                to_numpy(f"{prefix}.self_attn.in_proj_bias"),
                to_numpy(f"{prefix}.self_attn.out_proj.weight"),
                to_numpy(f"{prefix}.self_attn.out_proj.bias"),
                model.d_model,
                model.nhead,
            ),
            "linear1": {
                "kernel": torch_to_jax_linear(to_numpy(f"{prefix}.linear1.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.linear1.bias")),
            },
            "linear2": {
                "kernel": torch_to_jax_linear(to_numpy(f"{prefix}.linear2.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.linear2.bias")),
            },
            "norm1": {
                "scale": jnp.array(to_numpy(f"{prefix}.norm1.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.norm1.bias")),
            },
            "norm2": {
                "scale": jnp.array(to_numpy(f"{prefix}.norm2.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.norm2.bias")),
            },
        }
        encoder_layers.append(layer_params)
    params["encoder_layers"] = encoder_layers

    # Encoder final norm
    if "transformer.encoder.norm.weight" in checkpoint:
        params["encoder_norm"] = {
            "scale": jnp.array(to_numpy("transformer.encoder.norm.weight")),
            "bias": jnp.array(to_numpy("transformer.encoder.norm.bias")),
        }

    # Decoder layers
    decoder_layers = []
    for i in range(model.num_decoder_layers):
        prefix = f"transformer.decoder.layers.{i}"
        layer_params = {
            "self_attn": torch_to_jax_multihead_attention(
                to_numpy(f"{prefix}.self_attn.in_proj_weight"),
                to_numpy(f"{prefix}.self_attn.in_proj_bias"),
                to_numpy(f"{prefix}.self_attn.out_proj.weight"),
                to_numpy(f"{prefix}.self_attn.out_proj.bias"),
                model.d_model,
                model.nhead,
            ),
            "cross_attn": torch_to_jax_multihead_attention(
                to_numpy(f"{prefix}.multihead_attn.in_proj_weight"),
                to_numpy(f"{prefix}.multihead_attn.in_proj_bias"),
                to_numpy(f"{prefix}.multihead_attn.out_proj.weight"),
                to_numpy(f"{prefix}.multihead_attn.out_proj.bias"),
                model.d_model,
                model.nhead,
            ),
            "linear1": {
                "kernel": torch_to_jax_linear(to_numpy(f"{prefix}.linear1.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.linear1.bias")),
            },
            "linear2": {
                "kernel": torch_to_jax_linear(to_numpy(f"{prefix}.linear2.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.linear2.bias")),
            },
            "norm1": {
                "scale": jnp.array(to_numpy(f"{prefix}.norm1.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.norm1.bias")),
            },
            "norm2": {
                "scale": jnp.array(to_numpy(f"{prefix}.norm2.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.norm2.bias")),
            },
            "norm3": {
                "scale": jnp.array(to_numpy(f"{prefix}.norm3.weight")),
                "bias": jnp.array(to_numpy(f"{prefix}.norm3.bias")),
            },
        }
        decoder_layers.append(layer_params)
    params["decoder_layers"] = decoder_layers

    # Decoder final norm
    if "transformer.decoder.norm.weight" in checkpoint:
        params["decoder_norm"] = {
            "scale": jnp.array(to_numpy("transformer.decoder.norm.weight")),
            "bias": jnp.array(to_numpy("transformer.decoder.norm.bias")),
        }

    # Per-query projection heads
    # PyTorch: n_queries separate (d_model -> out_i) heads
    proj_heads = []
    i = 0
    while f"proj.{i}.weight" in checkpoint:
        proj_heads.append(
            {
                "kernel": torch_to_jax_linear(to_numpy(f"proj.{i}.weight")),
                "bias": jnp.array(to_numpy(f"proj.{i}.bias")),
            }
        )
        i += 1

    params["proj_heads"] = proj_heads

    return params


def _update_mha_weights(layer_attn, attn_params: dict):
    """Update MultiHeadAttention weights from converted params."""
    layer_attn.query.kernel[...] = attn_params["query"]["kernel"]
    layer_attn.query.bias[...] = attn_params["query"]["bias"]
    layer_attn.key.kernel[...] = attn_params["key"]["kernel"]
    layer_attn.key.bias[...] = attn_params["key"]["bias"]
    layer_attn.value.kernel[...] = attn_params["value"]["kernel"]
    layer_attn.value.bias[...] = attn_params["value"]["bias"]
    layer_attn.out.kernel[...] = attn_params["out"]["kernel"]
    layer_attn.out.bias[...] = attn_params["out"]["bias"]


def _update_model_weights(model: SynthRL, params: dict, encoder_only: bool = False):
    """Manually update model weights from converted params dict.

    nnx.update() doesn't work with nnx.List, so we update manually.

    Args:
        model: SynthRL model to update.
        params: Converted parameter dict from convert_pytorch_checkpoint.
        encoder_only: If True, only load backbone + encoder weights and delete
            the decoder and projection heads from the model.
    """
    # CNN Backbone
    backbone_params = params["backbone"]
    model.backbone.conv0.kernel[...] = backbone_params["conv0"]["kernel"]
    model.backbone.conv0.bias[...] = backbone_params["conv0"]["bias"]

    for i in range(1, 5):
        conv = getattr(model.backbone, f"conv{i}")
        bn = getattr(model.backbone, f"bn{i}")
        conv.kernel[...] = backbone_params[f"conv{i}"]["kernel"]
        conv.bias[...] = backbone_params[f"conv{i}"]["bias"]
        bn.scale[...] = backbone_params[f"bn{i}"]["scale"]
        bn.bias[...] = backbone_params[f"bn{i}"]["bias"]
        bn.mean[...] = backbone_params[f"bn{i}"]["mean"]
        bn.var[...] = backbone_params[f"bn{i}"]["var"]

    # Encoder layers
    for i, layer in enumerate(model.encoder_layers):
        layer_params = params["encoder_layers"][i]
        _update_mha_weights(layer.self_attn, layer_params["self_attn"])
        layer.linear1.kernel[...] = layer_params["linear1"]["kernel"]
        layer.linear1.bias[...] = layer_params["linear1"]["bias"]
        layer.linear2.kernel[...] = layer_params["linear2"]["kernel"]
        layer.linear2.bias[...] = layer_params["linear2"]["bias"]
        layer.norm1.scale[...] = layer_params["norm1"]["scale"]
        layer.norm1.bias[...] = layer_params["norm1"]["bias"]
        layer.norm2.scale[...] = layer_params["norm2"]["scale"]
        layer.norm2.bias[...] = layer_params["norm2"]["bias"]

    # Encoder norm
    if "encoder_norm" in params:
        model.encoder_norm.scale[...] = params["encoder_norm"]["scale"]
        model.encoder_norm.bias[...] = params["encoder_norm"]["bias"]

    if encoder_only:
        return

    # Learnable queries (decoder)
    model.tgt[...] = params["tgt"]

    # Decoder layers
    for i, layer in enumerate(model.decoder_layers):
        layer_params = params["decoder_layers"][i]
        _update_mha_weights(layer.self_attn, layer_params["self_attn"])
        _update_mha_weights(layer.cross_attn, layer_params["cross_attn"])
        layer.linear1.kernel[...] = layer_params["linear1"]["kernel"]
        layer.linear1.bias[...] = layer_params["linear1"]["bias"]
        layer.linear2.kernel[...] = layer_params["linear2"]["kernel"]
        layer.linear2.bias[...] = layer_params["linear2"]["bias"]
        layer.norm1.scale[...] = layer_params["norm1"]["scale"]
        layer.norm1.bias[...] = layer_params["norm1"]["bias"]
        layer.norm2.scale[...] = layer_params["norm2"]["scale"]
        layer.norm2.bias[...] = layer_params["norm2"]["bias"]
        layer.norm3.scale[...] = layer_params["norm3"]["scale"]
        layer.norm3.bias[...] = layer_params["norm3"]["bias"]

    # Decoder norm
    if "decoder_norm" in params:
        model.decoder_norm.scale[...] = params["decoder_norm"]["scale"]
        model.decoder_norm.bias[...] = params["decoder_norm"]["bias"]

    # Per-query projection heads
    if "proj_heads" in params:
        ckpt_heads = params["proj_heads"]
        if len(model.proj_heads) != len(ckpt_heads):
            raise ValueError(
                f"Projection head count mismatch: model has {len(model.proj_heads)}, "
                f"checkpoint has {len(ckpt_heads)}. Pass a matching "
                f"PresetIndexesHelper, or use encoder_only=True."
            )
        for i, head in enumerate(model.proj_heads):
            if (
                head.kernel.shape != ckpt_heads[i]["kernel"].shape
                or head.bias.shape != ckpt_heads[i]["bias"].shape
            ):
                raise ValueError(
                    f"Projection head {i} shape mismatch: model has "
                    f"kernel {head.kernel.shape}, checkpoint has "
                    f"{ckpt_heads[i]['kernel'].shape}. Pass a matching "
                    f"PresetIndexesHelper, or use encoder_only=True."
                )
            head.kernel[...] = ckpt_heads[i]["kernel"]
            head.bias[...] = ckpt_heads[i]["bias"]


def load_model(
    checkpoint_path: str | Path,
    preset_helper: PresetIndexesHelper | None = None,
    n_queries: int | None = None,
    encoder_only: bool = False,
    **kwargs,
) -> SynthRL:
    """Load SynthRL model from a pretrained key or checkpoint path.

    Args:
        checkpoint_path: A pretrained key — ``"in-domain-dexed"`` (Dexed,
            in-domain) or ``"out-of-domain-surge"`` (Surge, out-of-domain),
            downloaded and cached as a converted ``.safetensors`` — or a path
            to a local checkpoint (``.safetensors``, ``.pt``, or ``.pth``). The
            architecture (``d_model``, layer counts, ``n_queries``,
            ``dim_feedforward``) is inferred from the checkpoint unless
            overridden via ``**kwargs``.
        preset_helper: Preset index helper. If None, creates identity helper.
        n_queries: Number of queries (parameters). Used if preset_helper is None.
        encoder_only: If True, only load the backbone + encoder and delete the
            decoder and projection heads. Use this when you only need encode()
            (e.g., as a feature extractor). Avoids projection head shape
            mismatches when no PresetIndexesHelper is available.
        **kwargs: Override any SynthRL constructor kwargs (d_model, nhead, etc.).
            If not provided, values are inferred from the checkpoint.

    Returns:
        SynthRL model with loaded weights in eval mode.
    """
    # Resolve a model key or file path; pretrained keys resolve to a converted .safetensors.
    checkpoint_path = str(checkpoint_path)
    if checkpoint_path in PRETRAINED_MODELS:
        checkpoint_path = download_checkpoint(checkpoint_path)
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".safetensors":
        # Flat state dict converted to numpy once at download time -> torch-free load.
        state_dict = load_file(str(checkpoint_path))
    else:
        # Manual override: a raw PyTorch checkpoint.
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

    # Infer architecture from checkpoint, then let kwargs override.
    # Default dropout to 0.0: a pretrained model is loaded for inference (eval mode), so dropout
    # is a no-op at runtime — but the dropout *rate* still perturbs construction-time RNG, so
    # pinning it keeps a config-free load bit-identical to an explicit ``dropout=0.0`` load.
    inferred = {"dropout": 0.0}

    # Infer d_model from backbone
    if "backbone.conv.4.0.weight" in state_dict:
        inferred["d_model"] = state_dict["backbone.conv.4.0.weight"].shape[0]

    # Count encoder/decoder layers
    num_encoder_layers = 0
    while (
        f"transformer.encoder.layers.{num_encoder_layers}.self_attn.in_proj_weight"
        in state_dict
    ):
        num_encoder_layers += 1
    if num_encoder_layers > 0:
        inferred["num_encoder_layers"] = num_encoder_layers

    num_decoder_layers = 0
    while (
        f"transformer.decoder.layers.{num_decoder_layers}.self_attn.in_proj_weight"
        in state_dict
    ):
        num_decoder_layers += 1
    if num_decoder_layers > 0:
        inferred["num_decoder_layers"] = num_decoder_layers

    # Infer n_queries from tgt
    if "transformer.tgt" in state_dict:
        inferred["n_queries"] = state_dict["transformer.tgt"].shape[0]
    elif n_queries is not None:
        inferred["n_queries"] = n_queries

    # Infer dim_feedforward from encoder FFN
    if "transformer.encoder.layers.0.linear1.weight" in state_dict:
        inferred["dim_feedforward"] = state_dict[
            "transformer.encoder.layers.0.linear1.weight"
        ].shape[0]

    # kwargs override inferred values
    model_kwargs = {**inferred, **kwargs}

    # Create preset helper if not provided
    if preset_helper is None:
        # Count projection heads to determine learnable size
        num_proj = 0
        total_out_dim = 0
        while f"proj.{num_proj}.weight" in state_dict:
            total_out_dim += state_dict[f"proj.{num_proj}.weight"].shape[0]
            num_proj += 1

        preset_helper = PresetIndexesHelper(
            dataset=None,
            nb_params=total_out_dim,
        )

    # Create model
    rngs = nnx.Rngs(0)
    model = SynthRL(preset_helper, rngs=rngs, encoder_only=encoder_only, **model_kwargs)

    if not encoder_only:
        # Run forward pass to initialize shapes (requires decoder)
        dummy_input = jnp.zeros((1, 128, 128, 1))
        _ = model(dummy_input)

    # Convert and load weights
    params = convert_pytorch_checkpoint(state_dict, model)

    # Update model weights manually (nnx.update doesn't work with nnx.List)
    _update_model_weights(model, params, encoder_only=encoder_only)

    # Set to eval mode
    model.eval()

    return model
