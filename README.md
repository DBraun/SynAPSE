# FM Synthesizer Audio-Parameter Shared Embeddings

FM-SynAPSE (shortened to "SynAPSE") learns a joint embedding space for **DX7 audio** and **DX7 synthesizer presets**.
By cosine similarity, an audio query can retrieve presets (and vice versa).

This codebase accompanies the paper **"FM Synthesizer Audio-Parameter Shared Embeddings"** by [David Braun](https://github.com/dbraun/) and [Adam Finkelstein](https://www.cs.princeton.edu/~af/), in Proc. of the 29th Int. Conference on Digital Audio Effects
([DAFx26](https://dafx26.mit.edu/)), Cambridge, MA, September 2026.
Preprint: [arXiv:2608.18226](https://arxiv.org/abs/2608.18226)

> [!NOTE]
> This codebase contains both the standalone Python package (`src`) and the interactive web [demo](https://dbraun.github.io/SynAPSE) (`web`).
> Expect training code to arrive in the future.

## Quickstart

The trained models are on [Hugging Face](https://huggingface.co/davidbraun/fm-synapse-dx7-gnn), so audio-to-preset retrieval works without training anything.
The code below renders a DX7 voice with [`dexed-py`](https://github.com/DBraun/dexed-py) (a base dependency) and asks the model to find it again in a 32-voice cartridge:

```bash
pip install "synapse[hf] @ git+https://github.com/DBraun/SynAPSE"
```

```python
import numpy as np
from audiotree import AudioTree
from dexed import DexedSynth, Patch
from jax import numpy as jnp
from synapse import SynAPSE

SAMPLE_RATE = 44_100

# "held-out" was evaluated on 8 DX7 algorithms it never saw in training;
# "80-10-10" saw all 32. Both pair a PANNs audio encoder with the DX7-GNN.
model = SynAPSE.from_pretrained("davidbraun/fm-synapse-dx7-gnn", subfolder="held-out")

# Any DX7 cartridge (.syx) works. This one ships with the repo, so clone it or
# point load_bank at a bank of your own.
syx_filepath = "tests/syx/DX7_AllTheWeb_Aminet_2.syx"
presets = [patch.to_preset() for patch in Patch.load_bank(syx_filepath)]

# Render one voice the way the model was trained: 4 s at 44.1 kHz,
# C4 at velocity 85, a 3 s note plus 1 s of release.
target = 0
synth = DexedSynth(sample_rate=SAMPLE_RATE)
synth.load_preset(presets[target])
audio = synth.render(midi_note=60, velocity=85, note_duration=3.0, render_duration=4.0)

query = AudioTree.create(audio, SAMPLE_RATE)
gallery = AudioTree.create(
    None,  # waveform not needed for parameter encoding
    SAMPLE_RATE,
    extras={
        "params": jnp.stack([p.to_array() for p in presets]),
        "algorithm": jnp.asarray([p.algorithm for p in presets], dtype=jnp.int32),
    },
)

# arm1 is the waveform/pre-computed mel encoder.
# arm2 is the parameter encoder.
q_audio = model.arm1(query).prediction     # [1, 384]
q_preset = model.arm2(gallery).prediction  # [N, 384]

# Both arms L2-normalize, so a dot product is already the cosine score.
ranking = np.argsort(-np.asarray(q_audio @ q_preset.T)[0])
print(f"rendered preset {target}; model ranked it #{list(ranking).index(target) + 1} of {len(presets)}")
```

Run that over every voice in the cartridge and the model puts the right preset first for 27 of 32, and in the top five for 31 of 32.
The paper's headline number is 52.2% R@1 over a 4,096-preset gallery, on DX7 algorithms the model never saw during training.

Both arms consume one `audiotree.AudioTree`.
The audio arm (`arm1`) reads the AudioTree's `.waveform` (or `.extras["mel"]`).
The parameter arm (`arm2`) reads `.extras["params"]` (145-dim `Preset.to_array()`) and `.extras["algorithm"]`.
`SynAPSE.__call__` exercises both arms at once:

```python
# the single render paired with its own preset
batch = AudioTree(
    jnp.asarray(audio)[None, None, :],
    sample_rate=SAMPLE_RATE,
    extras={
        "params": jnp.asarray(presets[target].to_array())[None],
        "algorithm": jnp.asarray([presets[target].algorithm], dtype=jnp.int32),
    },
)

out = model(batch)                     # SynAPSEOutput dataclass
z_A = out.audio_output.projection      # [B, 384] audio projection
z_P = out.parameter_output.projection  # [B, 384] preset projection
q_A = out.audio_output.prediction      # [B, 384] audio prediction
q_P = out.parameter_output.prediction  # [B, 384] preset prediction
# cosine similarity between q_A and q_P is the retrieval score used at eval time.
```

## Architecture

```
SynAPSE (inherits SLAP)
├── arm1: audio encoder ──┐
│     ├── SynthRLWrapper  │  (DX7-specialized CNN+Transformer; frozen / fine_tune / scratch)
│     └── PANNsWrapper    │  (Cnn14 log-mel CNN, trained from scratch)
│                         ├── projector → predictor  (shared embedding space)
└── arm2: param encoder ──┘
      ├── DX7GNNEncoder           (6-operator FM graph, output-level gating, shared-weight message passing)
      ├── TransformerParamEncoder (6 operator tokens + algorithm cross-attention)
      └── FlatParamEncoder        (flattened features + algorithm embedding, residual/highway MLP)
```

The two encoders are chosen by string:

| `audio_encoder`     | Backbone                  | Notes                                                                        |
|:--------------------|:--------------------------|:-----------------------------------------------------------------------------|
| `"panns"` (default) | `Cnn14` (PANNs)           | log-mel CNN trained from scratch; what the released checkpoints use          |
| `"synthrl"`         | `SynthRL` CNN+Transformer | `fine_tune` loads pretrained weights (optional extra) / `frozen` / `scratch` |

| `encoder`         | Param encoder             | Topology info                         |
|:------------------|:--------------------------|:--------------------------------------|
| `"gnn"` (default) | `DX7GNNEncoder`           | algorithm graph + output-level gating |
| `"transformer"`   | `TransformerParamEncoder` | none (learned algorithm embedding)    |
| `"flat"`          | `FlatParamEncoder`        | none (learned algorithm embedding)    |

The **DX7-GNN** (`DX7GNNEncoder`) treats a patch as a graph of six nodes representing the six oscillators.
The nodes pass messages along the algorithm's modulation/feedback edges.
At the end of the message passing update, each operator's state is gated by its output level.
Algorithm topology is read at runtime from the [`dexed`](https://github.com/DBraun/dexed-py) package.

## Install

```bash
pip install -e .                       # base: torch-free
pip install -e ".[synthrl-pretrained]" # + load pretrained SynthRL weights (needs torch, one-time)
pip install -e ".[hf]"                 # + from_pretrained() from a HuggingFace repo id
pip install -e ".[test]"               # pytest
```

Base dependencies: `jax`, `flax`, `einops`, `numpy`, `absl-py`, `orbax-checkpoint`, `safetensors`,
`pyyaml`, plus the DBraun packages
[`argbind-dbraun`](https://github.com/DBraun/argbind),
[`audiotree`](https://github.com/DBraun/audiotree),
[`librosax`](https://github.com/DBraun/librosax) (JAX mel/STFT front-ends), and
[`dexed-py`](https://github.com/DBraun/dexed-py).
All of them resolve from PyPI.

The base package is **torch-free**, and so are the default encoders and the released checkpoints.
`torch` is needed only for the one-time PyTorch-to-safetensors conversion of the pretrained SynthRL
checkpoint (`audio_encoder="synthrl"` in its default `fine_tune` mode). After that first conversion,
loading is torch-free.

## Configuration

Every module is decorated with `@argbind.bind()`, enabling easy configuration from YAML or the command line.
See the [argbind](https://github.com/DBraun/argbind) fork for more details.

## Saving & loading

`SynAPSE` (and every backbone) inherits `SaveLoadModule`, a uniform save/load mixin:

```python
model.save_safetensors("model.safetensors")           # weights only (Param/BatchStat)
model.load_safetensors("model.safetensors")           # in place, into a same-shape model
model.save_orbax("ckpt/"); model.load_orbax("ckpt/")  # Orbax checkpoint dir

model.save_pretrained("out/", argbind_config={...})   # argbind_params.yml + model.safetensors
model = SynAPSE.from_pretrained("out/")               # local dir, or a HF repo id ([hf] extra)

# One repo can hold several models; name the one you want.
model = SynAPSE.from_pretrained("davidbraun/fm-synapse-dx7-gnn", subfolder="80-10-10")
model.train()  # switch to training mode (pretrained models default to eval)
```

`from_checkpoint` / `from_pretrained` replay the argbind config under `argbind.scope` and rebuild via `nnx.eval_shape` (no double allocation) before filling weights.

## Package layout

```
src/synapse/
├── model.py         # SynAPSE (top-level SLAP model)
├── slap.py          # SLAP, SiameseArm, SLAPConfig, SiameseOutput, MLP
├── base_module.py   # SaveLoadModule (checkpoint / distribution mixin + audio helpers)
├── checkpoint.py    # safetensors + Orbax weight save/load helpers
├── activations.py   # parse_activation
├── utils.py         # parse_dtype, get_cache_dir
├── layers/          # AttentionPool, torch_nn (PyTorch-init Conv/Linear)
├── backbones/       # Cnn14 (+ PANNs blocks), SynthRL (+ layers, PresetIndexesHelper)
├── audio_encoders/  # PANNsWrapper, SynthRLWrapper
├── param_encoders/  # dx7_features, FlatParamEncoder, TransformerParamEncoder, dx7gnn/
└── pretrained/      # OPTIONAL SynthRL pretrained loader (torch/download)
```
