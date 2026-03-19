# DX7 engine — differences from Dexed / dexed-py

This directory is the in-browser FM synth that auditions DX7 presets. It is a
TypeScript port of **dx7-synth-js**, not of Dexed. The dataset's audio, and the
`dx7_preset` arrays in `embeddings.json`, come from **dexed-py**
(`github.com/DBraun/dexed-py`), whose engine is the **msfa** core
(music-synthesizer-for-android) also used by the official **Dexed**
(`github.com/asb2m10/dexed`). Reference sources:

- `dexed-py/src/msfa/dx7note.cc` — per-note setup, keyboard scaling
- `dexed-py/src/msfa/env.cc` — envelope
- `dexed-py/src/msfa/fm_op_kernel.cc` — operator gain / FM
- official Dexed mirrors these at `Source/msfa/*`.

This file records where the browser engine matches the reference and where it
deliberately diverges. It exists because the two engines have different
lineages, so **audio here will not sample-match dexed-py** even though both
render the same presets.

## Parameter format (matches)

`presetMapping.ts` reads the 145-element `dexed.Preset.to_array()` layout
(field-major, normalized [0,1]) exactly, including the conventions dexed-py
uses — e.g. detune stored 0–14 is remapped to the engine's −7…+7, algorithm is
0-indexed in the array and 1-indexed in the engine. This is faithful; a preset
array round-trips to the same parameter values the reference sees.

## Behaviors ported to match the reference

These were **missing** and have been added, ported verbatim from `dx7note.cc`
(see `scaling.ts`):

| Behavior | Reference | Here |
|----------|-----------|------|
| Keyboard **level scaling** (`ScaleLevel`/`ScaleCurve`) | `dx7note.cc` | `scaling.ts` → folded into operator output level in `FMVoice.ts` |
| Keyboard **rate scaling** (`ScaleRate`) | `dx7note.cc` | `scaling.ts` → `EnvelopeDX7` rate |
| **Reverse-release** voices (non-zero final level L4 keeps ringing after note-off) | `Env::isActive` (`env.cc`) | `EnvelopeDX7.isFinished()` |

Without level scaling, an operator whose *base* output level is 0 but which is
raised by the breakpoint/depth/curve stays silent — so a carrier that relies on
level scaling (e.g. pair_03881's op3) rendered as pure silence. Without the
`isActive` rule, a reverse-release patch was culled the instant its release
reached L4, so its loud release tail was never heard.

## Architectural differences (will not sample-match)

These are inherent to the dx7-synth-js lineage and are **not** bugs to fix
piecemeal — matching them would mean porting the msfa fixed-point core.

1. **Output level placement, and its coupling to the FM index.**
   In msfa, output level + level scaling + velocity fold into the envelope's
   log-domain target (`outlevel_` in `env.cc`), and an operator's output *is*
   the envelope's `Exp2` gain (`fm_op_kernel.cc`). Here the envelope carries
   only the L1–L4 *shape* with a fixed baked-in output level (the `- 224`
   constant in `EnvelopeDX7.advance` equals msfa's `outlevel_ - 4256` with
   `outlevel_ ≈ 126<<5`), and output level is applied **separately** as a linear
   multiplier in `FMVoice.ts` (`carrier.val * carrier.outputLevel`). That single
   multiplier drives **both** carrier amplitude **and** the FM modulation index
   (`mod += modOp.val * modOp.outputLevel`). Consequence: output level cannot be
   moved into the envelope here without collapsing the modulation index for
   every preset.

2. **Level-scaling fold is an approximation.**
   Because of (1), `scaling.ts:effectiveOutputLevelGain` folds level scaling into
   the linear output multiplier rather than the envelope: it applies the offset
   in msfa's 0–127 log domain (`ENVELOPE_OUTPUT_LEVEL` is msfa's `scaleoutlevel`
   table), clamps to 127, then inverts back to the 0–99 gain-table index. For an
   unscaled operator this is **exact** (equals the previous gain). For log levels
   ≥ 48 the inverse is 1:1; below 48 (near-silent) it is approximate.

3. **Numeric domain / absolute gain staging.**
   The envelope uses a float output LUT (`outputLUT[i] = 20^((i-3824)*0.0235/20)`)
   instead of msfa's fixed-point `Exp2`/Q24 math, and the voice mix applies
   `outputScaling = 1/carrierCount` and `PER_VOICE_LEVEL = 0.125/6`. So **absolute
   loudness (peak/RMS/LUFS) differs from dexed-py** — only the relative character
   (attack, sustain, release balance, timbre) tracks the reference.

4. **Velocity model.**
   msfa uses `ScaleVelocity` (a `velocity_data` table, velocity 0–127, folded
   into `outlevel_`). Here velocity is 0–1 and applied linearly:
   `1 + (velocity - 1) * (velocitySens / 7)` in `FMVoice.ts`.

5. **Envelope timing.**
   `EnvelopeDX7` follows msfa's `Env` structure but omits the `ACCURATE_ENVELOPE`
   static-segment timing (enabled in dexed-py via `env.h`), so segment durations
   for flat/zero-rate stages are approximate.

## Playback note (UI, not engine)

Because reverse-release patches now ring indefinitely (faithful to the DX7), the
audition in `hooks/useDX7Synth.ts` is bounded: hold, release, let the tail ring,
then fade a master gain and silence the voices. The hold/tail lengths are a
preview choice and do not match the dataset's render length (3 s note / 4 s
clip) unless set to.

## Verifying against the engine

`web/scripts/verify_render.mjs` drives the real engine headlessly (the same
classes the `ScriptProcessorNode` calls) and prints peak/RMS windows for a
preset id, for regression checks after touching this directory.
