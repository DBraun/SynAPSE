import type { DX7Params, DX7OperatorParams } from './types';
import { ALGORITHMS } from './constants';

/**
 * Denormalize a continuous value from [0,1] to native DX7 range.
 * Rounds to integer since most DX7 params are integers.
 */
function denorm(value: number, maxVal: number): number {
  return Math.max(0, Math.min(maxVal, Math.round(value * maxVal)));
}

/**
 * Layout matches dexed-py Preset.to_array() — see dexed-py docs/parameter-format.md.
 *
 * Indices 0-122: continuous params normalized [0,1]
 *   Global (15): indices 0-14
 *   Per-operator (108): indices 15-122, **field-major** order.
 *     Each field spans all 6 ops before the next field starts:
 *       15-38:  env_rates    (6 × 4)
 *       39-62:  env_levels   (6 × 4)
 *       63-68:  output_level (6)
 *       69-74:  freq_coarse  (6)
 *       75-80:  freq_fine    (6)
 *       81-86:  detune       (6)
 *       87-92:  velocity_sens(6)
 *       93-98:  amp_mod_sens (6)
 *       99-104: rate_scaling (6)
 *       105-110:breakpoint   (6)
 *       111-116:left_depth   (6)
 *       117-122:right_depth  (6)
 *
 * Indices 123-144: integer params (stored as float, round to int)
 *   123: osc_key_sync, 124: lfo_sync, 125: algorithm (0-indexed), 126: lfo_wave
 *   127-132: freq_mode (6), 133-138: left_curve (6), 139-144: right_curve (6)
 */
export function float32ToDX7Params(preset: number[], name: string = 'PRESET'): DX7Params {
  // Global continuous params (indices 0-14)
  const feedback = denorm(preset[0], 7);
  const transpose = denorm(preset[1], 48);
  const pitchModSensitivity = denorm(preset[2], 7);
  const lfoSpeed = denorm(preset[3], 99);
  const lfoDelay = denorm(preset[4], 99);
  const lfoPitchModDepth = denorm(preset[5], 99);
  const lfoAmpModDepth = denorm(preset[6], 99);

  const pitchEnvRates: [number, number, number, number] = [
    denorm(preset[7], 99), denorm(preset[8], 99),
    denorm(preset[9], 99), denorm(preset[10], 99),
  ];
  const pitchEnvLevels: [number, number, number, number] = [
    denorm(preset[11], 99), denorm(preset[12], 99),
    denorm(preset[13], 99), denorm(preset[14], 99),
  ];

  // Per-operator params — field-major layout from dexed-py Preset.to_array()
  const operators: DX7OperatorParams[] = [];
  for (let op = 0; op < 6; op++) {
    const envRates: [number, number, number, number] = [
      denorm(preset[15 + op * 4 + 0], 99), denorm(preset[15 + op * 4 + 1], 99),
      denorm(preset[15 + op * 4 + 2], 99), denorm(preset[15 + op * 4 + 3], 99),
    ];
    const envLevels: [number, number, number, number] = [
      denorm(preset[39 + op * 4 + 0], 99), denorm(preset[39 + op * 4 + 1], 99),
      denorm(preset[39 + op * 4 + 2], 99), denorm(preset[39 + op * 4 + 3], 99),
    ];
    const outputLevel = denorm(preset[63 + op], 99);
    const freqCoarse = denorm(preset[69 + op], 31);
    const freqFine = denorm(preset[75 + op], 99);
    // dexed-py detune range is 0-14, dx7-synth-js expects -7 to +7
    const detuneRaw = denorm(preset[81 + op], 14);
    const detune = detuneRaw - 7;
    const velocitySens = denorm(preset[87 + op], 7);
    const ampModSens = denorm(preset[93 + op], 3);
    const rateScaling = denorm(preset[99 + op], 7);
    const breakpoint = denorm(preset[105 + op], 99);
    const leftDepth = denorm(preset[111 + op], 99);
    const rightDepth = denorm(preset[117 + op], 99);

    // Integer per-op params (field-major)
    const freqMode = Math.round(preset[127 + op]);
    const leftCurve = Math.round(preset[133 + op]);
    const rightCurve = Math.round(preset[139 + op]);

    operators.push({
      idx: op,
      enabled: true,
      rates: envRates,
      levels: envLevels,
      detune,
      velocitySens,
      lfoAmpModSens: ampModSens,
      volume: outputLevel,
      oscMode: freqMode,
      freqCoarse,
      freqFine,
      pan: 0,
      outputLevel: 0, // computed by Synth.loadPreset
      freqRatio: 1, // computed by Synth.loadPreset
      freqFixed: 1, // computed by Synth.loadPreset
      ampL: Math.cos(Math.PI / 4), // computed by Synth.loadPreset
      ampR: Math.sin(Math.PI / 4), // computed by Synth.loadPreset
      rateScaling,
      breakpoint,
      leftDepth,
      rightDepth,
      leftCurve,
      rightCurve,
    });
  }

  // Global integer params
  const oscKeySync = Math.round(preset[123]);
  const lfoSync = Math.round(preset[124]);
  const algorithm = Math.round(preset[125]) + 1; // 0-indexed in dexed-py, 1-indexed in dx7-synth-js

  const lfoWaveform = Math.round(preset[126]);

  return {
    name,
    algorithm,
    feedback,
    fbRatio: 0, // computed by Synth.loadPreset
    operators,
    lfoSpeed,
    lfoDelay,
    lfoPitchModDepth,
    lfoAmpModDepth,
    lfoPitchModSens: pitchModSensitivity,
    lfoWaveform,
    lfoSync,
    oscKeySync,
    pitchEnvelope: { rates: pitchEnvRates, levels: pitchEnvLevels },
    transpose,
    pitchModSensitivity,
    controllerModVal: 0,
    aftertouchEnabled: 0,
  };
}

/** Get the number of carrier operators for a given algorithm (0-indexed). */
export function getCarrierCount(algorithm0: number): number {
  return ALGORITHMS[algorithm0]?.outputMix.length ?? 0;
}

/** Get which operators are carriers for a given algorithm (0-indexed). */
export function getCarriers(algorithm0: number): number[] {
  return ALGORITHMS[algorithm0]?.outputMix ?? [];
}
