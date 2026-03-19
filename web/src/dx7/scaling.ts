import { OUTPUT_LEVEL_TABLE, ENVELOPE_OUTPUT_LEVEL } from './constants';

// DX7 keyboard level scaling, keyboard rate scaling, and the output-level fold.
//
// Ported verbatim from the Dexed / music-synthesizer-for-android reference
// (`msfa/dx7note.cc`: ScaleCurve / ScaleLevel / ScaleRate and the outlevel
// computation in Dx7Note::init). This is what makes an operator whose *base*
// output level is 0 still audible: keyboard level scaling raises its effective
// level for notes on one side of the breakpoint. Without it, presets that rely
// on level scaling for a carrier's loudness render as silence.

// exp_scale_data from dx7note.cc (33 entries).
const EXP_SCALE_DATA = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 16, 19, 23, 27, 33, 39, 47, 56, 66,
  80, 94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 250,
];

/**
 * One side of the keyboard level-scaling curve.
 *
 * @param group Distance from the breakpoint in 3-semitone groups.
 * @param depth Scaling depth for this side (0-99).
 * @param curve Curve style: 0=-LIN, 1=-EXP, 2=+EXP, 3=+LIN.
 * @returns Signed level offset in DX7 output-level units (0-127 domain).
 */
export function scaleCurve(group: number, depth: number, curve: number): number {
  let scale: number;
  if (curve === 0 || curve === 3) {
    // linear
    scale = (group * depth * 329) >> 12;
  } else {
    // exponential
    const rawExp = EXP_SCALE_DATA[Math.min(group, EXP_SCALE_DATA.length - 1)];
    scale = (rawExp * depth * 329) >> 15;
  }
  if (curve < 2) {
    scale = -scale;
  }
  return scale;
}

/**
 * Keyboard level scaling: signed output-level offset for a given note.
 *
 * @param midinote MIDI note being played.
 * @param breakpoint Breakpoint note parameter (0-99).
 * @param leftDepth Left-side scaling depth (0-99).
 * @param rightDepth Right-side scaling depth (0-99).
 * @param leftCurve Left-side curve style (0-3).
 * @param rightCurve Right-side curve style (0-3).
 * @returns Signed offset added to the operator's output level (0-127 domain).
 */
export function scaleLevel(
  midinote: number,
  breakpoint: number,
  leftDepth: number,
  rightDepth: number,
  leftCurve: number,
  rightCurve: number,
): number {
  const offset = midinote - breakpoint - 17;
  if (offset >= 0) {
    return scaleCurve(Math.floor((offset + 1) / 3), rightDepth, rightCurve);
  }
  return scaleCurve(Math.floor(-(offset - 1) / 3), leftDepth, leftCurve);
}

/**
 * Keyboard rate scaling: quantised envelope-rate delta for a given note.
 *
 * @param midinote MIDI note being played.
 * @param sensitivity Rate-scaling sensitivity (0-7).
 * @returns Quantised rate delta added to each envelope stage's rate.
 */
export function scaleRate(midinote: number, sensitivity: number): number {
  const x = Math.min(31, Math.max(0, Math.floor(midinote / 3) - 7));
  return (sensitivity * x) >> 3;
}

// Inverse of `scaleoutlevel` (ENVELOPE_OUTPUT_LEVEL): map a 0-127 log-domain
// level back to the nearest 0-99 output-level index, so a level boosted by
// keyboard scaling can be looked up in OUTPUT_LEVEL_TABLE. ENVELOPE_OUTPUT_LEVEL
// is non-decreasing, so LOG_TO_PARAM[l] is the largest index whose log level is
// <= l.
const LOG_TO_PARAM: number[] = (() => {
  const table = new Array<number>(128).fill(0);
  for (let param = 0; param < ENVELOPE_OUTPUT_LEVEL.length; param++) {
    for (let level = ENVELOPE_OUTPUT_LEVEL[param]; level < 128; level++) {
      table[level] = param;
    }
  }
  return table;
})();

/**
 * Effective linear output-level gain after keyboard level scaling.
 *
 * Combines the base output level with the level-scaling offset in the DX7's
 * 0-127 log domain (matching `Dx7Note::init`), clamps to 127, then converts
 * back to the linear amplitude used by the operator/carrier mix. Matches the
 * scaling applied by `DX7Synth.setOutputLevel` for the unscaled case.
 *
 * @param volume Base output-level parameter (0-99).
 * @param levelScaling Signed offset from {@link scaleLevel}.
 * @returns Linear gain multiplier for the operator.
 */
export function effectiveOutputLevelGain(volume: number, levelScaling: number): number {
  const baseLog = ENVELOPE_OUTPUT_LEVEL[Math.max(0, Math.min(99, Math.floor(volume)))];
  const effLog = Math.max(0, Math.min(127, baseLog + levelScaling));
  const param = LOG_TO_PARAM[effLog];
  return OUTPUT_LEVEL_TABLE[param] * 1.27;
}
