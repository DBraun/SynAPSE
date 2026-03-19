/**
 * Parse DX7 sysex bank files (.syx) in the browser.
 *
 * A DX7 bulk dump is 4104 bytes:
 *   - 6 bytes header (F0 43 00 09 20 00)
 *   - 4096 bytes voice data (32 voices × 128 bytes packed)
 *   - 1 byte checksum
 *   - 1 byte end-of-exclusive (F7)
 *
 * Each packed voice is 128 bytes. We unpack to 156 bytes, then
 * normalize to the 145-dim float32 array matching dexed-py Preset.to_array().
 */

export interface SysexPreset {
  name: string;
  preset: number[]; // 145-dim float32 array (dexed-py format)
}

/** Unpack a 128-byte packed voice to 156-byte unpacked format. */
function unpackVoice(packed: Uint8Array): Uint8Array {
  const u = new Uint8Array(156);

  for (let op = 0; op < 6; op++) {
    // Copy first 11 bytes directly
    for (let i = 0; i < 11; i++) {
      u[op * 21 + i] = packed[op * 17 + i];
    }

    // Unpack combined bytes
    const leftRightCurves = packed[op * 17 + 11];
    u[op * 21 + 11] = leftRightCurves & 0x03;
    u[op * 21 + 12] = (leftRightCurves >> 2) & 0x03;

    const detuneRs = packed[op * 17 + 12];
    u[op * 21 + 13] = detuneRs & 0x07;       // rate_scaling
    u[op * 21 + 20] = detuneRs >> 3;          // detune

    const kvsAms = packed[op * 17 + 13];
    u[op * 21 + 14] = kvsAms & 0x03;          // amp_mod_sensitivity
    u[op * 21 + 15] = kvsAms >> 2;            // velocity_sensitivity

    u[op * 21 + 16] = packed[op * 17 + 14];   // output_level

    const fcoarseMode = packed[op * 17 + 15];
    u[op * 21 + 17] = fcoarseMode & 0x01;     // frequency_mode
    u[op * 21 + 18] = fcoarseMode >> 1;       // frequency_coarse

    u[op * 21 + 19] = packed[op * 17 + 16];   // frequency_fine
  }

  // Pitch EG and other globals
  for (let i = 0; i < 9; i++) u[126 + i] = packed[102 + i];

  const oksFb = packed[111];
  u[135] = oksFb & 0x07;         // feedback
  u[136] = oksFb >> 3;           // osc_key_sync

  for (let i = 0; i < 4; i++) u[137 + i] = packed[112 + i];

  const lpmsLfwLks = packed[116];
  u[141] = lpmsLfwLks & 0x01;           // lfo_sync
  u[142] = (lpmsLfwLks >> 1) & 0x07;    // lfo_wave
  u[143] = lpmsLfwLks >> 4;             // pitch_mod_sensitivity

  for (let i = 0; i < 11; i++) u[144 + i] = packed[117 + i];
  u[155] = 0x3F; // operator on/off flags (all on)

  return u;
}

/**
 * Convert 156-byte unpacked DX7 voice data to 145-dim float32 array.
 * Matches dexed-py Preset.to_array() exactly.
 *
 * Unpacked format (from dexed-py Patch.from_sysex):
 *   Per operator (6 × 21 bytes, stored OP6-first but we reverse):
 *     [0-3] envelope rates, [4-7] envelope levels,
 *     [8] breakpoint, [9] left_depth, [10] right_depth,
 *     [11] left_curve, [12] right_curve,
 *     [13] rate_scaling, [14] amp_mod_sensitivity, [15] velocity_sensitivity,
 *     [16] output_level, [17] frequency_mode, [18] frequency_coarse,
 *     [19] frequency_fine, [20] detune
 *   Global:
 *     [126-129] pitch_env_rates, [130-133] pitch_env_levels,
 *     [134] algorithm, [135] feedback, [136] osc_key_sync,
 *     [137] lfo_speed, [138] lfo_delay, [139] lfo_pitch_mod_depth,
 *     [140] lfo_amp_mod_depth, [141] lfo_sync, [142] lfo_wave,
 *     [143] pitch_mod_sensitivity, [144] transpose,
 *     [145-154] name (ASCII)
 */
function unpackedToFloat32(data: Uint8Array): { preset: number[]; name: string } {
  const arr: number[] = new Array(145);

  // --- Read operators in OP1-6 order (sysex stores OP6 first) ---
  const ops: {
    envRates: number[]; envLevels: number[];
    breakpoint: number; leftDepth: number; rightDepth: number;
    leftCurve: number; rightCurve: number;
    rateScaling: number; ampModSens: number; velocitySens: number;
    outputLevel: number; freqMode: number; freqCoarse: number;
    freqFine: number; detune: number;
  }[] = [];

  for (let sysexOp = 0; sysexOp < 6; sysexOp++) {
    const base = sysexOp * 21;
    const op = {
      envRates: [data[base], data[base + 1], data[base + 2], data[base + 3]],
      envLevels: [data[base + 4], data[base + 5], data[base + 6], data[base + 7]],
      breakpoint: data[base + 8],
      leftDepth: data[base + 9],
      rightDepth: data[base + 10],
      leftCurve: data[base + 11] & 0x03,
      rightCurve: data[base + 12] & 0x03,
      rateScaling: data[base + 13] & 0x07,
      ampModSens: data[base + 14] & 0x03,
      velocitySens: data[base + 15] & 0x07,
      outputLevel: data[base + 16],
      freqMode: data[base + 17] & 0x01,
      freqCoarse: data[base + 18] & 0x1F,
      freqFine: data[base + 19],
      detune: data[base + 20] & 0x0F,
    };
    ops.push(op);
  }
  // Reverse to get OP1-6 order (sysex stores OP6 first)
  ops.reverse();

  // --- Global continuous (indices 0–14) ---
  const algorithm = data[134] & 0x1F;
  const feedback = data[135] & 0x07;
  const pitchModSens = data[143] & 0x07;
  const lfoSpeed = data[137];
  const lfoDelay = data[138];
  const lfoPitchModDepth = data[139];
  const lfoAmpModDepth = data[140];

  arr[0] = feedback / 7;
  arr[1] = data[144] / 48;                       // transpose
  arr[2] = pitchModSens / 7;
  arr[3] = lfoSpeed / 99;
  arr[4] = lfoDelay / 99;
  arr[5] = lfoPitchModDepth / 99;
  arr[6] = lfoAmpModDepth / 99;

  // Pitch envelope rates/levels
  for (let i = 0; i < 4; i++) {
    arr[7 + i] = data[126 + i] / 99;             // pitch_env_rates
    arr[11 + i] = data[130 + i] / 99;            // pitch_env_levels
  }

  // --- Per-operator continuous (indices 15–122, 6 ops × 18 each) ---
  for (let op = 0; op < 6; op++) {
    const base = 15 + op * 18;
    const o = ops[op];
    for (let i = 0; i < 4; i++) {
      arr[base + i] = o.envRates[i] / 99;
      arr[base + 4 + i] = o.envLevels[i] / 99;
    }
    arr[base + 8] = o.outputLevel / 99;
    arr[base + 9] = o.freqCoarse / 31;
    arr[base + 10] = o.freqFine / 99;
    arr[base + 11] = o.detune / 14;
    arr[base + 12] = o.velocitySens / 7;
    arr[base + 13] = o.ampModSens / 3;
    arr[base + 14] = o.rateScaling / 7;
    arr[base + 15] = o.breakpoint / 99;
    arr[base + 16] = o.leftDepth / 99;
    arr[base + 17] = o.rightDepth / 99;
  }

  // --- Integer params (indices 123–144) ---
  arr[123] = data[136] & 0x01;                   // osc_key_sync
  arr[124] = data[141] & 0x01;                   // lfo_sync
  arr[125] = algorithm;                           // algorithm (0-indexed)
  arr[126] = data[142] & 0x07;                   // lfo_wave

  for (let op = 0; op < 6; op++) {
    const intBase = 127 + op * 3;
    arr[intBase] = ops[op].freqMode;
    arr[intBase + 1] = ops[op].leftCurve;
    arr[intBase + 2] = ops[op].rightCurve;
  }

  // Name
  let name = '';
  for (let i = 145; i < 155 && i < data.length; i++) {
    name += String.fromCharCode(data[i] & 0x7F);
  }
  name = name.trim();

  return { preset: arr, name: name || 'UNNAMED' };
}

/**
 * Parse a .syx file (DX7 32-voice bulk dump) and return 32 presets.
 * Each preset is a 145-dim float32 array + name.
 */
export function parseSysexBank(fileData: ArrayBuffer): SysexPreset[] {
  const bytes = new Uint8Array(fileData);

  // Find the start of voice data.
  // Standard header: F0 43 00 09 20 00 → data starts at byte 6
  // Some files may have different sub-status bytes, so just look for F0 43
  let dataStart = 0;
  if (bytes[0] === 0xF0 && bytes[1] === 0x43) {
    dataStart = 6;
  }
  // Some files are raw 4096 bytes with no header
  if (bytes.length === 4096) {
    dataStart = 0;
  }

  const presets: SysexPreset[] = [];
  for (let i = 0; i < 32; i++) {
    const offset = dataStart + i * 128;
    if (offset + 128 > bytes.length) break;

    const packed = bytes.slice(offset, offset + 128);
    const unpacked = unpackVoice(packed);
    const { preset, name } = unpackedToFloat32(unpacked);
    presets.push({ name, preset });
  }

  return presets;
}
