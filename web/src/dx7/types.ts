export interface DX7OperatorParams {
  idx: number;
  enabled: boolean;
  rates: [number, number, number, number];
  levels: [number, number, number, number];
  detune: number; // -7 to +7
  velocitySens: number; // 0-7
  lfoAmpModSens: number; // -3 to 3
  volume: number; // 0-99
  oscMode: number; // 0=ratio, 1=fixed
  freqCoarse: number; // 0-31
  freqFine: number; // 0-99
  pan: number; // -50 to +50
  outputLevel: number; // computed from volume via OUTPUT_LEVEL_TABLE
  freqRatio: number; // computed
  freqFixed: number; // computed
  ampL: number; // computed from pan
  ampR: number; // computed from pan
  rateScaling: number; // 0-7
  breakpoint: number; // 0-99
  leftDepth: number; // 0-99
  rightDepth: number; // 0-99
  leftCurve: number; // 0-3
  rightCurve: number; // 0-3
}

export interface DX7PitchEnvelope {
  rates: [number, number, number, number];
  levels: [number, number, number, number];
}

export interface DX7Params {
  name: string;
  algorithm: number; // 1-32
  feedback: number; // 0-7
  fbRatio: number; // computed from feedback
  operators: DX7OperatorParams[];
  lfoSpeed: number; // 0-99
  lfoDelay: number; // 0-99
  lfoPitchModDepth: number; // 0-99
  lfoAmpModDepth: number; // 0-99
  lfoPitchModSens: number; // 0-7
  lfoWaveform: number; // 0-5
  lfoSync: number; // 0-1
  oscKeySync: number; // 0-1
  pitchEnvelope: DX7PitchEnvelope;
  transpose: number; // 0-48
  pitchModSensitivity: number; // 0-7
  controllerModVal: number;
  aftertouchEnabled: number;
}

export interface AlgorithmDef {
  outputMix: number[];
  modulationMatrix: number[][];
}

export interface LfoGlobalState {
  phaseStep: number;
  pitchModDepth: number;
  ampModDepth: number;
  sampleHoldRandom: number;
  delayTimes: [number, number, number];
  delayIncrements: [number, number, number];
  delayVals: [number, number, number];
}

export interface DX7Config {
  sampleRate: number;
  lfoSamplePeriod: number;
  bufferSize: number;
  polyphony: number;
}
