import type { DX7Params, LfoGlobalState } from './types';
import { DX7_CONFIG, OUTPUT_LEVEL_TABLE, LFO_FREQUENCY_TABLE, PER_VOICE_LEVEL } from './constants';
import { FMVoice } from './FMVoice';

const PERIOD = Math.PI * 2;

export class DX7Synth {
  params: DX7Params;
  private voices: FMVoice[] = [];
  private polyphony: number;
  private bend: number = 0;
  sampleRate: number;
  private lfoSamplePeriod: number;
  lfoState: LfoGlobalState;

  constructor(sampleRate?: number) {
    this.sampleRate = sampleRate ?? DX7_CONFIG.sampleRate;
    this.polyphony = DX7_CONFIG.polyphony;
    this.lfoSamplePeriod = DX7_CONFIG.lfoSamplePeriod;
    this.lfoState = {
      phaseStep: 0,
      pitchModDepth: 0,
      ampModDepth: 0,
      sampleHoldRandom: 0,
      delayTimes: [0, 0, 0],
      delayIncrements: [0, 0, 0],
      delayVals: [0, 0, 1],
    };
    // Initialize with a default silent preset
    this.params = this.createDefaultParams();
  }

  private createDefaultParams(): DX7Params {
    const operators = [];
    for (let i = 0; i < 6; i++) {
      operators.push({
        idx: i,
        enabled: true,
        rates: [99, 99, 99, 99] as [number, number, number, number],
        levels: [99, 99, 99, 0] as [number, number, number, number],
        detune: 0,
        velocitySens: 0,
        lfoAmpModSens: 0,
        volume: 0,
        oscMode: 0,
        freqCoarse: 1,
        freqFine: 0,
        pan: 0,
        outputLevel: 0,
        freqRatio: 1,
        freqFixed: 1,
        ampL: Math.cos(Math.PI / 4),
        ampR: Math.sin(Math.PI / 4),
        rateScaling: 0,
        breakpoint: 39,
        leftDepth: 0,
        rightDepth: 0,
        leftCurve: 0,
        rightCurve: 0,
      });
    }
    return {
      name: 'INIT',
      algorithm: 1,
      feedback: 0,
      fbRatio: Math.pow(2, (0 - 7)),
      operators,
      lfoSpeed: 35,
      lfoDelay: 0,
      lfoPitchModDepth: 0,
      lfoAmpModDepth: 0,
      lfoPitchModSens: 0,
      lfoWaveform: 0,
      lfoSync: 0,
      oscKeySync: 1,
      pitchEnvelope: {
        rates: [99, 99, 99, 99],
        levels: [50, 50, 50, 50],
      },
      transpose: 24,
      pitchModSensitivity: 0,
      controllerModVal: 0,
      aftertouchEnabled: 0,
    };
  }

  loadPreset(params: DX7Params): void {
    this.params = params;
    // Compute derived values
    this.setFeedback(params.feedback);
    for (let i = 0; i < 6; i++) {
      this.setOutputLevel(i, params.operators[i].volume);
      this.updateFrequency(i);
      this.setPan(i, params.operators[i].pan);
    }
    this.updateLFO();
    // Stop all current voices when loading a new preset
    this.panic();
  }

  private setFeedback(value: number): void {
    this.params.fbRatio = Math.pow(2, (value - 7));
  }

  private setOutputLevel(operatorIndex: number, value: number): void {
    const idx = Math.min(99, Math.max(0, Math.floor(value)));
    this.params.operators[operatorIndex].outputLevel = OUTPUT_LEVEL_TABLE[idx] * 1.27;
  }

  private updateFrequency(operatorIndex: number): void {
    const op = this.params.operators[operatorIndex];
    if (op.oscMode === 0) {
      const freqCoarse = op.freqCoarse || 0.5;
      op.freqRatio = freqCoarse * (1 + op.freqFine / 100);
    } else {
      op.freqFixed = Math.pow(10, op.freqCoarse % 4) * (1 + (op.freqFine / 99) * 8.772);
    }
  }

  private setPan(operatorIndex: number, value: number): void {
    const op = this.params.operators[operatorIndex];
    op.ampL = Math.cos(Math.PI / 2 * (value + 50) / 100);
    op.ampR = Math.sin(Math.PI / 2 * (value + 50) / 100);
  }

  private updateLFO(): void {
    const frequency = LFO_FREQUENCY_TABLE[this.params.lfoSpeed] ?? 0;
    const lfoRate = this.sampleRate / this.lfoSamplePeriod;
    this.lfoState.phaseStep = PERIOD * frequency / lfoRate;
    this.lfoState.ampModDepth = this.params.lfoAmpModDepth * 0.01;
    this.lfoState.delayTimes[0] = (lfoRate * 0.001753 * Math.pow(this.params.lfoDelay, 3.10454) + 169.344 - 168) / 1000;
    this.lfoState.delayTimes[1] = (lfoRate * 0.321877 * Math.pow(this.params.lfoDelay, 2.01163) + 494.201 - 168) / 1000;
    this.lfoState.delayIncrements[1] = 1 / (this.lfoState.delayTimes[1] - this.lfoState.delayTimes[0]);
  }

  noteOn(note: number, velocity: number): void {
    const voice = new FMVoice(
      note, velocity, this.params, this.lfoState,
      this.bend, this.sampleRate, this.lfoSamplePeriod,
    );
    if (this.voices.length >= this.polyphony) {
      this.voices.shift();
    }
    this.voices.push(voice);
  }

  noteOff(note: number): void {
    for (let i = 0; i < this.voices.length; i++) {
      const voice = this.voices[i];
      if (voice && voice.note === note && voice.down) {
        voice.down = false;
        voice.noteOff();
        break;
      }
    }
  }

  panic(): void {
    for (const voice of this.voices) {
      voice.noteOff();
    }
    this.voices = [];
  }

  render(): [number, number] {
    let outputL = 0;
    let outputR = 0;

    for (let i = this.voices.length - 1; i >= 0; i--) {
      const voice = this.voices[i];
      if (voice.isFinished()) {
        this.voices.splice(i, 1);
      } else {
        const output = voice.render();
        outputL += output[0];
        outputR += output[1];
      }
    }
    return [outputL * PER_VOICE_LEVEL, outputR * PER_VOICE_LEVEL];
  }
}
