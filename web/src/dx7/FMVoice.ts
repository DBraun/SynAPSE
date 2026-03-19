import type { DX7Params, LfoGlobalState } from './types';
import { ALGORITHMS } from './constants';
import { EnvelopeDX7 } from './EnvelopeDX7';
import { LfoDX7 } from './LfoDX7';
import { Operator } from './Operator';
import { scaleLevel, scaleRate, effectiveOutputLevelGain } from './scaling';

export class FMVoice {
  down: boolean = true;
  note: number;
  private velocity: number;
  private operators: Operator[];
  private params: DX7Params;
  private bend: number;

  constructor(
    note: number,
    velocity: number,
    params: DX7Params,
    lfoState: LfoGlobalState,
    bend: number,
    sampleRate: number,
    lfoSamplePeriod: number,
  ) {
    this.note = note;
    this.velocity = velocity;
    this.params = params;
    this.bend = bend;
    const frequency = FMVoice.frequencyFromNoteNumber(this.note);

    this.operators = new Array(6);
    for (let i = 0; i < 6; i++) {
      const opParams = params.operators[i];

      // DX7 keyboard scaling is note-dependent, so it is resolved here (per
      // voice) rather than at preset load. Level scaling folds into the
      // operator's output level; rate scaling speeds up the envelope for
      // higher notes.
      const levelScaling = scaleLevel(
        note,
        opParams.breakpoint,
        opParams.leftDepth,
        opParams.rightDepth,
        opParams.leftCurve,
        opParams.rightCurve,
      );
      const rateScaling = scaleRate(note, opParams.rateScaling);

      const op = new Operator(
        opParams,
        frequency,
        new EnvelopeDX7(opParams.levels, opParams.rates, rateScaling),
        new LfoDX7(opParams, params, lfoState, lfoSamplePeriod),
        sampleRate,
      );
      const velocityGain = 1 + (this.velocity - 1) * (opParams.velocitySens / 7);
      op.outputLevel = velocityGain * effectiveOutputLevelGain(opParams.volume, levelScaling);
      this.operators[i] = op;
    }
    this.updatePitchBend();
  }

  static frequencyFromNoteNumber(note: number): number {
    return 440 * Math.pow(2, (note - 69) / 12);
  }

  render(): [number, number] {
    const algorithmIdx = this.params.algorithm - 1;
    const modulationMatrix = ALGORITHMS[algorithmIdx].modulationMatrix;
    const outputMix = ALGORITHMS[algorithmIdx].outputMix;
    const outputScaling = 1 / outputMix.length;
    let outputL = 0;
    let outputR = 0;

    for (let i = 5; i >= 0; i--) {
      let mod = 0;
      if (this.params.operators[i].enabled) {
        for (let j = 0; j < modulationMatrix[i].length; j++) {
          const modulator = modulationMatrix[i][j];
          if (this.params.operators[modulator].enabled) {
            const modOp = this.operators[modulator];
            if (modulator === i) {
              mod += modOp.val * this.params.fbRatio;
            } else {
              mod += modOp.val * modOp.outputLevel;
            }
          }
        }
      }
      this.operators[i].render(mod);
    }

    for (let k = 0; k < outputMix.length; k++) {
      if (this.params.operators[outputMix[k]].enabled) {
        const carrier = this.operators[outputMix[k]];
        const carrierParams = this.params.operators[outputMix[k]];
        const carrierLevel = carrier.val * carrier.outputLevel;
        outputL += carrierLevel * carrierParams.ampL;
        outputR += carrierLevel * carrierParams.ampR;
      }
    }
    return [outputL * outputScaling, outputR * outputScaling];
  }

  noteOff(): void {
    this.down = false;
    for (let i = 0; i < 6; i++) {
      this.operators[i].noteOff();
    }
  }

  updatePitchBend(): void {
    const frequency = FMVoice.frequencyFromNoteNumber(this.note + this.bend);
    for (let i = 0; i < 6; i++) {
      this.operators[i].updateFrequency(frequency);
    }
  }

  isFinished(): boolean {
    const outputMix = ALGORITHMS[this.params.algorithm - 1].outputMix;
    for (let i = 0; i < outputMix.length; i++) {
      if (!this.operators[outputMix[i]].isFinished()) return false;
    }
    return true;
  }
}
