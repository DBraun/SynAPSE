import type { DX7OperatorParams, DX7Params, LfoGlobalState } from './types';
import {
  LFO_MODE_TRIANGLE, LFO_MODE_SAW_DOWN, LFO_MODE_SAW_UP,
  LFO_MODE_SQUARE, LFO_MODE_SINE, LFO_MODE_SAMPLE_HOLD,
  LFO_DELAY_ONSET, LFO_DELAY_RAMP, LFO_DELAY_COMPLETE,
  LFO_PITCH_MOD_TABLE,
} from './constants';

const PERIOD = Math.PI * 2;
const PERIOD_HALF = PERIOD / 2;
const PERIOD_RECIP = 1 / PERIOD;

export class LfoDX7 {
  private opParams: DX7OperatorParams;
  private params: DX7Params;
  private lfoState: LfoGlobalState;
  private lfoSamplePeriod: number;

  private phase: number = 0;
  pitchVal: number = 0;
  private counter: number = 0;
  ampVal: number = 1;
  private ampValTarget: number = 1;
  private ampIncrement: number = 0;
  private delayVal: number = 0;
  private delayState: number = LFO_DELAY_ONSET;

  constructor(
    opParams: DX7OperatorParams,
    params: DX7Params,
    lfoState: LfoGlobalState,
    lfoSamplePeriod: number,
  ) {
    this.opParams = opParams;
    this.params = params;
    this.lfoState = lfoState;
    this.lfoSamplePeriod = lfoSamplePeriod;
  }

  render(): number {
    let amp: number = 0;
    if (this.counter % this.lfoSamplePeriod === 0) {
      switch (this.params.lfoWaveform) {
        case LFO_MODE_TRIANGLE:
          if (this.phase < PERIOD_HALF)
            amp = 4 * this.phase * PERIOD_RECIP - 1;
          else
            amp = 3 - 4 * this.phase * PERIOD_RECIP;
          break;
        case LFO_MODE_SAW_DOWN:
          amp = 1 - 2 * this.phase * PERIOD_RECIP;
          break;
        case LFO_MODE_SAW_UP:
          amp = 2 * this.phase * PERIOD_RECIP - 1;
          break;
        case LFO_MODE_SQUARE:
          amp = (this.phase < PERIOD_HALF) ? -1 : 1;
          break;
        case LFO_MODE_SINE:
          amp = Math.sin(this.phase);
          break;
        case LFO_MODE_SAMPLE_HOLD:
          amp = this.lfoState.sampleHoldRandom;
          break;
      }

      switch (this.delayState) {
        case LFO_DELAY_ONSET:
        case LFO_DELAY_RAMP:
          this.delayVal += this.lfoState.delayIncrements[this.delayState];
          if (this.counter / this.lfoSamplePeriod > this.lfoState.delayTimes[this.delayState]) {
            this.delayState++;
            this.delayVal = this.lfoState.delayVals[this.delayState];
          }
          break;
        case LFO_DELAY_COMPLETE:
          break;
      }

      amp *= this.delayVal;
      const pitchModDepth = 1 +
        LFO_PITCH_MOD_TABLE[this.params.lfoPitchModSens] *
        (this.params.controllerModVal + this.params.lfoPitchModDepth / 99);
      this.pitchVal = Math.pow(pitchModDepth, amp);

      const ampSensDepth = Math.abs(this.opParams.lfoAmpModSens) * 0.333333;
      const phase = (this.opParams.lfoAmpModSens > 0) ? 1 : -1;
      this.ampValTarget = 1 - ((this.lfoState.ampModDepth + this.params.controllerModVal) *
        ampSensDepth * (amp * phase + 1) * 0.5);
      this.ampIncrement = (this.ampValTarget - this.ampVal) / this.lfoSamplePeriod;

      this.phase += this.lfoState.phaseStep;
      if (this.phase >= PERIOD) {
        this.lfoState.sampleHoldRandom = 1 - Math.random() * 2;
        this.phase -= PERIOD;
      }
    }
    this.counter++;
    return this.pitchVal;
  }

  renderAmp(): number {
    this.ampVal += this.ampIncrement;
    return this.ampVal;
  }
}
