import type { DX7OperatorParams } from './types';
import { EnvelopeDX7 } from './EnvelopeDX7';
import { LfoDX7 } from './LfoDX7';

const OCTAVE_1024 = 1.0006771307; // Math.exp(Math.log(2)/1024)
const PERIOD = Math.PI * 2;

export class Operator {
  private phase: number = 0;
  val: number = 0;
  outputLevel: number;
  private params: DX7OperatorParams;
  private envelope: EnvelopeDX7;
  private lfo: LfoDX7;
  private phaseStep: number = 0;
  private sampleRate: number;

  constructor(
    params: DX7OperatorParams,
    baseFrequency: number,
    envelope: EnvelopeDX7,
    lfo: LfoDX7,
    sampleRate: number,
  ) {
    this.params = params;
    this.outputLevel = params.outputLevel;
    this.envelope = envelope;
    this.lfo = lfo;
    this.sampleRate = sampleRate;
    this.updateFrequency(baseFrequency);
  }

  updateFrequency(baseFrequency: number): void {
    const frequency = this.params.oscMode
      ? this.params.freqFixed
      : baseFrequency * this.params.freqRatio * Math.pow(OCTAVE_1024, this.params.detune);
    this.phaseStep = PERIOD * frequency / this.sampleRate;
  }

  render(mod: number): number {
    this.val = Math.sin(this.phase + mod) * this.envelope.render() * this.lfo.renderAmp();
    this.phase += this.phaseStep * this.lfo.render();
    if (this.phase >= PERIOD) {
      this.phase -= PERIOD;
    }
    return this.val;
  }

  noteOff(): void {
    this.envelope.noteOff();
  }

  isFinished(): boolean {
    return this.envelope.isFinished();
  }
}
