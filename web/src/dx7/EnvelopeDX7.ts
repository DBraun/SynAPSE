import { ENVELOPE_OUTPUT_LEVEL } from './constants';

const ENV_OFF = 4;

// Pre-compute output LUT: DX7 internal level -> amplitude
const outputLUT: number[] = [];
for (let i = 0; i < 4096; i++) {
  const dB = (i - 3824) * 0.0235;
  outputLUT[i] = Math.pow(20, dB / 20);
}

export class EnvelopeDX7 {
  private levels: number[];
  private rates: number[];
  private rateScaling: number;
  private level: number = 0;
  private state: number = 0;
  private down: boolean = true;
  private rising: boolean = false;
  private targetlevel: number = 0;
  private qr: number = 0;
  private decayIncrement: number = 0;

  constructor(levels: number[], rates: number[], rateScaling: number = 0) {
    this.levels = levels;
    this.rates = rates;
    this.rateScaling = rateScaling;
    this.advance(0);
  }

  render(): number {
    if (this.state < 3 || (this.state < 4 && !this.down)) {
      let lev = this.level;
      if (this.rising) {
        lev += this.decayIncrement * (2 + (this.targetlevel - lev) / 256);
        if (lev >= this.targetlevel) {
          lev = this.targetlevel;
          this.advance(this.state + 1);
        }
      } else {
        lev -= this.decayIncrement;
        if (lev <= this.targetlevel) {
          lev = this.targetlevel;
          this.advance(this.state + 1);
        }
      }
      this.level = lev;
    }
    return outputLUT[Math.floor(this.level)];
  }

  private advance(newstate: number): void {
    this.state = newstate;
    if (this.state < 4) {
      const newlevel = this.levels[this.state];
      this.targetlevel = Math.max(0, (ENVELOPE_OUTPUT_LEVEL[newlevel] << 5) - 224);
      this.rising = (this.targetlevel - this.level) > 0;
      this.qr = Math.min(63, this.rateScaling + ((this.rates[this.state] * 41) >> 6));
      this.decayIncrement = Math.pow(2, this.qr / 4) / 2048;
    }
  }

  noteOff(): void {
    this.down = false;
    this.advance(3);
  }

  isFinished(): boolean {
    // Matches msfa Env::isActive (env.cc): an envelope with a non-zero final
    // level (L4) keeps ringing after release rather than being culled. Without
    // this, reverse-release patches — quiet while held, loud on release — are
    // cut off the instant the release reaches L4.
    return this.state === ENV_OFF && this.levels[ENV_OFF - 1] <= 0;
  }
}
