// Headless check: drive the real DX7 engine (the same classes the
// ScriptProcessorNode calls) and confirm a preset renders non-silent.
// Usage: npx tsx scripts/verify_render.mjs <preset_id>
import fs from 'node:fs';
import { DX7Synth, float32ToDX7Params } from '../src/dx7/index.ts';

const id = process.argv[2] ?? 'preset_03881';
const data = JSON.parse(fs.readFileSync(new URL('../public/data/embeddings.json', import.meta.url)));
const point = data.find((d) => d.id === id);
if (!point) throw new Error(`no point ${id}`);

const sampleRate = 44100;
const synth = new DX7Synth(sampleRate);
const params = float32ToDX7Params(point.dx7_preset, point.preset_name);
params.transpose = 24; // matches useDX7Synth.loadPreset
synth.loadPreset(params);
synth.noteOn(60, 0.8); // matches playNote() defaults (C4)

const noteOffSample = Math.floor(sampleRate * 1.5); // playNote() releases after 1.5s
const n = sampleRate * 4;
const win = Math.floor(sampleRate / 2);
let peak = 0;
let sumsq = 0;
const winSumsq = [];
let curWin = 0;
for (let i = 0; i < n; i++) {
  if (i === noteOffSample) synth.noteOff(60);
  const [l] = synth.render();
  const a = Math.abs(l);
  if (a > peak) peak = a;
  sumsq += l * l;
  curWin += l * l;
  if ((i + 1) % win === 0) {
    winSumsq.push(Math.sqrt(curWin / win));
    curWin = 0;
  }
}
const rms = Math.sqrt(sumsq / n);
const windows = winSumsq.map((r) => r.toFixed(4)).join(' ');
console.log(`${id}: peak=${peak.toFixed(5)} rms=${rms.toFixed(5)} | 0.5s-rms: ${windows}`);
