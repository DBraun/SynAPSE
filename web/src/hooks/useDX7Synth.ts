import { useRef, useCallback, useState, useEffect } from 'react';
import { DX7Synth, float32ToDX7Params } from '../dx7';

export interface UseDX7SynthReturn {
  isReady: boolean;
  hasPreset: boolean;
  loadPreset: (preset: number[], name?: string) => void;
  playNote: (note?: number, velocity?: number, durationMs?: number) => void;
  noteOn: (note: number, velocity?: number) => void;
  noteOff: (note: number) => void;
  stopAll: () => void;
}

// Standard piano keyboard mapping (like Ableton / FL Studio / most DAWs).
// Home row (A-;) = white keys, QWERTY row above = black keys.
// Z = octave down, X = octave up.
//
//    W  E     T  Y  U     O  P
//   A  S  D  F  G  H  J  K  L  ;
//   C  D  E  F  G  A  B  C  D  E
//    C# D#    F# G# A#    C# D#
//
const KEY_TO_SEMITONE: Record<string, number> = {
  // White keys (home row)
  a: 0,   // C
  s: 2,   // D
  d: 4,   // E
  f: 5,   // F
  g: 7,   // G
  h: 9,   // A
  j: 11,  // B
  k: 12,  // C (+1 oct)
  l: 14,  // D (+1 oct)
  ';': 16, // E (+1 oct)
  // Black keys (QWERTY row)
  w: 1,   // C#
  e: 3,   // D#
  t: 6,   // F#
  y: 8,   // G#
  u: 10,  // A#
  o: 13,  // C# (+1 oct)
  p: 15,  // D# (+1 oct)
};

export function useDX7Synth(): UseDX7SynthReturn {
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const masterGainRef = useRef<GainNode | null>(null);
  const synthRef = useRef<DX7Synth | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [hasPreset, setHasPreset] = useState(false);
  const noteOffTimeoutRef = useRef<number | null>(null);
  const stopTimeoutRef = useRef<number | null>(null);
  const heldKeysRef = useRef<Set<string>>(new Set());
  const octaveRef = useRef<number>(4); // base octave (C4 = MIDI 60)

  const ensureAudioContext = useCallback(() => {
    if (audioContextRef.current) {
      if (audioContextRef.current.state === 'suspended') {
        audioContextRef.current.resume();
      }
      return;
    }

    const ctx = new AudioContext();
    audioContextRef.current = ctx;

    const synth = new DX7Synth(ctx.sampleRate);
    synthRef.current = synth;

    // Large buffer to avoid glitches — ScriptProcessorNode runs on the main
    // thread, so React re-renders (e.g. hover updates) can starve it.
    const bufferSize = 4096;
    const processor = ctx.createScriptProcessor(bufferSize, 0, 2);
    processor.onaudioprocess = (e) => {
      const outputL = e.outputBuffer.getChannelData(0);
      const outputR = e.outputBuffer.getChannelData(1);
      for (let i = 0; i < outputL.length; i++) {
        const output = synth.render();
        outputL[i] = output[0];
        outputR[i] = output[1];
      }
    };
    // A master gain sits between the processor and the output so a preview can
    // be faded out cleanly (reverse-release patches ring at full level, so a
    // hard cut would click).
    const masterGain = ctx.createGain();
    masterGain.gain.value = 1;
    processor.connect(masterGain);
    masterGain.connect(ctx.destination);
    processorRef.current = processor;
    masterGainRef.current = masterGain;

    setIsReady(true);
  }, []);

  const loadPreset = useCallback((preset: number[], name?: string) => {
    ensureAudioContext();
    if (!synthRef.current) return;
    const params = float32ToDX7Params(preset, name);
    params.transpose = 24;
    synthRef.current.loadPreset(params);
    setHasPreset(true);
  }, [ensureAudioContext]);

  const noteOn = useCallback((note: number, velocity: number = 0.8) => {
    ensureAudioContext();
    synthRef.current?.noteOn(note, velocity);
  }, [ensureAudioContext]);

  const noteOff = useCallback((note: number) => {
    synthRef.current?.noteOff(note);
  }, []);

  const playNote = useCallback((
    note: number = 60,
    velocity: number = 0.8,
    durationMs: number = 1500,
    tailMs: number = 1500,
  ) => {
    ensureAudioContext();
    if (!synthRef.current) return;

    if (noteOffTimeoutRef.current !== null) {
      clearTimeout(noteOffTimeoutRef.current);
    }
    if (stopTimeoutRef.current !== null) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }

    // Restore full gain in case a previous preview left it faded.
    const ctx = audioContextRef.current;
    const gain = masterGainRef.current;
    if (ctx && gain) {
      gain.gain.cancelScheduledValues(ctx.currentTime);
      gain.gain.setValueAtTime(1, ctx.currentTime);
    }

    synthRef.current.noteOn(note, velocity);
    noteOffTimeoutRef.current = window.setTimeout(() => {
      synthRef.current?.noteOff(note);
      noteOffTimeoutRef.current = null;
    }, durationMs);

    // Reverse-release patches (non-zero L4) ring indefinitely after note-off,
    // so bound the preview: fade the master gain, then silence the voices.
    stopTimeoutRef.current = window.setTimeout(() => {
      const fadeSec = 0.04;
      if (ctx && gain) {
        gain.gain.cancelScheduledValues(ctx.currentTime);
        gain.gain.setValueAtTime(gain.gain.value, ctx.currentTime);
        gain.gain.linearRampToValueAtTime(0, ctx.currentTime + fadeSec);
      }
      window.setTimeout(() => synthRef.current?.panic(), fadeSec * 1000 + 20);
      stopTimeoutRef.current = null;
    }, durationMs + tailMs);
  }, [ensureAudioContext]);

  const stopAll = useCallback(() => {
    if (noteOffTimeoutRef.current !== null) {
      clearTimeout(noteOffTimeoutRef.current);
      noteOffTimeoutRef.current = null;
    }
    if (stopTimeoutRef.current !== null) {
      clearTimeout(stopTimeoutRef.current);
      stopTimeoutRef.current = null;
    }
    const ctx = audioContextRef.current;
    const gain = masterGainRef.current;
    if (ctx && gain) {
      gain.gain.cancelScheduledValues(ctx.currentTime);
      gain.gain.setValueAtTime(1, ctx.currentTime);
    }
    heldKeysRef.current.clear();
    synthRef.current?.panic();
  }, []);

  // Keyboard event handlers
  useEffect(() => {
    // Track which key maps to which MIDI note (so key-up releases the right note
    // even if octave changed while held)
    const keyToActiveNote = new Map<string, number>();

    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't capture when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
      if (!hasPreset) return;
      if (e.repeat) return;

      const key = e.key.toLowerCase();

      // Octave controls
      if (key === 'z') {
        e.preventDefault();
        octaveRef.current = Math.max(0, octaveRef.current - 1);
        return;
      }
      if (key === 'x') {
        e.preventDefault();
        octaveRef.current = Math.min(8, octaveRef.current + 1);
        return;
      }

      const semitone = KEY_TO_SEMITONE[key];
      if (semitone !== undefined) {
        e.preventDefault();
        if (!heldKeysRef.current.has(key)) {
          const note = octaveRef.current * 12 + semitone;
          if (note >= 0 && note <= 127) {
            heldKeysRef.current.add(key);
            keyToActiveNote.set(key, note);
            noteOn(note);
          }
        }
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      const activeNote = keyToActiveNote.get(key);
      if (activeNote !== undefined && heldKeysRef.current.has(key)) {
        heldKeysRef.current.delete(key);
        keyToActiveNote.delete(key);
        noteOff(activeNote);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [hasPreset, noteOn, noteOff]);

  useEffect(() => {
    return () => {
      if (noteOffTimeoutRef.current !== null) {
        clearTimeout(noteOffTimeoutRef.current);
      }
      if (stopTimeoutRef.current !== null) {
        clearTimeout(stopTimeoutRef.current);
      }
      processorRef.current?.disconnect();
      masterGainRef.current?.disconnect();
      audioContextRef.current?.close();
    };
  }, []);

  return { isReady, hasPreset, loadPreset, playNote, noteOn, noteOff, stopAll };
}
