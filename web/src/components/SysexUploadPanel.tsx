import { useState, useCallback, useRef } from 'react';
import { useTheme } from '../context/ThemeContext';
import { getTwClasses } from '../theme';
import { parseSysexBank, embedPreset, projectTo2D } from '../dx7';
import type { SysexPreset } from '../dx7';
import type { DataPoint } from '../types';

interface SysexUploadPanelProps {
  onPresetReady: (point: DataPoint) => void;
}

export function SysexUploadPanel({ onPresetReady }: SysexUploadPanelProps) {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [bank, setBank] = useState<SysexPreset[] | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [isEmbedding, setIsEmbedding] = useState(false);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const buffer = await file.arrayBuffer();
    const presets = parseSysexBank(buffer);

    if (presets.length === 0) {
      alert('No presets found in file. Make sure it is a DX7 32-voice bulk dump (.syx).');
      return;
    }

    setBank(presets);
    setFileName(file.name);
    setSelectedIdx(0);
  }, []);

  const handleRetrieve = useCallback(async () => {
    if (!bank) return;
    const preset = bank[selectedIdx];

    setIsEmbedding(true);
    try {
      const embedding = await embedPreset(preset.preset);
      const { x, y } = projectTo2D(embedding);

      const point: DataPoint = {
        id: `upload_${preset.name.trim().replace(/\s+/g, '_')}`,
        modality: 'preset',
        dx7_preset: preset.preset,
        preset_name: preset.name,
        embedding,
        x,
        y,
        pair_id: `upload_0`,
        tags: ['uploaded'],
      };

      onPresetReady(point);
    } finally {
      setIsEmbedding(false);
    }
  }, [bank, selectedIdx, onPresetReady]);

  return (
    <div className="p-4 space-y-3">
      <h3 className={`text-sm font-semibold ${twClasses.textPrimary}`}>Upload .syx</h3>

      {/* File picker */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".syx"
        onChange={handleFileChange}
        className="hidden"
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        className={`w-full px-3 py-2 text-sm rounded-lg ${twClasses.buttonSecondary} transition-colors text-left`}
      >
        {fileName ?? 'Choose .syx file...'}
      </button>

      {/* Preset selector */}
      {bank && (
        <>
          <select
            value={selectedIdx}
            onChange={(e) => setSelectedIdx(parseInt(e.target.value))}
            className={`w-full ${twClasses.bgInput} ${twClasses.textPrimary} rounded-lg px-3 py-2 text-sm border ${twClasses.borderInput} focus:outline-none`}
          >
            {bank.map((p, i) => (
              <option key={i} value={i}>
                {i + 1}. {p.name}
              </option>
            ))}
          </select>

          <div className={`text-xs px-3 py-2 rounded-lg bg-yellow-500/10 border border-yellow-500/30 text-yellow-200`}>
            Browser-based preset encoding is not yet supported. The preset will be placed at an approximate position.
          </div>

          <button
            onClick={handleRetrieve}
            disabled={isEmbedding}
            className={`w-full px-3 py-2 text-sm rounded-lg ${twClasses.buttonPrimary} transition-colors disabled:opacity-50`}
          >
            {isEmbedding ? 'Embedding...' : 'Retrieve Neighbors'}
          </button>
        </>
      )}
    </div>
  );
}
