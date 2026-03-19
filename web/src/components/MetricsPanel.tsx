import { useState, useEffect, useMemo, useRef } from 'react';
import type { DataPoint, DirectionMetrics, PrecomputedMetrics } from '../types';
import { useTheme } from '../context/ThemeContext';
import { getTwClasses } from '../theme';

interface MetricsPanelProps {
  filteredData: DataPoint[];
  allData: DataPoint[];
  distancesReady: boolean;
  precomputedMetrics: PrecomputedMetrics | null;
}

function averageMetrics(a: DirectionMetrics, b: DirectionMetrics): DirectionMetrics {
  return {
    recall_at_1: (a.recall_at_1 + b.recall_at_1) / 2,
    recall_at_5: (a.recall_at_5 + b.recall_at_5) / 2,
    recall_at_10: (a.recall_at_10 + b.recall_at_10) / 2,
    recall_at_20: (a.recall_at_20 + b.recall_at_20) / 2,
    sample_size: a.sample_size + b.sample_size,
  };
}

export function MetricsPanel({ filteredData, allData, distancesReady, precomputedMetrics }: MetricsPanelProps) {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const [metrics, setMetrics] = useState<PrecomputedMetrics | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const workerRef = useRef<Worker | null>(null);

  const lastComputedKeyRef = useRef<string | null>(null);

  const pairCount = useMemo(
    () => filteredData.filter((p) => p.modality === 'audio').length,
    [filteredData]
  );

  const totalPairCount = useMemo(
    () => allData.filter((p) => p.modality === 'audio').length,
    [allData]
  );

  // The current filter selects the whole dataset (no filtering applied).
  const isUnfiltered = allData.length > 0 && filteredData.length === allData.length;

  // Use sorted IDs to create a stable key that changes when filter content changes
  const computeKey = useMemo(() => {
    if (filteredData.length === 0) return 'empty';
    const ids = filteredData.map(p => p.id).sort().join(',');
    return `${ids}-${distancesReady}`;
  }, [filteredData, distancesReady]);

  // Clear metrics when filtered data is empty
  useEffect(() => {
    if (filteredData.length === 0) {
      setMetrics(null);
      lastComputedKeyRef.current = null;
    }
  }, [filteredData.length]);

  useEffect(() => {
    if (filteredData.length === 0) {
      return;
    }

    // Fast path: for the unfiltered dataset, show precomputed metrics immediately.
    // No distance matrix or worker is needed, so this runs even before distances
    // are ready — that is the whole point of shipping metrics.json.
    if (isUnfiltered && precomputedMetrics) {
      if (lastComputedKeyRef.current === 'precomputed') return;
      lastComputedKeyRef.current = 'precomputed';
      setMetrics(precomputedMetrics);
      setIsComputing(false);
      return;
    }

    // Worker path (filtered view, or no metrics.json). The worker computes
    // cross-modal cosine distances on the fly from the embeddings.
    if (!distancesReady) {
      // Data not loaded yet — drop any full-dataset metrics so we never show
      // numbers that don't match the current filter.
      setMetrics(null);
      setIsComputing(false);
      lastComputedKeyRef.current = null;
      return;
    }

    if (lastComputedKeyRef.current === computeKey) {
      return;
    }

    lastComputedKeyRef.current = computeKey;
    setIsComputing(true);

    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = null;
    }

    const worker = new Worker(
      new URL('../workers/metricsWorker.ts', import.meta.url),
      { type: 'module' }
    );
    workerRef.current = worker;

    worker.onmessage = (event) => {
      if (event.data.type === 'result') {
        setMetrics({
          audio_to_preset: event.data.audio_to_code,
          preset_to_audio: event.data.code_to_audio,
        });
        setIsComputing(false);
      }
    };

    worker.onerror = () => {
      setIsComputing(false);
    };

    const audioPoints = filteredData.filter(p => p.modality === 'audio');
    const codePoints = filteredData.filter(p => p.modality === 'preset');

    const minimalAudio = audioPoints.map(p => ({ id: p.id, modality: p.modality, pair_id: p.pair_id, embedding: p.embedding }));
    const minimalCode = codePoints.map(p => ({ id: p.id, modality: p.modality, pair_id: p.pair_id, embedding: p.embedding }));
    const minimalAll = allData.map(p => ({ id: p.id, modality: p.modality, pair_id: p.pair_id }));

    worker.postMessage({
      type: 'compute',
      audioPoints: minimalAudio,
      codePoints: minimalCode,
      allData: minimalAll,
    });

    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
        workerRef.current = null;
      }
    };
  }, [computeKey, isUnfiltered, precomputedMetrics, distancesReady, filteredData, allData]);

  const avgMetrics = metrics ? averageMetrics(metrics.audio_to_preset, metrics.preset_to_audio) : null;

  return (
    <div className="p-4">
      <h3 className={`text-sm font-medium ${twClasses.textSecondary} mb-3`}>
        Retrieval Metrics
      </h3>

      <div className={`text-xs ${twClasses.textSubtle} mb-3`}>
        Filtered to {pairCount} of {totalPairCount} pairs ({((pairCount / totalPairCount) * 100).toFixed(1)}%)
      </div>

      {isComputing ? (
        <div className={`text-xs ${twClasses.textSubtle} text-center py-4`}>
          Computing...
        </div>
      ) : metrics && avgMetrics ? (
        <div className="space-y-4">
          <MetricsSection labelNode={<><span className={twClasses.audioText}>Audio</span> → <span className={twClasses.presetText}>Preset</span></>} metrics={metrics.audio_to_preset} twClasses={twClasses} />
          <MetricsSection labelNode={<><span className={twClasses.presetText}>Preset</span> → <span className={twClasses.audioText}>Audio</span></>} metrics={metrics.preset_to_audio} twClasses={twClasses} />
          <RecallBarChart metrics={avgMetrics} label="Average" twClasses={twClasses} />
        </div>
      ) : !distancesReady ? (
        <div className={`text-xs ${twClasses.textSubtle} text-center py-4`}>
          Waiting for distances...
        </div>
      ) : (
        <div className={`text-xs ${twClasses.textSubtle} text-center py-4`}>
          No data to evaluate
        </div>
      )}
    </div>
  );
}

type TwClasses = ReturnType<typeof getTwClasses>;

function MetricsSection({ label, labelNode, metrics, highlight, twClasses }: { label?: string; labelNode?: React.ReactNode; metrics: DirectionMetrics; highlight?: boolean; twClasses: TwClasses }) {
  return (
    <div className={highlight ? `pt-3 border-t ${twClasses.borderSubtle}` : ''}>
      <div className={`text-xs mb-2 ${highlight ? `${twClasses.textPrimary} font-medium` : twClasses.textMuted}`}>
        {labelNode || label}
      </div>
      <div className="grid grid-cols-4 gap-1 text-center">
        <RecallCell label="@1" value={metrics.recall_at_1} twClasses={twClasses} />
        <RecallCell label="@5" value={metrics.recall_at_5} twClasses={twClasses} />
        <RecallCell label="@10" value={metrics.recall_at_10} twClasses={twClasses} />
        <RecallCell label="@20" value={metrics.recall_at_20} twClasses={twClasses} />
      </div>
    </div>
  );
}

function RecallCell({ label, value, twClasses }: { label: string; value: number; twClasses: TwClasses }) {
  return (
    <div className={`${twClasses.bgElevated} rounded px-1 py-1.5`}>
      <div className={`text-[10px] ${twClasses.textSubtle}`}>{label}</div>
      <div className={`text-xs ${twClasses.textPrimary} font-medium`}>{(value * 100).toFixed(1)}%</div>
    </div>
  );
}

function RecallBarChart({ metrics, label, twClasses }: { metrics: DirectionMetrics; label?: string; twClasses: TwClasses }) {
  const bars = [
    { k: '@1', value: metrics.recall_at_1 },
    { k: '@5', value: metrics.recall_at_5 },
    { k: '@10', value: metrics.recall_at_10 },
    { k: '@20', value: metrics.recall_at_20 },
  ];

  return (
    <div className={`py-3 border-y ${twClasses.borderSubtle}`}>
      {label && <div className={`text-xs ${twClasses.textPrimary} font-medium mb-2`}>{label}</div>}
      <div className="flex items-end gap-1 h-16">
        {bars.map(bar => (
          <div key={bar.k} className="flex-1 flex flex-col items-center">
            <div className="w-full flex flex-col justify-end h-12">
              <div
                className={`w-full ${twClasses.recallBar} rounded-t transition-all duration-300`}
                style={{ height: `${bar.value * 100}%` }}
              />
            </div>
            <div className={`text-[9px] ${twClasses.textSubtle} mt-1`}>{bar.k}</div>
            <div className={`text-[9px] ${twClasses.textMuted}`}>{(bar.value * 100).toFixed(0)}%</div>
          </div>
        ))}
      </div>
    </div>
  );
}
