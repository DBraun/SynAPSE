import { useMemo, useCallback, useState, useEffect, useRef } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, LineLayer } from '@deck.gl/layers';
import { OrthographicView } from '@deck.gl/core';
import type { DataPoint, NeighborResult, ColorBy, PointSizeSettings, RetrievalMode, ModalityFilter } from '../types';
import { DX7PresetPanel } from './DX7PresetPanel';
import { getCarrierCount } from '../dx7';
import { cosineDistance } from '../utils/distance';
import { useTheme } from '../context/ThemeContext';
import { deckColors, getTwClasses, getTheme } from '../theme';

// Strip audio_/code_ prefix from ID since modality is shown separately
function stripPrefix(id: string): string {
  return id.replace(/^(audio_|preset_)/, '');
}

// Workaround for luma.gl initialization race condition
const useDeferredMount = () => {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const timer = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(timer);
  }, []);
  return mounted;
};

interface ScatterPlotProps {
  data: DataPoint[];
  filteredData: DataPoint[];
  selectedPointId: string | null;
  neighbors: NeighborResult[];
  colorBy: ColorBy;
  modalityFilter: ModalityFilter;
  pointSize: PointSizeSettings;
  k: number;
  retrievalMode: RetrievalMode;
  hideFiltered: boolean;
  hoveredPointForDiff: DataPoint | null;
  onPointClick: (id: string | null) => void;
  onPointHover?: (point: DataPoint | null) => void;
  onPointSizeChange: (settings: Partial<PointSizeSettings>) => void;
  onKChange: (k: number) => void;
  onRetrievalModeChange: (mode: RetrievalMode) => void;
  onHideFilteredChange: (hide: boolean) => void;
}

// Color schemes - using centralized theme
const MODALITY_COLORS: Record<string, [number, number, number]> = {
  audio: deckColors.audio,
  preset: deckColors.preset,
};

// Hash algorithm number (0-31) to a hue-based color
function getAlgorithmColor(algorithm0: number): [number, number, number] {
  const hue = (algorithm0 / 32) * 360;
  // HSL to RGB conversion for s=70%, l=55%
  const s = 0.7, l = 0.55;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
  const m = l - c / 2;
  let r = 0, g = 0, b = 0;
  if (hue < 60) { r = c; g = x; }
  else if (hue < 120) { r = x; g = c; }
  else if (hue < 180) { g = c; b = x; }
  else if (hue < 240) { g = x; b = c; }
  else if (hue < 300) { r = x; b = c; }
  else { r = c; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function getCarrierCountColor(count: number): [number, number, number] {
  return deckColors.carrierCount[count] ?? [156, 163, 175];
}

function getModulatorCountColor(count: number): [number, number, number] {
  return deckColors.modulatorCount[count] ?? [156, 163, 175];
}

export function ScatterPlot({
  data,
  filteredData,
  selectedPointId,
  neighbors,
  colorBy,
  modalityFilter,
  pointSize,
  k,
  retrievalMode,
  hideFiltered,
  hoveredPointForDiff,
  onPointClick,
  onPointHover,
  onPointSizeChange,
  onKChange,
  onRetrievalModeChange,
  onHideFilteredChange,
}: ScatterPlotProps) {
  const { mode, isDark } = useTheme();
  const twClasses = getTwClasses(mode);
  const theme = getTheme(mode);
  const mounted = useDeferredMount();
  const containerRef = useRef<HTMLDivElement>(null);
  const initialViewSet = useRef(false);

  const [viewState, setViewState] = useState<{
    target: [number, number, number];
    zoom: number;
  } | null>(null);

  const [hoveredPoint, setHoveredPoint] = useState<DataPoint | null>(null);
  const hoverRafRef = useRef<number | null>(null);
  const [showHelp, setShowHelp] = useState(true);

  // Sorted list of 1-indexed algorithms present in the dataset
  const presentAlgorithms = useMemo(() => {
    const algos = new Set<number>();
    for (const point of data) {
      algos.add(Math.round(point.dx7_preset[125]) + 1);
    }
    return [...algos].sort((a, b) => a - b);
  }, [data]);

  // Plexus effect state
  const [plexusEnabled, setPlexusEnabled] = useState(false);
  const [plexusDistanceRange, setPlexusDistanceRange] = useState<[number, number]>([0, 0.5]);

  // Cosine distance of every audio↔preset ground-truth pair, keyed by pair_id and
  // computed once from the full dataset. Both the histogram and the plexus overlay
  // read from this map so neither recomputes cosine distance when filters change.
  const pairDistanceByPairId = useMemo(() => {
    const pairMap = new Map<string, { audio?: DataPoint; preset?: DataPoint }>();
    for (const point of data) {
      const existing = pairMap.get(point.pair_id) || {};
      if (point.modality === 'audio') {
        existing.audio = point;
      } else {
        existing.preset = point;
      }
      pairMap.set(point.pair_id, existing);
    }

    const distanceByPairId = new Map<string, number>();
    for (const [pairId, pair] of pairMap) {
      if (!pair.audio || !pair.preset) continue;
      distanceByPairId.set(pairId, cosineDistance(pair.audio.embedding, pair.preset.embedding));
    }
    return distanceByPairId;
  }, [data]);

  // Ground truth pair distances for the histogram, sorted ascending (uses all data, not filtered)
  const { allPairDistances, totalPairs } = useMemo(() => {
    const distances = [...pairDistanceByPairId.values()].sort((a, b) => a - b);
    return { allPairDistances: distances, totalPairs: distances.length };
  }, [pairDistanceByPairId]);

  // Compute histogram bins (10 bins from 0 to 1)
  const histogramBins = useMemo(() => {
    const numBins = 10;
    const bins = Array(numBins).fill(0);
    for (const d of allPairDistances) {
      const binIndex = Math.min(Math.floor(d * numBins), numBins - 1);
      bins[binIndex]++;
    }
    const maxCount = Math.max(...bins, 1);
    return bins.map((count, i) => ({
      start: i / numBins,
      end: (i + 1) / numBins,
      count,
      height: count / maxCount,
    }));
  }, [allPairDistances]);

  // Compute initial view to fit all points
  useEffect(() => {
    if (data.length === 0 || initialViewSet.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    if (width === 0 || height === 0) return;

    // Compute bounding box
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const point of data) {
      minX = Math.min(minX, point.x);
      maxX = Math.max(maxX, point.x);
      minY = Math.min(minY, point.y);
      maxY = Math.max(maxY, point.y);
    }

    // Add padding (10%)
    const paddingX = (maxX - minX) * 0.1;
    const paddingY = (maxY - minY) * 0.1;
    minX -= paddingX;
    maxX += paddingX;
    minY -= paddingY;
    maxY += paddingY;

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const dataWidth = maxX - minX;
    const dataHeight = maxY - minY;

    // Calculate zoom to fit (OrthographicView zoom is log2 scale)
    const scaleX = width / dataWidth;
    const scaleY = height / dataHeight;
    const scale = Math.min(scaleX, scaleY);
    const zoom = Math.log2(scale);

    setViewState({
      target: [centerX, centerY, 0],
      zoom,
    });
    initialViewSet.current = true;
  }, [data]);

  // Create a set of filtered IDs for quick lookup
  const filteredIds = useMemo(() => new Set(filteredData.map(p => p.id)), [filteredData]);

  // Get color for a point based on colorBy setting
  const getPointColor = useCallback(
    (point: DataPoint): [number, number, number, number] => {
      const isFiltered = filteredIds.has(point.id);
      const alpha = isFiltered ? 200 : 40;

      let rgb: [number, number, number];

      switch (colorBy) {
        case 'algorithm': {
          // Extract algorithm from the 145-dim preset array (index 125, 0-indexed)
          const algo0 = Math.round(point.dx7_preset[125]);
          rgb = getAlgorithmColor(algo0);
          break;
        }
        case 'carrierCount': {
          const algo0 = Math.round(point.dx7_preset[125]);
          const count = getCarrierCount(algo0);
          rgb = getCarrierCountColor(count);
          break;
        }
        case 'modulatorCount': {
          const algo0 = Math.round(point.dx7_preset[125]);
          const count = 6 - getCarrierCount(algo0);
          rgb = getModulatorCountColor(count);
          break;
        }
        case 'modality':
        default:
          rgb = MODALITY_COLORS[point.modality];
      }

      return [...rgb, alpha];
    },
    [colorBy, filteredIds]
  );

  // Get radius for a point (in world units - will scale with zoom)
  const getPointRadius = useCallback(
    (): number => {
      return pointSize.baseRadius;
    },
    [pointSize.baseRadius]
  );

  // Data to display - either all points or only filtered ones
  // When modality filter is active, always hide the opposite modality
  const displayData = (hideFiltered || modalityFilter !== 'all') ? filteredData : data;

  // Main scatter layer - uses world units so points scale with zoom
  const scatterLayer = new ScatterplotLayer({
    id: 'scatter-layer',
    data: displayData,
    getPosition: (d: DataPoint) => [d.x, d.y, 0],
    getFillColor: getPointColor,
    getRadius: getPointRadius,
    radiusUnits: 'common',  // World-space units, scales with zoom
    radiusMinPixels: pointSize.minPixels,
    radiusMaxPixels: pointSize.maxPixels,
    pickable: true,
    onClick: ({ object }: { object?: DataPoint }) => {
      onPointClick(object?.id || null);
    },
    onHover: ({ object }: { object?: DataPoint }) => {
      if (hoverRafRef.current !== null) cancelAnimationFrame(hoverRafRef.current);
      hoverRafRef.current = requestAnimationFrame(() => {
        hoverRafRef.current = null;
        setHoveredPoint(object || null);
        onPointHover?.(object || null);
      });
    },
    updateTriggers: {
      getFillColor: [colorBy, filteredIds],
      getRadius: [pointSize.baseRadius],
    },
  });

  // Selection ring layer - use dark color in light mode for visibility
  const selectionColor: [number, number, number, number] = isDark
    ? [255, 255, 255, 255]  // white in dark mode
    : [30, 30, 30, 255];    // dark gray in light mode
  const selectionLayer = new ScatterplotLayer({
    id: 'selection-layer',
    data: selectedPointId ? data.filter(p => p.id === selectedPointId) : [],
    getPosition: (d: DataPoint) => [d.x, d.y, 0],
    getFillColor: [0, 0, 0, 0],
    getLineColor: selectionColor,
    getRadius: 1.2,
    radiusUnits: 'common',
    radiusMinPixels: 6,
    radiusMaxPixels: 25,
    stroked: true,
    lineWidthUnits: 'pixels',
    getLineWidth: 2,
    pickable: false,
  });

  // Neighbor highlight layer
  const neighborLayer = new ScatterplotLayer({
    id: 'neighbor-layer',
    data: neighbors.map(n => n.point),
    getPosition: (d: DataPoint) => [d.x, d.y, 0],
    getFillColor: [0, 0, 0, 0],
    getLineColor: (d: DataPoint) => d.modality === 'audio'
      ? [...deckColors.neighborsAudio, 255] as [number, number, number, number]
      : [...deckColors.neighborsPreset, 255] as [number, number, number, number],
    getRadius: 1.1,
    radiusUnits: 'common',
    radiusMinPixels: 5,
    radiusMaxPixels: 20,
    stroked: true,
    lineWidthUnits: 'pixels',
    getLineWidth: 1.5,
    pickable: false,
  });

  // Lines connecting selected point to neighbors
  const selectedPoint = data.find(p => p.id === selectedPointId);
  const lineData = useMemo(() => {
    if (!selectedPoint || neighbors.length === 0) return [];
    return neighbors.map(n => ({
      source: [selectedPoint.x, selectedPoint.y] as [number, number],
      target: [n.point.x, n.point.y] as [number, number],
      modality: n.point.modality,
    }));
  }, [selectedPoint, neighbors]);

  // Plexus effect: lines connecting code-audio pairs within distance range
  // Uses filteredData so connections update when filters are applied
  const plexusLineData = useMemo(() => {
    if (!plexusEnabled) return [];

    // Group filtered points by pair_id
    const pairMap = new Map<string, { audio?: DataPoint; preset?: DataPoint }>();
    for (const point of filteredData) {
      const existing = pairMap.get(point.pair_id) || {};
      if (point.modality === 'audio') {
        existing.audio = point;
      } else {
        existing.preset = point;
      }
      pairMap.set(point.pair_id, existing);
    }

    // Build line data for pairs within distance range
    const lines: { source: [number, number]; target: [number, number]; distance: number }[] = [];
    const [minDist, maxDist] = plexusDistanceRange;

    for (const [pairId, pair] of pairMap) {
      if (!pair.audio || !pair.preset) continue;

      const distance = pairDistanceByPairId.get(pairId);
      if (distance === undefined) continue;
      if (distance >= minDist && distance <= maxDist) {
        lines.push({
          source: [pair.preset.x, pair.preset.y],
          target: [pair.audio.x, pair.audio.y],
          distance,
        });
      }
    }

    return lines;
  }, [plexusEnabled, plexusDistanceRange, filteredData, pairDistanceByPairId]);

  const lineLayer = new LineLayer({
    id: 'line-layer',
    data: lineData,
    getSourcePosition: (d: { source: [number, number] }) => d.source,
    getTargetPosition: (d: { target: [number, number] }) => d.target,
    getColor: (d: { modality: 'audio' | 'preset' }) => d.modality === 'audio'
      ? deckColors.neighborLineAudio
      : deckColors.neighborLinePreset,
    getWidth: 1,
    pickable: false,
  });

  // Ground Truth LineLayer
  const plexusLineLayer = new LineLayer({
    id: 'plexus-line-layer',
    data: plexusLineData,
    getSourcePosition: (d: { source: [number, number] }) => d.source,
    getTargetPosition: (d: { target: [number, number] }) => d.target,
    getColor: deckColors.groundTruthLine,
    getWidth: 1,
    pickable: false,
  });

  const layers = [plexusLineLayer, lineLayer, scatterLayer, neighborLayer, selectionLayer];

  return (
    <div ref={containerRef} className="relative w-full h-full" style={{ backgroundColor: theme.bg.canvas }}>
      {mounted && viewState && (
        <DeckGL
          views={new OrthographicView({ id: 'ortho' })}
          viewState={viewState}
          onViewStateChange={({ viewState: vs }) => setViewState(vs as typeof viewState)}
          controller={{
            keyboard: { moveSpeed: -100 },
          }}
          layers={layers}
          getCursor={({ isHovering }) => (isHovering ? 'pointer' : 'grab')}
        />
      )}

      {/* Tooltip */}
      {hoveredPoint && (
        <div
          className={`absolute pointer-events-none ${twClasses.bgElevated} ${twClasses.textPrimary} px-3 py-2 rounded-lg shadow-lg text-sm z-10`}
          style={{
            left: '50%',
            bottom: '16px',
            transform: 'translateX(-50%)',
          }}
        >
          <div className="font-medium flex items-center gap-1.5">
            <span
              className={`text-xs px-1.5 py-0.5 rounded ${
                hoveredPoint.modality === 'audio'
                  ? twClasses.audioBadge
                  : twClasses.presetBadge
              }`}
            >
              {hoveredPoint.modality === 'audio' ? 'audio' : 'preset'}
            </span>
            {stripPrefix(hoveredPoint.id)}
          </div>
        </div>
      )}

      {/* DX7 Preset panel - shows preset parameters */}
      {selectedPoint && (
        <DX7PresetPanel
          selectedPoint={selectedPoint}
          hoveredPoint={hoveredPoint || hoveredPointForDiff}
          onClose={() => onPointClick(null)}
        />
      )}

      {/* Legend & Stats */}
      <div className={`absolute bottom-4 left-4 ${twClasses.bgElevatedSemi} px-3 py-2 rounded-lg text-xs`}>
        {colorBy === 'modality' && (
          <div className="flex items-center gap-3 mb-1">
            <div className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${twClasses.audioDot}`}></span>
              <span className={twClasses.textSecondary}>Audio</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className={`w-2.5 h-2.5 rounded-full ${twClasses.presetDot}`}></span>
              <span className={twClasses.textSecondary}>Preset</span>
            </div>
          </div>
        )}
        {colorBy === 'carrierCount' && (
          <div className="flex items-center gap-3 mb-1 flex-wrap">
            {[1, 2, 3, 4, 5, 6].map(n => {
              const color = getCarrierCountColor(n);
              return (
                <div key={n} className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})` }}
                  ></span>
                  <span className={twClasses.textSecondary}>{n}</span>
                </div>
              );
            })}
          </div>
        )}
        {colorBy === 'modulatorCount' && (
          <div className="flex items-center gap-3 mb-1 flex-wrap">
            {[0, 1, 2, 3, 4, 5].map(n => {
              const color = getModulatorCountColor(n);
              return (
                <div key={n} className="flex items-center gap-1.5">
                  <span
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})` }}
                  ></span>
                  <span className={twClasses.textSecondary}>{n}</span>
                </div>
              );
            })}
          </div>
        )}
        {colorBy === 'algorithm' && (
          <div className="mb-1">
            <div className="grid gap-x-3 gap-y-1" style={{ gridTemplateColumns: 'repeat(4, auto)' }}>
              {presentAlgorithms.map(algo1 => {
                const color = getAlgorithmColor(algo1 - 1);
                return (
                  <div key={algo1} className="flex items-center gap-1">
                    <span
                      className="w-2.5 h-2.5 rounded-full"
                      style={{ backgroundColor: `rgb(${color[0]}, ${color[1]}, ${color[2]})` }}
                    ></span>
                    <span className={twClasses.textSecondary}>{algo1}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        <div className={twClasses.textMuted}>
          {filteredData.length} / {data.length} points
        </div>
      </div>

      {/* Controls Panel */}
      <div className={`absolute top-4 left-4 ${twClasses.bgElevatedSemi} px-3 py-2 rounded-lg text-xs w-44`}>
        {/* K Slider */}
        <div className="mb-3">
          <label className={`block ${twClasses.textSecondary} font-medium mb-1`}>
            Neighbors (K): {k}
          </label>
          <input
            type="range"
            min="1"
            max="50"
            value={k}
            onChange={(e) => onKChange(parseInt(e.target.value))}
            className={`w-full h-1 ${twClasses.sliderTrack} rounded-lg appearance-none cursor-pointer ${twClasses.sliderAccent}`}
          />
        </div>

        {/* Retrieval Mode */}
        <div className="mb-3">
          <div className={`${twClasses.textSecondary} font-medium mb-1`}>Retrieval</div>
          <div className="flex gap-3">
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="retrievalMode"
                value="cross"
                checked={retrievalMode === 'cross'}
                onChange={() => onRetrievalModeChange('cross')}
                className={twClasses.sliderAccent}
              />
              <span className={twClasses.textMuted}>Cross</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer">
              <input
                type="radio"
                name="retrievalMode"
                value="intra"
                checked={retrievalMode === 'intra'}
                onChange={() => onRetrievalModeChange('intra')}
                className={twClasses.sliderAccent}
              />
              <span className={twClasses.textMuted}>Intra</span>
            </label>
          </div>
        </div>

        {/* Point Size */}
        <div className={`${twClasses.borderSubtle} border-t pt-2`}>
          <div className={`${twClasses.textSecondary} font-medium mb-1`}>Point Size</div>
          <div className="space-y-1.5">
            <div>
              <label className={`block ${twClasses.textMuted} mb-0.5`}>
                Radius: {pointSize.baseRadius.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="2"
                step="0.05"
                value={pointSize.baseRadius}
                onChange={(e) => onPointSizeChange({ baseRadius: parseFloat(e.target.value) })}
                className={`w-full h-1 ${twClasses.sliderTrack} rounded-lg appearance-none cursor-pointer ${twClasses.sliderAccent}`}
              />
            </div>
            <div>
              <label className={`block ${twClasses.textMuted} mb-0.5`}>
                Min px: {pointSize.minPixels}
              </label>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={pointSize.minPixels}
                onChange={(e) => onPointSizeChange({ minPixels: parseInt(e.target.value) })}
                className={`w-full h-1 ${twClasses.sliderTrack} rounded-lg appearance-none cursor-pointer ${twClasses.sliderAccent}`}
              />
            </div>
            <div>
              <label className={`block ${twClasses.textMuted} mb-0.5`}>
                Max px: {pointSize.maxPixels}
              </label>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={pointSize.maxPixels}
                onChange={(e) => onPointSizeChange({ maxPixels: parseInt(e.target.value) })}
                className={`w-full h-1 ${twClasses.sliderTrack} rounded-lg appearance-none cursor-pointer ${twClasses.sliderAccent}`}
              />
            </div>
            <label className="flex items-center gap-2 cursor-pointer mt-2">
              <input
                type="checkbox"
                checked={hideFiltered}
                onChange={(e) => onHideFilteredChange(e.target.checked)}
                className={twClasses.checkbox}
              />
              <span className={twClasses.textMuted}>Hide filtered</span>
            </label>
          </div>
        </div>

        {/* Ground Truth Links */}
        <div className={`${twClasses.borderSubtle} border-t pt-2 mt-2`}>
          <label className="flex items-center gap-2 cursor-pointer mb-2">
            <input
              type="checkbox"
              checked={plexusEnabled}
              onChange={(e) => setPlexusEnabled(e.target.checked)}
              className={twClasses.checkbox}
            />
            <span className={`${twClasses.textSecondary} font-medium`}>Ground Truth</span>
          </label>
          {plexusEnabled && (
            <div className="space-y-1.5">
              <div className={`${twClasses.textMuted} mb-1`}>
                Distance: {plexusDistanceRange[0].toFixed(2)} - {plexusDistanceRange[1].toFixed(2)}
              </div>
              <div className="relative h-6">
                {/* Track background */}
                <div className={`absolute top-1/2 -translate-y-1/2 w-full h-1 ${twClasses.sliderTrack} rounded-lg`} />
                {/* Active range highlight */}
                <div
                  className={`absolute top-1/2 -translate-y-1/2 h-1 ${twClasses.rangeSliderTrack} rounded-lg pointer-events-none`}
                  style={{
                    left: `${plexusDistanceRange[0] * 100}%`,
                    width: `${(plexusDistanceRange[1] - plexusDistanceRange[0]) * 100}%`,
                  }}
                />
                {/* Min slider - pointer-events:none on track, auto on thumb */}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={plexusDistanceRange[0]}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (val <= plexusDistanceRange[1]) {
                      setPlexusDistanceRange([val, plexusDistanceRange[1]]);
                    }
                  }}
                  className={twClasses.rangeSliderThumb}
                />
                {/* Max slider - pointer-events:none on track, auto on thumb */}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={plexusDistanceRange[1]}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value);
                    if (val >= plexusDistanceRange[0]) {
                      setPlexusDistanceRange([plexusDistanceRange[0], val]);
                    }
                  }}
                  className={twClasses.rangeSliderThumb}
                />
              </div>
              <div className={`${twClasses.textSubtle} text-[10px] mt-1`}>
                {plexusLineData.length}/{totalPairs} connections ({totalPairs > 0 ? ((plexusLineData.length / totalPairs) * 100).toFixed(0) : 0}%)
              </div>
              {/* Distance histogram */}
              <div className="mt-3">
                <div className={`${twClasses.textSubtle} text-[10px] mb-1`}>Distance distribution</div>
                <div className="flex items-end gap-px h-8">
                  {histogramBins.map((bin, i) => {
                    const inRange = bin.start < plexusDistanceRange[1] && bin.end > plexusDistanceRange[0];
                    return (
                      <div
                        key={i}
                        className={`flex-1 ${inRange ? 'bg-purple-500' : twClasses.histogramBar} rounded-t transition-colors`}
                        style={{ height: `${Math.max(bin.height * 100, 2)}%` }}
                        title={`${bin.start.toFixed(1)}-${bin.end.toFixed(1)}: ${bin.count}`}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between text-[8px] mt-0.5">
                  <span className={twClasses.textSubtle}>0</span>
                  <span className={twClasses.textSubtle}>1</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Help panel */}
      {showHelp && (
        <div className={`absolute bottom-4 right-4 ${twClasses.bgElevatedSemi} px-3 py-2 rounded-lg text-xs ${twClasses.textMuted}`}>
          <div className="flex items-center justify-between mb-1">
            <div className={`font-medium ${twClasses.textSecondary}`}>Controls</div>
            <button
              onClick={() => setShowHelp(false)}
              className={twClasses.buttonGhost + ' ml-2 leading-none'}
            >
              ×
            </button>
          </div>
          <div className="space-y-0.5">
            <div><span className={twClasses.textSubtle}>Drag</span> Pan</div>
            <div><span className={twClasses.textSubtle}>Scroll</span> Zoom</div>
            <div><span className={twClasses.textSubtle}>Click point</span> Select + load preset</div>
          </div>
          <div className={`font-medium ${twClasses.textSecondary} mt-2 mb-0.5`}>Keyboard</div>
          <div className="space-y-0.5">
            <div><span className={twClasses.textSubtle}>A S D ... :</span> White keys (C D E ...)</div>
            <div><span className={twClasses.textSubtle}>W E T Y U O P:</span> Black keys</div>
            <div><span className={twClasses.textSubtle}>Z / X</span> Octave down / up</div>
          </div>
        </div>
      )}
    </div>
  );
}
