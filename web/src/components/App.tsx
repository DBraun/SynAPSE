import { useState, useMemo, useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { useTheme } from '../context/ThemeContext';
import { useNearestNeighbors } from '../hooks/useNearestNeighbors';
import { useDX7Synth } from '../hooks/useDX7Synth';
import { ScatterPlot } from './ScatterPlot';
import { ControlPanel } from './ControlPanel';
import { ResultsTable } from './ResultsTable';
import { MetricsPanel } from './MetricsPanel';
import { InfoModal } from './InfoModal';
// TODO: render <SysexUploadPanel onPresetReady={...} /> once the TensorFlow.js
// model can embed an uploaded DX7 preset and project it into the scatter plot.
// import { SysexUploadPanel } from './SysexUploadPanel';
import type { DataPoint } from '../types';
import { getTwClasses, getTheme } from '../theme';

export function App() {
  const {
    state,
    setSelectedPoint,
    setK,
    setRetrievalMode,
    setTextFilter,
    toggleTag,
    setColorBy,
    setModalityFilter,
    setPointSize,
    setHideFiltered,
    getDistance,
    distancesReady,
    precomputedMetrics,
  } = useApp();

  const { mode, toggleTheme, isDark } = useTheme();
  const twClasses = getTwClasses(mode);
  const theme = getTheme(mode);

  const dx7Synth = useDX7Synth();

  // Play a DX7 preset from a data point
  const playPreset = useCallback((point: DataPoint) => {
    if (point.dx7_preset) {
      dx7Synth.loadPreset(point.dx7_preset, point.preset_name);
      dx7Synth.playNote();
    }
  }, [dx7Synth]);

  const handlePointClick = (id: string | null) => {
    setSelectedPoint(id);

    if (id !== null) {
      const point = state.data.find((p) => p.id === id);
      if (point?.dx7_preset) {
        // Always load the preset into the synth so keyboard playback works
        dx7Synth.loadPreset(point.dx7_preset, point.preset_name);
        // Auto-play if enabled
        if (state.autoPlay) {
          dx7Synth.playNote();
        }
      }
    }
  };

  const distanceFn = distancesReady ? getDistance : undefined;

  const neighbors = useNearestNeighbors(
    state.selectedPointId,
    state.data,
    state.filteredData,
    state.k,
    state.retrievalMode,
    distanceFn
  );

  const selectedPoint = state.selectedPointId
    ? state.data.find((p) => p.id === state.selectedPointId) || null
    : null;

  // Compute ground truth rank (where the paired point ranks among filtered candidates)
  const groundTruthRank = useMemo(() => {
    if (!selectedPoint || !distancesReady || !getDistance) return null;

    // Find the ground truth paired point
    const pairedPoint = state.data.find(
      p => p.pair_id === selectedPoint.pair_id && p.modality !== selectedPoint.modality
    );
    if (!pairedPoint) return null;

    // Get filtered candidates (opposite modality for cross-modal, same for intra)
    const candidates = state.filteredData.filter(p => {
      if (p.id === selectedPoint.id) return false;
      if (state.retrievalMode === 'cross') {
        return p.modality !== selectedPoint.modality;
      } else {
        return p.modality === selectedPoint.modality;
      }
    });

    // For intra-modal, ground truth doesn't apply
    if (state.retrievalMode === 'intra') return null;

    // Check if paired point is in filtered data
    const pairedPointInFiltered = candidates.some(c => c.id === pairedPoint.id);
    if (!pairedPointInFiltered) return null;

    // Sort candidates by distance and find rank
    const gtDistance = getDistance(selectedPoint.id, pairedPoint.id);
    let rank = 1;
    for (const candidate of candidates) {
      if (candidate.id === pairedPoint.id) continue;
      const dist = getDistance(selectedPoint.id, candidate.id);
      if (dist < gtDistance) rank++;
    }

    return { rank, total: candidates.length };
  }, [selectedPoint, state.filteredData, state.data, state.retrievalMode, distancesReady, getDistance]);

  // Track hovered point from table for diff view
  const [hoveredPointForDiff, setHoveredPointForDiff] = useState<DataPoint | null>(null);


  if (state.isLoading) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${twClasses.textPrimary}`} style={{ backgroundColor: theme.bg.canvas }}>
        <div className="text-center">
          <div className={`animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 ${twClasses.spinner} mx-auto mb-4`}></div>
          <p className={twClasses.textMuted}>Loading embedding data...</p>
        </div>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className={`min-h-screen flex items-center justify-center ${twClasses.textPrimary}`} style={{ backgroundColor: theme.bg.canvas }}>
        <div className="text-center max-w-md">
          <div className={`${twClasses.errorIcon} text-5xl mb-4`}>!</div>
          <h2 className={`text-xl font-semibold ${twClasses.textPrimary} mb-2`}>
            Failed to Load Data
          </h2>
          <p className={twClasses.textMuted}>{state.error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`h-screen flex ${twClasses.textPrimary}`} style={{ backgroundColor: theme.bg.canvas }}>
      {/* Left sidebar - Controls */}
      <aside className={`w-72 border-r ${twClasses.borderColor} overflow-y-auto flex flex-col`} style={{ backgroundColor: theme.bg.panel }}>
        {/* Sidebar header with title */}
        <div className={`flex-shrink-0 px-4 py-3 border-b ${twClasses.borderColor}`}>
          <h1 className="text-lg font-bold">SynAPSE</h1>
          <div className={`text-xs ${twClasses.textMuted}`}>
            Synthesizer <span className={twClasses.audioText}>Audio</span> <span className={twClasses.presetText}>Preset</span> Embeddings
          </div>
          <div className="flex items-center gap-2 mt-2">
            <InfoModal />
            <button
              onClick={toggleTheme}
              className={`p-1.5 rounded-lg ${twClasses.buttonSecondary} transition-colors`}
              title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <ControlPanel
            textFilter={state.textFilter}
            activeTags={state.activeTags}
            availableTags={state.availableTags}
            colorBy={state.colorBy}
            modalityFilter={state.modalityFilter}
            onTextFilterChange={setTextFilter}
            onToggleTag={toggleTag}
            onColorByChange={setColorBy}
            onModalityFilterChange={setModalityFilter}
          />

          {/* Upload .syx panel — temporarily hidden
          <div className={`border-t ${twClasses.borderColor}`}>
            <SysexUploadPanel
              onPresetReady={(point) => {
                dispatch({ type: 'ADD_UPLOADED_POINT', payload: point });
                playPreset(point);
              }}
            />
          </div>
          */}

          {/* Metrics panel at bottom of sidebar */}
          <div className={`border-t ${twClasses.borderColor}`}>
            <MetricsPanel
              filteredData={state.filteredData}
              allData={state.data}
              distancesReady={distancesReady}
              precomputedMetrics={precomputedMetrics}
            />
          </div>
        </div>
      </aside>

      {/* Center - Scatter plot */}
      <main className="flex-1">
        <ScatterPlot
          data={state.data}
          filteredData={state.filteredData}
          selectedPointId={state.selectedPointId}
          neighbors={neighbors}
          colorBy={state.colorBy}
          modalityFilter={state.modalityFilter}
          pointSize={state.pointSize}
          k={state.k}
          retrievalMode={state.retrievalMode}
          hideFiltered={state.hideFiltered}
          hoveredPointForDiff={hoveredPointForDiff}
          onPointClick={handlePointClick}
          onPointSizeChange={setPointSize}
          onKChange={setK}
          onRetrievalModeChange={setRetrievalMode}
          onHideFilteredChange={setHideFiltered}
        />
      </main>

      {/* Right sidebar - Results */}
      <aside className={`w-64 border-l ${twClasses.borderColor} overflow-hidden flex flex-col`} style={{ backgroundColor: theme.bg.panel }}>
        <ResultsTable
          selectedPoint={selectedPoint}
          neighbors={neighbors}
          groundTruthRank={groundTruthRank}
          onRowClick={handlePointClick}
          onRowHover={setHoveredPointForDiff}
          onPlayPreset={playPreset}
        />
      </aside>

    </div>
  );
}
