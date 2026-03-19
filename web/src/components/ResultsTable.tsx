import type { NeighborResult, DataPoint } from '../types';
import { useTheme } from '../context/ThemeContext';
import { useApp } from '../context/AppContext';
import { getTwClasses } from '../theme';

// Strip audio_/preset_ prefix from ID since modality is shown separately
function stripPrefix(id: string): string {
  return id.replace(/^(audio_|preset_)/, '');
}

interface ResultsTableProps {
  selectedPoint: DataPoint | null;
  neighbors: NeighborResult[];
  groundTruthRank: { rank: number; total: number } | null;
  onRowClick: (id: string) => void;
  onRowHover: (point: DataPoint | null) => void;
  onPlayPreset?: (point: DataPoint) => void;
}

export function ResultsTable({ selectedPoint, neighbors, groundTruthRank, onRowClick, onRowHover, onPlayPreset }: ResultsTableProps) {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const { state, setAutoPlay } = useApp();

  if (!selectedPoint) {
    return (
      <div className={`p-4 text-center ${twClasses.textSubtle}`}>
        <p>Click a point on the scatter plot to see its nearest neighbors</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-x-hidden">
      {/* Selected point info */}
      <div className={`p-4 border-b ${twClasses.borderSubtle}`}>
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span
              className={`text-xs px-1.5 py-0.5 rounded ${
                selectedPoint.modality === 'audio'
                  ? twClasses.audioBadge
                  : twClasses.presetBadge
              }`}
            >
              {selectedPoint.modality}
            </span>
            <span className={`font-medium ${twClasses.textPrimary}`}>{stripPrefix(selectedPoint.id)}</span>
          </div>
          <span className={`text-xs font-medium ${twClasses.textMuted} uppercase tracking-wide`}>query</span>
        </div>
        <div className="flex items-center gap-3 mb-2">
          <div className="flex-1">
            {onPlayPreset && (
              <button
                onClick={() => onPlayPreset(selectedPoint)}
                className={`px-3 py-1.5 rounded text-xs ${twClasses.buttonSecondary} transition-colors`}
                title="Play this preset"
              >
                Play
              </button>
            )}
          </div>
          <label className="flex items-center gap-1.5 cursor-pointer" title="Auto-play when selecting a new point">
            <input
              type="checkbox"
              checked={state.autoPlay}
              onChange={(e) => setAutoPlay(e.target.checked)}
              className="w-3.5 h-3.5 rounded border-gray-300 text-blue-600 focus:ring-blue-500 cursor-pointer"
            />
            <span className={`text-xs ${twClasses.textMuted} whitespace-nowrap`}>Auto</span>
          </label>
        </div>
        {/* Preset name */}
        <div className="mt-2 text-xs">
          <span className={twClasses.textMuted}>Preset: </span>
          <span className={`font-medium ${twClasses.textSecondary}`}>
            {selectedPoint.preset_name}
          </span>
        </div>
        {/* Algorithm */}
        <div className="mt-1 text-xs">
          <span className={twClasses.textMuted}>Algorithm: </span>
          <span className={`font-medium ${twClasses.textSecondary}`}>
            {Math.round(selectedPoint.dx7_preset[125]) + 1}
          </span>
        </div>
        {/* Ground truth retrieval rank indicator */}
        {groundTruthRank !== null && (
          <div className="mt-2 text-xs">
            <span className={twClasses.textMuted}>Ground truth rank: </span>
            <span className={`font-medium ${groundTruthRank.rank === 1 ? twClasses.rankGood : groundTruthRank.rank <= 5 ? twClasses.rankOk : twClasses.rankPoor}`}>
              #{groundTruthRank.rank} of {groundTruthRank.total}
            </span>
          </div>
        )}
        {selectedPoint.tags.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {selectedPoint.tags.map((tag) => (
              <span
                key={tag}
                className={`px-1.5 py-0.5 ${twClasses.bgInput} ${twClasses.textSecondary} rounded text-xs`}
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Neighbors table */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className={`text-sm font-medium ${twClasses.textSecondary} px-4 py-2 border-b ${twClasses.borderSubtle}`}>
          {neighbors.length} Nearest Neighbors
        </div>
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          <table className="w-full text-sm table-fixed">
            <thead className={`${twClasses.bgElevated} sticky top-0`}>
              <tr className={`text-left ${twClasses.textMuted}`}>
                <th className="pl-2 pr-1 py-2 font-medium w-6">#</th>
                <th className="px-1 py-2 font-medium w-14">ID</th>
                <th className="px-1 py-2 font-medium w-11">Type</th>
                <th className="px-1 py-2 font-medium w-12">Dist</th>
                <th className="pl-1 pr-2 py-2 font-medium w-8"></th>
              </tr>
            </thead>
            <tbody className={`divide-y ${twClasses.borderSubtle.replace('border', 'divide')}`}>
              {neighbors.map((neighbor) => (
                <tr
                  key={neighbor.point.id}
                  onClick={() => onRowClick(neighbor.point.id)}
                  onMouseEnter={() => onRowHover(neighbor.point)}
                  onMouseLeave={() => onRowHover(null)}
                  className={`cursor-pointer ${twClasses.bgElevatedHover}/50 ${
                    neighbor.point.pair_id === selectedPoint.pair_id
                      ? twClasses.groundTruthRow
                      : ''
                  }`}
                >
                  <td className={`pl-2 pr-1 py-2 ${twClasses.textMuted}`}>{neighbor.rank}</td>
                  <td className={`px-1 py-2 ${twClasses.textPrimary} font-mono text-xs truncate`}>
                    {stripPrefix(neighbor.point.id)}
                    {neighbor.point.pair_id === selectedPoint.pair_id && (
                      <span className={`ml-1 text-xs ${twClasses.groundTruthMarker}`}>*</span>
                    )}
                  </td>
                  <td className="px-1 py-2">
                    <span
                      className={`text-xs px-1 py-0.5 rounded ${
                        neighbor.point.modality === 'audio'
                          ? twClasses.audioBadge
                          : twClasses.presetBadge
                      }`}
                    >
                      {neighbor.point.modality}
                    </span>
                  </td>
                  <td className={`px-1 py-2 ${twClasses.textMuted} font-mono text-xs`}>
                    {neighbor.distance.toFixed(3)}
                  </td>
                  <td className="pl-1 pr-2 py-2">
                    {onPlayPreset && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onPlayPreset(neighbor.point);
                        }}
                        className={`w-6 h-6 flex items-center justify-center rounded-full border border-current ${twClasses.textMuted} text-xs`}
                        title="Play preset"
                      >
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M8 5v14l11-7z" />
                        </svg>
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
