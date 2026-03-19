import { useState } from 'react';
import type { ColorBy, ModalityFilter } from '../types';
import { useTheme } from '../context/ThemeContext';
import { getTwClasses } from '../theme';

interface ControlPanelProps {
  textFilter: string;
  activeTags: string[];
  availableTags: string[];
  colorBy: ColorBy;
  modalityFilter: ModalityFilter;
  onTextFilterChange: (filter: string) => void;
  onToggleTag: (tag: string) => void;
  onColorByChange: (colorBy: ColorBy) => void;
  onModalityFilterChange: (filter: ModalityFilter) => void;
}

export function ControlPanel({
  // textFilter / onTextFilterChange are intentionally not destructured here:
  // the text-filter input below is disabled until real preset names exist.
  activeTags,
  availableTags,
  colorBy,
  modalityFilter,
  onToggleTag,
  onColorByChange,
  onModalityFilterChange,
}: ControlPanelProps) {
  const { mode } = useTheme();
  const twClasses = getTwClasses(mode);
  const [showAllTags, setShowAllTags] = useState(false);
  const displayedTags = showAllTags ? availableTags : availableTags.slice(0, 10);

  return (
    <div className="p-4 space-y-6">
      <div>
        <h2 className={`text-lg font-semibold ${twClasses.textPrimary} mb-4`}>Filters</h2>
      </div>

      {/* Text Filter — disabled until real preset names are available
      <div>
        <label className={`block text-sm font-medium ${twClasses.textSecondary} mb-2`}>
          Preset Filter
        </label>
        <input
          type="text"
          value={textFilter}
          onChange={(e) => onTextFilterChange(e.target.value)}
          placeholder="e.g., BRASS AND NOT PIANO"
          className={`w-full ${twClasses.bgInput} ${twClasses.textPrimary} rounded-lg px-3 py-2 text-sm border ${twClasses.borderInput} ${twClasses.inputFocus} focus:outline-none placeholder-gray-500`}
        />
        <p className={`text-xs ${twClasses.textSubtle} mt-1`}>
          Searches preset names. Supports AND, OR, NOT
        </p>
      </div>
      */}

      {/* Tag Filter */}
      <div>
        <label className={`block text-sm font-medium ${twClasses.textSecondary} mb-2`}>
          Tag Filter
        </label>
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {displayedTags.map((tag) => (
            <label
              key={tag}
              className={`flex items-center gap-2 cursor-pointer ${twClasses.bgElevatedHover}/50 px-2 py-1 rounded`}
            >
              <input
                type="checkbox"
                checked={activeTags.includes(tag)}
                onChange={() => onToggleTag(tag)}
                className={twClasses.checkbox}
              />
              <span className={`text-sm ${twClasses.textSecondary}`}>{tag}</span>
            </label>
          ))}
        </div>
        {availableTags.length > 10 && (
          <button
            onClick={() => setShowAllTags(!showAllTags)}
            className={`text-xs ${twClasses.audioText} hover:opacity-80 mt-2`}
          >
            {showAllTags ? 'Show less' : `Show all (${availableTags.length})`}
          </button>
        )}
        {activeTags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {activeTags.map((tag) => (
              <span
                key={tag}
                className={`inline-flex items-center gap-1 px-2 py-0.5 ${twClasses.audioBadge} rounded text-xs`}
              >
                {tag}
                <button
                  onClick={() => onToggleTag(tag)}
                  className="hover:text-white"
                >
                  &times;
                </button>
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Color By */}
      <div>
        <label className={`block text-sm font-medium ${twClasses.textSecondary} mb-2`}>
          Color By
        </label>
        <select
          value={colorBy}
          onChange={(e) => onColorByChange(e.target.value as ColorBy)}
          className={`w-full ${twClasses.bgInput} ${twClasses.textPrimary} rounded-lg px-3 py-2 text-sm border ${twClasses.borderInput} ${twClasses.inputFocus} focus:outline-none`}
        >
          <option value="modality">Modality</option>
          <option value="algorithm">Algorithm</option>
          <option value="carrierCount">Carrier Count</option>
          <option value="modulatorCount">Modulator Count</option>
        </select>
      </div>

      {/* Modality Filter */}
      <div>
        <label className={`block text-sm font-medium ${twClasses.textSecondary} mb-2`}>
          Show Points
        </label>
        <select
          value={modalityFilter}
          onChange={(e) => onModalityFilterChange(e.target.value as ModalityFilter)}
          className={`w-full ${twClasses.bgInput} ${twClasses.textPrimary} rounded-lg px-3 py-2 text-sm border ${twClasses.borderInput} ${twClasses.inputFocus} focus:outline-none`}
        >
          <option value="all">All (Audio + Preset)</option>
          <option value="audio">Audio Only</option>
          <option value="preset">Preset Only</option>
        </select>
      </div>
    </div>
  );
}
