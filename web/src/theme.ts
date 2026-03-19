/**
 * Centralized theme configuration for PEACE Map
 *
 * Supports both dark and light themes.
 * To change colors globally, update the values here.
 */

export type ThemeMode = 'dark' | 'light';

// Shared colors (same in both themes)
const sharedColors = {
  // Modality colors
  audio: {
    primary: '#3B82F6',    // blue-500
    light: '#60A5FA',      // blue-400
    lighter: '#93C5FD',    // blue-300
    bg: 'rgba(59, 130, 246, 0.2)',
  },
  preset: {
    primary: '#F97316',    // orange-500
    light: '#FB923C',      // orange-400
    lighter: '#FDBA74',    // orange-300
    bg: 'rgba(249, 115, 22, 0.2)',
  },
  // Status colors
  success: {
    primary: '#22c55e',    // green-500
    light: '#4ade80',      // green-400
    bg: 'rgba(34, 197, 94, 0.3)',
  },
  warning: {
    primary: '#eab308',    // yellow-500
    light: '#facc15',      // yellow-400
  },
  error: {
    primary: '#ef4444',    // red-500
    light: '#f87171',      // red-400
  },
  // Feature colors
  neighbors: {
    audio: '#60A5FA',
    audioRgb: [96, 165, 250] as [number, number, number],
    preset: '#FB923C',
    presetRgb: [251, 146, 60] as [number, number, number],
  },
  groundTruthLines: {
    primary: '#A855F7',    // purple-500
    rgb: [168, 85, 247] as [number, number, number],
  },
  // UI accent
  accent: {
    primary: '#3B82F6',    // blue-500
    hover: '#2563EB',      // blue-600
  },
};

// Dark theme backgrounds
const darkBg = {
  canvas: '#111827',     // gray-900 for scatterplot
  panel: '#1F2937',      // gray-800 for panels
  elevated: '#374151',   // gray-700 for elevated elements
  border: '#4B5563',     // gray-600 for borders
};

// Light theme backgrounds
const lightBg = {
  canvas: '#F3F4F6',     // gray-100 for scatterplot
  panel: '#FFFFFF',      // white for panels
  elevated: '#F9FAFB',   // gray-50 for elevated elements
  border: '#E5E7EB',     // gray-200 for borders
};

// Theme object factory
export const themes = {
  dark: {
    ...sharedColors,
    bg: darkBg,
  },
  light: {
    ...sharedColors,
    bg: lightBg,
  },
} as const;

// Default export (dark theme for backwards compatibility)
export const theme = themes.dark;

// Get theme by mode
export function getTheme(mode: ThemeMode) {
  return themes[mode];
}

// RGB values for deck.gl layers (ScatterPlot)
export const deckColors = {
  audio: [59, 130, 246] as [number, number, number],      // blue-500
  preset: [249, 115, 22] as [number, number, number],       // orange-500
  neighborsAudio: [96, 165, 250] as [number, number, number],   // blue-400
  neighborsPreset: [251, 146, 60] as [number, number, number],    // orange-400
  neighborLineAudio: [96, 165, 250, 100] as [number, number, number, number],
  neighborLinePreset: [251, 146, 60, 100] as [number, number, number, number],
  groundTruthLine: [168, 85, 247, 120] as [number, number, number, number],  // purple-500
  selection: [255, 255, 255] as [number, number, number], // white

  // Carrier count colors
  carrierCount: {
    1: [168, 85, 247] as [number, number, number],   // purple-500
    2: [59, 130, 246] as [number, number, number],    // blue-500
    3: [34, 197, 94] as [number, number, number],     // green-500
    4: [234, 179, 8] as [number, number, number],     // yellow-500
    5: [249, 115, 22] as [number, number, number],    // orange-500
    6: [239, 68, 68] as [number, number, number],     // red-500
  } as Record<number, [number, number, number]>,

  // Modulator count colors (6 - carriers)
  modulatorCount: {
    0: [156, 163, 175] as [number, number, number],   // gray-400
    1: [168, 85, 247] as [number, number, number],    // purple-500
    2: [59, 130, 246] as [number, number, number],    // blue-500
    3: [34, 197, 94] as [number, number, number],     // green-500
    4: [234, 179, 8] as [number, number, number],     // yellow-500
    5: [249, 115, 22] as [number, number, number],    // orange-500
  } as Record<number, [number, number, number]>,
} as const;

// Dark theme Tailwind classes
const darkTwClasses = {
  // Modality badges
  audioBadge: 'bg-blue-500/20 text-blue-300',
  presetBadge: 'bg-orange-500/20 text-orange-300',

  // Modality text colors (for labels, legends)
  audioText: 'text-blue-400',
  presetText: 'text-orange-400',

  // Modality legend dots
  audioDot: 'bg-blue-500',
  presetDot: 'bg-orange-500',

  // Rank indicators
  rankGood: 'text-green-400',
  rankOk: 'text-yellow-400',
  rankPoor: 'text-red-400',

  // Ground truth highlight in table
  groundTruthRow: 'bg-green-900/20',
  groundTruthMarker: 'text-green-400',

  // Form elements
  checkbox: 'rounded bg-gray-700 border-gray-600 accent-blue-500 focus:ring-blue-500 focus:ring-offset-0',
  inputFocus: 'focus:border-blue-500',
  sliderAccent: 'accent-blue-500',

  // Buttons
  buttonPrimary: 'bg-blue-600 hover:bg-blue-500 text-white',
  buttonActive: 'bg-blue-500 text-white',

  // Loading/progress
  spinner: 'border-blue-500',
  progressBar: 'bg-blue-500',

  // Recall bar chart color
  recallBar: 'bg-blue-500',

  // Ground truth range slider
  rangeSliderTrack: 'bg-purple-500/50',
  histogramBar: 'bg-gray-600',
  rangeSliderThumb: 'absolute top-0 w-full h-6 appearance-none bg-transparent cursor-pointer pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-purple-400 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:bg-purple-400 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer',

  // Carrier count legend dots
  carriers1Dot: 'bg-purple-500',
  carriers2Dot: 'bg-blue-500',
  carriers3Dot: 'bg-green-500',
  carriers4Dot: 'bg-yellow-500',
  carriers5Dot: 'bg-orange-500',
  carriers6Dot: 'bg-red-500',

  // Background colors
  bgPrimary: 'bg-gray-900',
  bgSecondary: 'bg-gray-800',
  bgTertiary: 'bg-gray-700',
  borderColor: 'border-gray-600',

  // Text colors
  textPrimary: 'text-white',
  textSecondary: 'text-gray-300',
  textMuted: 'text-gray-400',
  textSubtle: 'text-gray-500',

  // Warning popup
  warningBg: 'bg-gray-800',
  warningBorder: 'border-yellow-600',
  warningIcon: 'text-yellow-500',
  warningTitle: 'text-yellow-400',
  warningText: 'text-gray-300',
  warningButton: 'bg-yellow-600 hover:bg-yellow-500 text-white',

  // Error state
  errorIcon: 'text-red-500',

  // Code diff colors
  diffAddedBg: 'bg-green-900/30',
  diffRemovedBg: 'bg-red-900/30',
  diffAddedText: 'text-green-300',
  diffRemovedText: 'text-red-300',

  // Info modal highlight section
  infoHighlightBg: 'bg-blue-900/30',
  infoHighlightBorder: 'border-blue-700',
  infoHighlightTitle: 'text-blue-300',

  // Elevated UI elements
  bgElevated: 'bg-neutral-800',
  bgElevatedSemi: 'bg-neutral-800/90',
  bgElevatedHover: 'hover:bg-neutral-700',
  bgInput: 'bg-neutral-700',
  bgInputHover: 'hover:bg-neutral-600',

  // Slider/progress track
  sliderTrack: 'bg-neutral-700',
  progressTrack: 'bg-neutral-600',

  // Borders
  borderSubtle: 'border-neutral-700',
  borderInput: 'border-neutral-600',

  // Interactive button variants
  buttonSecondary: 'bg-neutral-700 hover:bg-neutral-600 text-gray-300',
  buttonGhost: 'text-gray-400 hover:text-white',
} as const;

// Light theme Tailwind classes
const lightTwClasses = {
  // Modality badges
  audioBadge: 'bg-blue-100 text-blue-700',
  presetBadge: 'bg-orange-100 text-orange-700',

  // Modality text colors
  audioText: 'text-blue-600',
  presetText: 'text-orange-600',

  // Modality legend dots
  audioDot: 'bg-blue-500',
  presetDot: 'bg-orange-500',

  // Rank indicators
  rankGood: 'text-green-600',
  rankOk: 'text-yellow-600',
  rankPoor: 'text-red-600',

  // Ground truth highlight in table
  groundTruthRow: 'bg-green-100/50',
  groundTruthMarker: 'text-green-600',

  // Form elements
  checkbox: 'rounded bg-white border-gray-300 accent-blue-500 focus:ring-blue-500 focus:ring-offset-0',
  inputFocus: 'focus:border-blue-500',
  sliderAccent: 'accent-blue-500',

  // Buttons
  buttonPrimary: 'bg-blue-600 hover:bg-blue-500 text-white',
  buttonActive: 'bg-blue-500 text-white',

  // Loading/progress
  spinner: 'border-blue-500',
  progressBar: 'bg-blue-500',

  // Recall bar chart color
  recallBar: 'bg-blue-500',

  // Ground truth range slider
  rangeSliderTrack: 'bg-purple-300/50',
  histogramBar: 'bg-gray-300',
  rangeSliderThumb: 'absolute top-0 w-full h-6 appearance-none bg-transparent cursor-pointer pointer-events-none [&::-webkit-slider-thumb]:pointer-events-auto [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-purple-500 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:pointer-events-auto [&::-moz-range-thumb]:appearance-none [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:bg-purple-500 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer',

  // Carrier count legend dots
  carriers1Dot: 'bg-purple-500',
  carriers2Dot: 'bg-blue-500',
  carriers3Dot: 'bg-green-500',
  carriers4Dot: 'bg-yellow-500',
  carriers5Dot: 'bg-orange-500',
  carriers6Dot: 'bg-red-500',

  // Background colors
  bgPrimary: 'bg-gray-100',
  bgSecondary: 'bg-white',
  bgTertiary: 'bg-gray-50',
  borderColor: 'border-gray-200',

  // Text colors
  textPrimary: 'text-gray-900',
  textSecondary: 'text-gray-700',
  textMuted: 'text-gray-600',
  textSubtle: 'text-gray-500',

  // Warning popup
  warningBg: 'bg-yellow-50',
  warningBorder: 'border-yellow-400',
  warningIcon: 'text-yellow-600',
  warningTitle: 'text-yellow-700',
  warningText: 'text-gray-700',
  warningButton: 'bg-yellow-500 hover:bg-yellow-400 text-white',

  // Error state
  errorIcon: 'text-red-500',

  // Code diff colors
  diffAddedBg: 'bg-green-100/50',
  diffRemovedBg: 'bg-red-100/50',
  diffAddedText: 'text-green-700',
  diffRemovedText: 'text-red-700',

  // Info modal highlight section
  infoHighlightBg: 'bg-blue-50',
  infoHighlightBorder: 'border-blue-300',
  infoHighlightTitle: 'text-blue-700',

  // Elevated UI elements
  bgElevated: 'bg-white',
  bgElevatedSemi: 'bg-white/95',
  bgElevatedHover: 'hover:bg-gray-100',
  bgInput: 'bg-white',
  bgInputHover: 'hover:bg-gray-50',

  // Slider/progress track
  sliderTrack: 'bg-gray-200',
  progressTrack: 'bg-gray-300',

  // Borders
  borderSubtle: 'border-gray-200',
  borderInput: 'border-gray-300',

  // Interactive button variants
  buttonSecondary: 'bg-gray-100 hover:bg-gray-200 text-gray-700',
  buttonGhost: 'text-gray-500 hover:text-gray-900',
} as const;

// Export both theme class sets
export const twClassesByTheme = {
  dark: darkTwClasses,
  light: lightTwClasses,
} as const;

// Default export (dark theme for backwards compatibility)
export const twClasses = darkTwClasses;

// Get Tailwind classes by mode
export function getTwClasses(mode: ThemeMode) {
  return twClassesByTheme[mode];
}
