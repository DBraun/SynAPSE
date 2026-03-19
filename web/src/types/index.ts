export type Modality = 'audio' | 'preset';

export type RetrievalMode = 'cross' | 'intra';

export type ColorBy = 'modality' | 'algorithm' | 'carrierCount' | 'modulatorCount';

export type ModalityFilter = 'all' | 'audio' | 'preset';

export interface PointSizeSettings {
  baseRadius: number;
  minPixels: number;
  maxPixels: number;
}

export interface DataPoint {
  id: string;
  modality: Modality;
  dx7_preset: number[]; // 145-dim float32 array (dexed-py format)
  preset_name: string;
  embedding: number[];
  x: number;
  y: number;
  pair_id: string;
  tags: string[];
}

export interface NeighborResult {
  point: DataPoint;
  distance: number;
  rank: number;
}

export interface DirectionMetrics {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  sample_size: number;
}

/**
 * Cross-modal retrieval metrics for the full (unfiltered) dataset, loaded from
 * `public/data/metrics.json`. Lets the metrics panel show results instantly on
 * load instead of waiting for the distance matrix and the metrics worker.
 */
export interface PrecomputedMetrics {
  audio_to_preset: DirectionMetrics;
  preset_to_audio: DirectionMetrics;
}

export interface AppState {
  data: DataPoint[];
  filteredData: DataPoint[];
  selectedPointId: string | null;
  k: number;
  retrievalMode: RetrievalMode;
  textFilter: string;
  activeTags: string[];
  colorBy: ColorBy;
  modalityFilter: ModalityFilter;
  availableTags: string[];
  isLoading: boolean;
  error: string | null;
  pointSize: PointSizeSettings;
  hideFiltered: boolean;
  autoPlay: boolean;
}

export type AppAction =
  | { type: 'SET_DATA'; payload: DataPoint[] }
  | { type: 'SET_SELECTED_POINT'; payload: string | null }
  | { type: 'SET_K'; payload: number }
  | { type: 'SET_RETRIEVAL_MODE'; payload: RetrievalMode }
  | { type: 'SET_TEXT_FILTER'; payload: string }
  | { type: 'SET_ACTIVE_TAGS'; payload: string[] }
  | { type: 'TOGGLE_TAG'; payload: string }
  | { type: 'SET_COLOR_BY'; payload: ColorBy }
  | { type: 'SET_MODALITY_FILTER'; payload: ModalityFilter }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_POINT_SIZE'; payload: Partial<PointSizeSettings> }
  | { type: 'SET_HIDE_FILTERED'; payload: boolean }
  | { type: 'SET_AUTO_PLAY'; payload: boolean }
  | { type: 'ADD_UPLOADED_POINT'; payload: DataPoint };
