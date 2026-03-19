import { createContext, useContext, useReducer, useEffect, useCallback, useState, useMemo } from 'react';
import type { ReactNode } from 'react';
import type { AppState, AppAction, DataPoint, ColorBy, RetrievalMode, PointSizeSettings, ModalityFilter, PrecomputedMetrics } from '../types';
import { applyFilters } from '../utils/filtering';
import { cosineDistance, type DistanceFn } from '../utils/distance';

const initialState: AppState = {
  data: [],
  filteredData: [],
  selectedPointId: null,
  k: 10,
  retrievalMode: 'cross',
  textFilter: '',
  activeTags: [],
  colorBy: 'modality',
  modalityFilter: 'all',
  availableTags: [],
  isLoading: true,
  error: null,
  pointSize: {
    baseRadius: 0.3,
    minPixels: 1,
    maxPixels: 10,
  },
  hideFiltered: false,
  autoPlay: false,
};

function extractAvailableTags(data: DataPoint[]): string[] {
  const tagSet = new Set<string>();
  data.forEach(point => point.tags.forEach(tag => tagSet.add(tag)));
  return Array.from(tagSet).sort((a, b) => {
    // Natural sort: extract trailing number and compare numerically
    const re = /^(.+?)(\d+)$/;
    const ma = a.match(re);
    const mb = b.match(re);
    if (ma && mb && ma[1] === mb[1]) {
      return parseInt(ma[2]) - parseInt(mb[2]);
    }
    return a.localeCompare(b);
  });
}

function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_DATA': {
      const availableTags = extractAvailableTags(action.payload);
      return {
        ...state,
        data: action.payload,
        filteredData: action.payload,
        availableTags,
        isLoading: false,
      };
    }
    case 'SET_SELECTED_POINT':
      return { ...state, selectedPointId: action.payload };
    case 'SET_K':
      return { ...state, k: action.payload };
    case 'SET_RETRIEVAL_MODE':
      return { ...state, retrievalMode: action.payload };
    case 'SET_TEXT_FILTER': {
      const newState = { ...state, textFilter: action.payload };
      newState.filteredData = applyFilters(
        state.data,
        action.payload,
        state.activeTags,
        state.modalityFilter
      );
      return newState;
    }
    case 'SET_ACTIVE_TAGS': {
      const newState = { ...state, activeTags: action.payload };
      newState.filteredData = applyFilters(
        state.data,
        state.textFilter,
        action.payload,
        state.modalityFilter
      );
      return newState;
    }
    case 'TOGGLE_TAG': {
      const newActiveTags = state.activeTags.includes(action.payload)
        ? state.activeTags.filter(t => t !== action.payload)
        : [...state.activeTags, action.payload];
      const newState = { ...state, activeTags: newActiveTags };
      newState.filteredData = applyFilters(
        state.data,
        state.textFilter,
        newActiveTags,
        state.modalityFilter
      );
      return newState;
    }
    case 'SET_COLOR_BY':
      return { ...state, colorBy: action.payload };
    case 'SET_MODALITY_FILTER': {
      const newState = { ...state, modalityFilter: action.payload };
      newState.filteredData = applyFilters(
        state.data,
        state.textFilter,
        state.activeTags,
        action.payload
      );
      return newState;
    }
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload, isLoading: false };
    case 'SET_POINT_SIZE':
      return { ...state, pointSize: { ...state.pointSize, ...action.payload } };
    case 'SET_HIDE_FILTERED':
      return { ...state, hideFiltered: action.payload };
    case 'SET_AUTO_PLAY':
      return { ...state, autoPlay: action.payload };
    case 'ADD_UPLOADED_POINT': {
      // Remove any previous upload point, then add the new one
      const withoutOldUpload = state.data.filter(p => !p.id.startsWith('upload_'));
      const newData = [...withoutOldUpload, action.payload];
      const availableTags = extractAvailableTags(newData);
      return {
        ...state,
        data: newData,
        filteredData: applyFilters(newData, state.textFilter, state.activeTags, state.modalityFilter),
        availableTags,
        selectedPointId: action.payload.id,
      };
    }
    default:
      return state;
  }
}

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  setSelectedPoint: (id: string | null) => void;
  setK: (k: number) => void;
  setRetrievalMode: (mode: RetrievalMode) => void;
  setTextFilter: (filter: string) => void;
  toggleTag: (tag: string) => void;
  setColorBy: (colorBy: ColorBy) => void;
  setModalityFilter: (filter: ModalityFilter) => void;
  setPointSize: (settings: Partial<PointSizeSettings>) => void;
  setHideFiltered: (hide: boolean) => void;
  setAutoPlay: (autoPlay: boolean) => void;
  // Cosine distance between any two points, computed on the fly from embeddings.
  getDistance: DistanceFn;
  distancesReady: boolean;
  precomputedMetrics: PrecomputedMetrics | null;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Precomputed metrics for the full dataset (optional metrics.json). When
  // present, the metrics panel shows unfiltered results immediately.
  const [precomputedMetrics, setPrecomputedMetrics] = useState<PrecomputedMetrics | null>(null);

  useEffect(() => {
    const worker = new Worker(
      new URL('../workers/dataWorker.ts', import.meta.url),
      { type: 'module' }
    );

    worker.onmessage = (event) => {
      const { type } = event.data;
      if (type === 'data') {
        dispatch({ type: 'SET_DATA', payload: event.data.data });
      } else if (type === 'error') {
        dispatch({ type: 'SET_ERROR', payload: event.data.message });
      }
    };

    worker.onerror = (err) => {
      dispatch({ type: 'SET_ERROR', payload: err.message });
    };

    worker.postMessage({ type: 'load', url: `${import.meta.env.BASE_URL}data/embeddings.json` });

    // Optionally load precomputed metrics for the full dataset. Absence is
    // expected (r.ok === false) and simply disables the instant fast path;
    // a genuine fetch failure is surfaced via console.warn rather than swallowed.
    fetch(`${import.meta.env.BASE_URL}data/metrics.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data: PrecomputedMetrics | null) => {
        if (data) setPrecomputedMetrics(data);
      })
      .catch((err) => console.warn('Could not load metrics.json:', err));

    return () => {
      worker.terminate();
    };
  }, []);

  // Look up each point by id so distances can be computed on the fly.
  const idToPoint = useMemo(() => {
    const map = new Map<string, DataPoint>();
    for (const point of state.data) map.set(point.id, point);
    return map;
  }, [state.data]);

  // Distances are available as soon as the data has loaded. Cosine distance is
  // computed on demand from the embeddings (O(dim) per lookup), which at this
  // dataset size is far cheaper than precomputing full N×N matrices on load.
  const distancesReady = state.data.length > 0;

  const getDistance = useCallback(
    (id1: string, id2: string): number => {
      const p1 = idToPoint.get(id1);
      const p2 = idToPoint.get(id2);
      if (!p1 || !p2) return Infinity;
      return cosineDistance(p1.embedding, p2.embedding);
    },
    [idToPoint]
  );

  const value: AppContextValue = {
    state,
    dispatch,
    setSelectedPoint: (id) => dispatch({ type: 'SET_SELECTED_POINT', payload: id }),
    setK: (k) => dispatch({ type: 'SET_K', payload: k }),
    setRetrievalMode: (mode) => dispatch({ type: 'SET_RETRIEVAL_MODE', payload: mode }),
    setTextFilter: (filter) => dispatch({ type: 'SET_TEXT_FILTER', payload: filter }),
    toggleTag: (tag) => dispatch({ type: 'TOGGLE_TAG', payload: tag }),
    setColorBy: (colorBy) => dispatch({ type: 'SET_COLOR_BY', payload: colorBy }),
    setModalityFilter: (filter) => dispatch({ type: 'SET_MODALITY_FILTER', payload: filter }),
    setPointSize: (settings) => dispatch({ type: 'SET_POINT_SIZE', payload: settings }),
    setHideFiltered: (hide) => dispatch({ type: 'SET_HIDE_FILTERED', payload: hide }),
    setAutoPlay: (autoPlay) => dispatch({ type: 'SET_AUTO_PLAY', payload: autoPlay }),
    getDistance,
    distancesReady,
    precomputedMetrics,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
