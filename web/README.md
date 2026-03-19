# SynAPSE-map

Interactive visualization for exploring a joint embedding space of DX7 synthesizer
presets and their corresponding audio. Visualize 2D projections, query nearest
neighbors across modalities, audition presets with a real-time in-browser FM
synthesizer, and evaluate cross-modal retrieval performance.

Companion demo for the paper *FM Synthesizer Audio-Parameter Shared Embeddings*
(FM-SynAPSE). The bundled embeddings come from the paper's DX7-GNN model evaluated on
eight DX7 algorithms held out of training (4, 8, ..., 32; 4,096 audio/preset pairs).

**Live site:** https://dbraun.github.io/SynAPSE/

## Features

- **Unified scatter plot**: WebGL/deck.gl rendering of audio and preset embeddings with pan/zoom
- **Point selection & neighbor retrieval**: click any point to find its K nearest neighbors in embedding space
- **Cross-modal retrieval**: query an audio embedding to find similar presets (and vice versa)
- **Real-time FM synthesis**: audition any DX7 preset in the browser (TypeScript port of dx7-synth-js)
- **DX7 preset panel**: algorithm diagram, operators, and envelopes for the selected preset
- **Recall@K evaluation**: compute retrieval metrics on filtered subsets
- **Filtering & color coding**: filter by tags; color points by modality, algorithm, or carrier count
- **Fast distances**: WebGPU compute shaders when available, with a Web Worker CPU fallback

## Quick Start

### Prerequisites

- Node.js 18+ (CI builds with Node 22)
- Python 3.8+ (only for mock data generation)

### Install & run

```bash
npm install        # first time only
npm run dev        # dev server at http://localhost:5173
```

### Generate mock data

A sample `public/data/embeddings.json` is committed, but you can regenerate it:

```bash
python3 scripts/prepare_fake_data.py --output public/data/embeddings.json
```

### Build for production

```bash
npm run build      # outputs to dist/
npx vite preview   # preview the production build under the /SynAPSE/ base
```

## Data Format

The app loads `public/data/embeddings.json` — an array of DataPoint objects. Each
audio sample and its corresponding preset are separate points linked by `pair_id`.

```json
[
  {
    "id": "audio_00000",
    "modality": "audio",
    "dx7_preset": [0.0, 0.5, "..."],
    "preset_name": "BRASS   1",
    "embedding": [0.123, -0.456, "..."],
    "x": 12.34,
    "y": -5.67,
    "pair_id": "pair_00000",
    "tags": ["algorithm_5", "carriers_3"]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for this point (e.g. `"audio_00000"`, `"preset_00000"`) |
| `modality` | string | `"audio"` or `"preset"` |
| `dx7_preset` | number[] | 145-element float32 array in dexed-py `Preset.to_array()` format |
| `preset_name` | string | DX7 preset name (10 chars) |
| `embedding` | number[] | High-dimensional embedding vector (same dimensionality for all points) |
| `x`, `y` | number | 2D projection coordinates (UMAP, t-SNE, PCA, etc.) |
| `pair_id` | string | Shared ID linking an audio point to its preset point |
| `tags` | string[] | Auto-extracted tags (algorithm, carrier count, feedback level, etc.) |

The same shape in TypeScript (`src/types/index.ts`):

```typescript
interface DataPoint {
  id: string;              // "audio_123" or "preset_123"
  modality: 'audio' | 'preset';
  dx7_preset: number[];    // 145-dim float32 array (dexed-py format)
  preset_name: string;     // DX7 preset name (10 chars)
  embedding: number[];     // High-dimensional vector
  x: number;               // t-SNE/UMAP x coordinate
  y: number;               // t-SNE/UMAP y coordinate
  pair_id: string;         // Links audio to its preset
  tags: string[];          // Categorization tags
}
```

## Project Structure

This app is the `web/` package of the SynAPSE repo — a single Vite + React + TypeScript project,
with a small Python helper in `scripts/` for data preparation. The deployment workflow lives at the
repo root, everything else here:

```
SynAPSE/
├── .github/workflows/deploy.yml      # GitHub Pages deployment (repo root)
└── web/                              # this package
    ├── public/
    │   └── data/embeddings.json      # Embedding data (audio .wav files are gitignored)
    ├── scripts/
    │   ├── prepare_fake_data.py      # Mock data generation
    │   └── compute_metrics.py        # Precompute metrics.json from embeddings.json
    ├── src/
    │   ├── components/               # React components
    │   ├── context/                  # State management (React Context)
    │   ├── dx7/                      # DX7 FM synthesizer engine
    │   ├── hooks/                     # Custom hooks
    │   ├── utils/                    # Distance, filtering utilities
    │   ├── workers/                  # Web Workers (data loading, distances, metrics)
    │   └── types/                    # TypeScript types
    ├── index.html
    ├── package.json
    └── vite.config.ts
```

### Source layout

```
src/
├── main.tsx              # App entry point
├── theme.ts              # Centralized color theme configuration
├── types/index.ts        # TypeScript types (DataPoint, AppState, etc.)
├── dx7/                  # DX7 FM synthesizer engine
│   ├── types.ts          # DX7Params, DX7OperatorParams interfaces
│   ├── constants.ts      # Algorithm definitions, lookup tables
│   ├── EnvelopeDX7.ts    # 4-stage envelope generator
│   ├── LfoDX7.ts         # LFO with 6 waveforms
│   ├── Operator.ts       # Single FM operator
│   ├── FMVoice.ts        # 6-operator FM voice
│   ├── Synth.ts          # Polyphonic voice manager
│   ├── presetMapping.ts  # dexed-py float32[145] → DX7Params conversion
│   └── index.ts          # Public API
├── context/
│   ├── AppContext.tsx    # Global state management (React Context + useReducer)
│   └── ThemeContext.tsx  # Light/dark mode toggle with localStorage persistence
├── components/
│   ├── App.tsx           # Main layout: header, sidebar, scatterplot, results
│   ├── ScatterPlot.tsx   # deck.gl ScatterplotLayer visualization
│   ├── ControlPanel.tsx  # Left sidebar: filters, tag selection, color-by
│   ├── ResultsTable.tsx  # Right sidebar: selected point info, neighbors list
│   ├── MetricsPanel.tsx  # Retrieval metrics (recall@k)
│   ├── DX7PresetPanel.tsx # DX7 preset GUI overlay (algorithm diagram, operators, envelopes)
│   └── InfoModal.tsx     # Explanatory modal for users
├── hooks/
│   ├── useDX7Synth.ts   # React hook for DX7 Web Audio synthesis
│   ├── useDistanceMatrix.ts   # Manages distance computation (WebGPU or CPU)
│   └── useNearestNeighbors.ts # Computes k-NN from precomputed distances
├── utils/
│   ├── distance.ts       # CPU cosine distance computation
│   ├── gpuDistance.ts    # WebGPU-accelerated distance computation
│   └── filtering.ts      # Boolean expression parser for preset name filter
└── workers/
    ├── dataLoaderWorker.ts  # Streams and parses large JSON data
    ├── dataWorker.ts        # Data loading coordination
    ├── distanceWorker.ts    # Background distance matrix computation
    └── metricsWorker.ts     # Background retrieval metrics calculation
```

## Architecture

### DX7 synthesizer engine

`src/dx7/` is a TypeScript port of [dx7-synth-js](https://github.com/mmontag/dx7-synth-js).
Key changes from the original:

- Converted from CommonJS prototype-based JS to TypeScript classes
- Eliminated global mutable state (`var params = {}`) by encapsulating it in a `DX7Synth` instance
- LFO shared state moved into an `LfoGlobalState` object owned by `DX7Synth`
- Audio rendering via `ScriptProcessorNode`, initialized lazily on first user gesture

Used through the `useDX7Synth` hook:

```typescript
const { loadPreset, playNote, stopAll } = useDX7Synth();
loadPreset(point.dx7_preset, point.preset_name);
playNote(60, 0.8, 1500); // note, velocity, duration_ms
```

### State management

`AppContext.tsx` provides global state via React Context:

- `data` / `filteredData`: all points and the filtered subset
- `selectedPointId`: currently selected point
- `k`: number of neighbors to retrieve
- `retrievalMode`: `'cross'` (audio↔preset) or `'intra'` (same modality)
- `textFilter`: boolean expression filter for preset names
- `colorBy`: how to color points (`'modality'`, `'algorithm'`, or `'carrierCount'`)

### Distance computation

1. **WebGPU (preferred)**: compute shaders for fast parallel cosine distance
2. **CPU fallback**: Web Workers for background computation

Distances are precomputed and cached, so neighbor lookups are instant.

### Theming

All colors live in `src/theme.ts`. Two modality color sets — audio blue (`#3B82F6`) and preset
orange (`#F97316`) — plus DX7-specific colors for algorithm diagrams and carrier/modulator operator
fills. Light/dark mode is handled by `ThemeContext.tsx`, persisted to `localStorage`.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds the app and
publishes `dist/` to GitHub Pages, served at https://dbraun.github.io/SynAPSE/.
One-time setup: repo **Settings → Pages → Source → "GitHub Actions"**.

`vite.config.ts` sets `base: '/SynAPSE/'` for production builds so assets resolve under the
project's Pages subpath. Runtime fetches use `import.meta.env.BASE_URL` (e.g.
`${import.meta.env.BASE_URL}data/embeddings.json`) so data paths respect that base too.

`public/data/embeddings.json` and `metrics.json` are committed so the workflow can deploy them;
anything else under `public/data/` is gitignored. There are no per-pair audio files — preset audio
is synthesized in the browser by the DX7 engine in `src/dx7/`.

`npx vite preview` (see [Build for production](#build-for-production)) serves the build under that
same `/SynAPSE/` base, so a local preview matches production. Opening `dist/index.html` directly
over `file://` will not work: browsers block ES module imports from `file://` URLs.

## Tech Stack

- **Frontend**: React 19 + TypeScript
- **Build**: Vite
- **Visualization**: deck.gl (ScatterplotLayer)
- **Styling**: Tailwind CSS
- **Synthesis**: in-browser DX7 FM engine (Web Audio)
- **Compute**: WebGPU compute shaders with a Web Worker CPU fallback
- **Data pipeline**: Python

## License

MIT
