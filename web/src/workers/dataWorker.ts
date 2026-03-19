/**
 * Web Worker for loading and parsing the embeddings JSON off the main thread.
 *
 * Parsing the multi-megabyte dataset here (rather than on the main thread)
 * keeps the initial page responsive. Distances are no longer precomputed:
 * they are computed on the fly from the embeddings where needed.
 */

interface DataPoint {
  id: string;
  modality: 'audio' | 'preset';
  dx7_preset: number[];
  preset_name: string;
  embedding: number[];
  x: number;
  y: number;
  pair_id: string;
  tags: string[];
}

interface LoadMessage {
  type: 'load';
  url: string;
}

interface DataResultMessage {
  type: 'data';
  data: DataPoint[];
}

interface ErrorMessage {
  type: 'error';
  message: string;
}

self.onmessage = async (event: MessageEvent<LoadMessage>) => {
  const { type } = event.data;

  if (type === 'load') {
    const { url } = event.data;

    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`Failed to load data: ${response.statusText}`);
      }
      const data: DataPoint[] = await response.json();

      const dataResult: DataResultMessage = { type: 'data', data };
      self.postMessage(dataResult);
    } catch (err) {
      const error: ErrorMessage = {
        type: 'error',
        message: err instanceof Error ? err.message : 'Failed to load data',
      };
      self.postMessage(error);
    }
  }
};
