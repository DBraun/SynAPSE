import type { DataPoint, NeighborResult, RetrievalMode } from '../types';

export type DistanceFn = (id1: string, id2: string) => number;

/**
 * Calculate Euclidean distance between two embedding vectors.
 */
export function euclideanDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Embedding dimensions must match');
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

/**
 * Calculate cosine distance between two embedding vectors.
 * Cosine distance = 1 - cosine_similarity
 */
export function cosineDistance(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Embedding dimensions must match');
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const cosineSim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
  return 1 - cosineSim;
}

/**
 * Find K nearest neighbors for a given point.
 *
 * If a getDistance function is provided (from pre-computed matrix),
 * it will be used for fast lookup. Otherwise, falls back to computing
 * cosine distance on-the-fly.
 */
export function findKNearestNeighbors(
  queryPoint: DataPoint,
  candidates: DataPoint[],
  k: number,
  retrievalMode: RetrievalMode,
  getDistance?: DistanceFn
): NeighborResult[] {
  // Filter candidates based on retrieval mode
  let filteredCandidates = candidates.filter(p => p.id !== queryPoint.id);

  switch (retrievalMode) {
    case 'cross':
      // Only retrieve from opposite modality
      filteredCandidates = filteredCandidates.filter(
        p => p.modality !== queryPoint.modality
      );
      break;
    case 'intra':
      // Only retrieve from same modality
      filteredCandidates = filteredCandidates.filter(
        p => p.modality === queryPoint.modality
      );
      break;
  }

  // Calculate distances using provided function or fallback to on-the-fly computation
  const distances: { point: DataPoint; distance: number }[] = filteredCandidates.map(
    candidate => ({
      point: candidate,
      distance: getDistance
        ? getDistance(queryPoint.id, candidate.id)
        : cosineDistance(queryPoint.embedding, candidate.embedding),
    })
  );

  // Sort by distance and take top K
  distances.sort((a, b) => a.distance - b.distance);

  return distances.slice(0, k).map((item, index) => ({
    point: item.point,
    distance: item.distance,
    rank: index + 1,
  }));
}

/**
 * Get the paired point for a given point
 */
export function getPairedPoint(
  point: DataPoint,
  data: DataPoint[]
): DataPoint | null {
  return (
    data.find(
      p => p.pair_id === point.pair_id && p.modality !== point.modality
    ) || null
  );
}
