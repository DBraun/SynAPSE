/**
 * Web Worker for computing recall metrics off the main thread.
 * Computes both directions (audio→preset and preset→audio) in one pass.
 *
 * Cross-modal cosine distances are computed on the fly from the embeddings.
 * Rank is computed via a stable sort of the candidate distances: for each
 * query, the 1-indexed position of the true pair (ties broken by candidate
 * order) is its rank. This matches the on-the-fly distance used elsewhere in
 * the app and the numpy argsort convention in the paper's analysis scripts.
 */

interface QueryPoint {
  id: string;
  modality: 'audio' | 'preset';
  pair_id: string;
  embedding: number[];
}

interface AllDataPoint {
  id: string;
  modality: 'audio' | 'preset';
  pair_id: string;
}

interface ComputeMessage {
  type: 'compute';
  audioPoints: QueryPoint[];
  codePoints: QueryPoint[];
  allData: AllDataPoint[];
}

interface DirectionMetrics {
  recall_at_1: number;
  recall_at_5: number;
  recall_at_10: number;
  recall_at_20: number;
  sample_size: number;
}

interface ResultMessage {
  type: 'result';
  audio_to_code: DirectionMetrics;
  code_to_audio: DirectionMetrics;
}

function cosineDistance(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return 1 - dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function getPairedPoint(point: AllDataPoint, allData: AllDataPoint[]): AllDataPoint | undefined {
  return allData.find(p => p.pair_id === point.pair_id && p.modality !== point.modality);
}

/**
 * Find the 1-indexed rank of the true pair among candidates by sorting
 * cross-modal cosine distances. Ties are broken by candidate order (stable
 * sort), matching numpy's argsort convention.
 */
function findRank(queryPoint: QueryPoint, truePairId: string, candidates: QueryPoint[]): number {
  const candidateDists: { id: string; dist: number }[] = [];
  for (const candidate of candidates) {
    candidateDists.push({
      id: candidate.id,
      dist: cosineDistance(queryPoint.embedding, candidate.embedding),
    });
  }

  // Stable sort by ascending distance (ties broken by original order)
  candidateDists.sort((a, b) => a.dist - b.dist);

  for (let i = 0; i < candidateDists.length; i++) {
    if (candidateDists[i].id === truePairId) {
      return i + 1; // 1-indexed
    }
  }
  return candidateDists.length + 1;
}

function computeDirectionMetrics(
  queryPoints: QueryPoint[],
  candidates: QueryPoint[],
  allData: AllDataPoint[]
): DirectionMetrics {
  let hits_at_1 = 0;
  let hits_at_5 = 0;
  let hits_at_10 = 0;
  let hits_at_20 = 0;
  let validQueries = 0;

  const candidateIds = new Set(candidates.map(c => c.id));

  for (const queryPoint of queryPoints) {
    const truePair = getPairedPoint(queryPoint, allData);
    if (!truePair) continue;

    if (!candidateIds.has(truePair.id)) continue;

    validQueries++;

    const rank = findRank(queryPoint, truePair.id, candidates);

    if (rank <= 1) hits_at_1++;
    if (rank <= 5) hits_at_5++;
    if (rank <= 10) hits_at_10++;
    if (rank <= 20) hits_at_20++;
  }

  return {
    recall_at_1: validQueries > 0 ? hits_at_1 / validQueries : 0,
    recall_at_5: validQueries > 0 ? hits_at_5 / validQueries : 0,
    recall_at_10: validQueries > 0 ? hits_at_10 / validQueries : 0,
    recall_at_20: validQueries > 0 ? hits_at_20 / validQueries : 0,
    sample_size: validQueries,
  };
}

self.onmessage = (event: MessageEvent<ComputeMessage>) => {
  const { type, audioPoints, codePoints, allData } = event.data;

  if (type !== 'compute') return;

  // Compute both directions
  const audioToCode = computeDirectionMetrics(audioPoints, codePoints, allData);
  const codeToAudio = computeDirectionMetrics(codePoints, audioPoints, allData);

  const response: ResultMessage = {
    type: 'result',
    audio_to_code: audioToCode,
    code_to_audio: codeToAudio,
  };

  self.postMessage(response);
};
