import { useMemo } from 'react';
import type { DataPoint, NeighborResult, RetrievalMode } from '../types';
import { findKNearestNeighbors, type DistanceFn } from '../utils/distance';

export function useNearestNeighbors(
  selectedPointId: string | null,
  data: DataPoint[],
  filteredData: DataPoint[],
  k: number,
  retrievalMode: RetrievalMode,
  getDistance?: DistanceFn
): NeighborResult[] {
  return useMemo(() => {
    if (!selectedPointId) return [];

    const selectedPoint = data.find(p => p.id === selectedPointId);
    if (!selectedPoint) return [];

    // Use filtered data as candidates
    return findKNearestNeighbors(selectedPoint, filteredData, k, retrievalMode, getDistance);
  }, [selectedPointId, data, filteredData, k, retrievalMode, getDistance]);
}
