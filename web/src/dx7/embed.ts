/**
 * Placeholder embedding and projection functions.
 *
 * These will eventually be replaced by a real TensorFlow.js model
 * that encodes DX7 presets into embedding vectors.
 */

import * as tf from '@tensorflow/tfjs';

// Ensure TF.js is ready (triggers WASM/WebGL backend init)
const _ready = tf.ready();

/** Embedding dimension — must match the trained model. */
export const EMBEDDING_DIM = 128;

/**
 * Compute an embedding for a DX7 preset (placeholder).
 *
 * Currently returns a deterministic pseudo-random vector seeded by the
 * preset parameters so the same preset always maps to the same point.
 * This will be replaced by a real model call.
 */
export async function embedPreset(preset: number[]): Promise<number[]> {
  await _ready;

  // Deterministic hash-based placeholder:
  // Use a simple linear projection of the 145-dim preset to EMBEDDING_DIM,
  // then L2-normalize. This ensures similar presets get somewhat similar
  // embeddings even without a trained model.
  const input = tf.tensor1d(preset);
  // Fixed seed: use a deterministic "random" projection matrix
  const seed = 42;
  const proj = tf.tidy(() => {
    // Create a reproducible projection by using preset values as seed offsets
    const rows = preset.length;
    const cols = EMBEDDING_DIM;
    const values = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        // Simple deterministic pseudo-random based on position
        const x = Math.sin((seed + 1) * (i * cols + j + 1) * 0.001) * 43758.5453;
        values[i * cols + j] = x - Math.floor(x) - 0.5;
      }
    }
    return tf.tensor2d(values, [rows, cols]);
  });

  const embedding = tf.tidy(() => {
    const raw = input.reshape([1, preset.length]).matMul(proj).squeeze();
    const norm = raw.norm();
    return raw.div(norm);
  });

  const result = Array.from(await embedding.data());

  input.dispose();
  proj.dispose();
  embedding.dispose();

  return result;
}

/**
 * Project an embedding to 2D coordinates (placeholder).
 *
 * In production this would use the same t-SNE/UMAP transform fitted on
 * the training data. For now we use a simple deterministic projection.
 */
export function projectTo2D(embedding: number[]): { x: number; y: number } {
  // Simple 2-component projection
  let x = 0, y = 0;
  for (let i = 0; i < embedding.length; i++) {
    x += embedding[i] * Math.sin(i * 0.7 + 0.3);
    y += embedding[i] * Math.cos(i * 0.5 + 1.1);
  }
  // Scale to roughly match the data range
  return { x: x * 15, y: y * 15 };
}
