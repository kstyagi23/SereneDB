import { invalid } from './errors.js';
import type { CollectionOptions, CollectionSchema, IndexConfig, Metadata, Metric, TransformersConfig } from './types.js';

export function validName(value: unknown, label = 'Name'): asserts value is string {
  if (typeof value !== 'string' || !value.trim() || value.length > 512) invalid(`${label} must be a nonempty string of at most 512 characters.`);
}

export function integer(value: unknown, label: string, min: number, max: number): asserts value is number {
  if (!Number.isInteger(value) || (value as number) < min || (value as number) > max) invalid(`${label} must be an integer between ${min} and ${max}.`);
}

export function prepareVector(value: unknown, metric: Metric, dimension?: number | null): Float32Array {
  if (!(Array.isArray(value) || value instanceof Float32Array || value instanceof Float64Array)) invalid('Vector must be a numeric array or float typed array. Pass a pooled tensor row, not a batched tensor.');
  integer(value.length, 'Vector dimension', 1, 16384);
  if (dimension != null && value.length !== dimension) invalid(`Vector dimension mismatch: expected ${dimension}, received ${value.length}.`, 'SCHEMA_MISMATCH');
  let scale = 0;
  for (let i = 0; i < value.length; i++) {
    if (typeof value[i] !== 'number' || !Number.isFinite(value[i])) invalid(`Vector element ${i} must be a finite number.`);
    scale = Math.max(scale, Math.abs(value[i]));
  }
  const result = new Float32Array(value.length);
  if (metric === 'cosine') {
    if (scale === 0) invalid('Cosine vectors must have a nonzero norm.');
    let sum = 0;
    for (let i = 0; i < value.length; i++) sum += (value[i] / scale) ** 2;
    const norm = Math.sqrt(sum);
    for (let i = 0; i < value.length; i++) result[i] = (value[i] / scale) / norm;
  } else {
    for (let i = 0; i < value.length; i++) {
      result[i] = value[i];
      if (!Number.isFinite(result[i])) invalid('Vector elements must fit in float32.');
    }
  }
  return result;
}

export function prepareMetadata(value: unknown = {}): Metadata {
  const seen = new Set<object>();
  const visit = (item: unknown, depth: number): unknown => {
    if (depth > 32) invalid('Metadata nesting exceeds 32 levels.');
    if (item === null || typeof item === 'string' || typeof item === 'boolean') return item;
    if (typeof item === 'number' && Number.isFinite(item)) return item;
    if (typeof item !== 'object' || !item) invalid('Metadata must contain only finite, JSON-compatible values.');
    if (seen.has(item)) invalid('Metadata must not contain cycles.');
    seen.add(item);
    let copy: unknown;
    if (Array.isArray(item)) copy = Array.from(item, entry => visit(entry, depth + 1));
    else {
      if (Object.getPrototypeOf(item) !== Object.prototype && Object.getPrototypeOf(item) !== null) invalid('Metadata objects must be plain objects.');
      copy = Object.fromEntries(Object.entries(item).map(([key, entry]) => [key, visit(entry, depth + 1)]));
    }
    seen.delete(item);
    return copy;
  };
  if (!value || typeof value !== 'object' || Array.isArray(value)) invalid('Metadata must be a plain object.');
  return visit(value, 0) as Metadata;
}

export function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`;
  return JSON.stringify(value);
}

export function embeddingFingerprint(config: TransformersConfig): string {
  const { device: _device, batchSize: _batch, batchTokens: _tokens, localModelPath: _path, allowRemoteModels: _remote, wasmPath: _wasm, ...space } = config;
  return canonical(space);
}

export function collectionSchema(options: CollectionOptions = {}): CollectionSchema {
  const metric = options.metric ?? 'cosine';
  if (!['cosine', 'dot', 'euclidean'].includes(metric)) invalid('Metric must be cosine, dot, or euclidean.');
  if (options.dimension !== undefined) integer(options.dimension, 'Dimension', 1, 16384);
  const input = typeof options.index === 'string' ? { type: options.index } : options.index ?? {};
  const index: IndexConfig = { type: 'hnsw', m: 16, efConstruction: 128, efSearch: 64, annThreshold: 256, ...input };
  if (index.type !== 'hnsw' && index.type !== 'flat') invalid('Index type must be hnsw or flat.');
  integer(index.m, 'HNSW m', 2, 64);
  integer(index.efConstruction, 'efConstruction', index.m, 4096);
  integer(index.efSearch, 'efSearch', 1, 65536);
  integer(index.annThreshold, 'annThreshold', 0, 10000000);
  return {
    dimension: options.dimension ?? null,
    metric,
    embedding: options.embedder ? { config: structuredClone(options.embedder), fingerprint: embeddingFingerprint(options.embedder) } : null,
    index,
  };
}
