export type Metric = 'cosine' | 'dot' | 'euclidean';
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export type Metadata = { [key: string]: JsonValue };

export interface IndexConfig {
  type: 'hnsw' | 'flat';
  m: number;
  efConstruction: number;
  efSearch: number;
  annThreshold: number;
}

export interface TransformersConfig {
  provider: 'transformers';
  model: string;
  revision: string;
  dtype: string;
  device: 'auto' | 'wasm' | 'webgpu';
  pooling: 'mean' | 'cls' | 'last_token';
  normalize: boolean;
  queryPrefix: string;
  documentPrefix: string;
  maxTokens: number;
  batchSize: number;
  batchTokens: number;
  localModelPath?: string;
  allowRemoteModels: boolean;
  wasmPath?: string;
}

export interface EmbeddingSchema {
  fingerprint: string;
  config: TransformersConfig;
}

export interface CollectionSchema {
  dimension: number | null;
  metric: Metric;
  embedding: EmbeddingSchema | null;
  index: IndexConfig;
}

export interface CollectionState {
  name: string;
  generation: string;
  schema: CollectionSchema;
  revision: number;
  count: number;
  createdAt: number;
  updatedAt: number;
}

export interface CollectionOptions {
  dimension?: number;
  metric?: Metric;
  embedder?: TransformersConfig;
  index?: 'hnsw' | 'flat' | Partial<IndexConfig>;
}

export interface DocumentInput {
  id: string;
  vector?: Float32Array | Float64Array | number[];
  text?: string;
  metadata?: Metadata;
  space?: string;
}

export interface StoredRecord {
  collection: string;
  id: string;
  vector: Float32Array;
  text?: string;
  metadata: Metadata;
  timestamp: number;
}

export interface PreparedRecord {
  id: string;
  vector: Float32Array;
  text?: string;
  metadata: Metadata;
  timestamp?: number;
}

export interface Mutation {
  upserts?: PreparedRecord[];
  deleteIds?: string[];
  clear?: boolean;
}

export interface MutationResult {
  state: CollectionState;
  upserted: StoredRecord[];
  deleted: number;
}

export type Filter = { [key: string]: unknown };

export interface SearchOptions {
  vector?: Float32Array | Float64Array | number[];
  text?: string;
  topK?: number;
  mode?: 'auto' | 'exact' | 'ann' | 'hybrid';
  efSearch?: number;
  filter?: Filter;
  minScore?: number;
  maxDistance?: number;
  includeVectors?: boolean;
  mmr?: number;
  signal?: AbortSignal;
  space?: string;
}

export interface SearchResult {
  id: string;
  score: number;
  distance?: number;
  text?: string;
  metadata: Metadata;
  vector?: Float32Array;
  timestamp: number;
}

export interface SearchResponse {
  results: SearchResult[];
  revision: number;
  generation: string;
  mode: 'exact' | 'ann' | 'hybrid';
  elapsedMs: number;
}

export interface OperationOptions {
  signal?: AbortSignal;
  onProgress?: (progress: ProgressEvent) => void;
}

export interface ProgressEvent {
  phase: 'loading-model' | 'embedding' | 'indexing' | 'importing';
  completed: number;
  total: number;
  detail?: unknown;
}

export interface OpenOptions {
  worker?: boolean;
  workerUrl?: string | URL;
  durability?: 'strict' | 'relaxed';
  blockedTimeoutMs?: number;
  wasm?: boolean;
}

export interface IndexMatch {
  id: string;
  score: number;
  distance?: number;
}

export interface VectorSearchOptions {
  topK: number;
  efSearch?: number;
  exact?: boolean;
  filter?: (id: string) => boolean;
  minScore?: number;
  maxDistance?: number;
  signal?: AbortSignal;
  scores?: Float32Array;
}

export interface IndexSnapshot {
  format: number;
  dimension: number;
  metric: Metric;
  options: IndexConfig;
  ids: string[];
  vectors: Float32Array;
  active: Uint8Array;
  levels: number[];
  neighbors: number[][][];
  entryPoint: number;
  maxLevel: number;
  randomState: number;
}

export interface PersistedIndex {
  generation: string;
  revision: number;
  payload: IndexSnapshot;
}

export interface DatabaseExport {
  format: 'serenedb';
  version: 2;
  collections: Array<{
    state: CollectionState;
    records: Array<Omit<StoredRecord, 'vector'> & { vector: number[] }>;
  }>;
}

export interface SerializedError {
  name: string;
  message: string;
  code?: string;
}
