import { checkAbort, invalid, SereneDBError } from './errors.js';
import type { OperationOptions, ProgressEvent, TransformersConfig } from './types.js';
import { integer, prepareVector, validName } from './validation.js';

const MAX_INPUT_CHARACTERS = 131072;
const DTYPES = ['fp32', 'fp16', 'q8', 'int8', 'uint8', 'q4', 'q4f16', 'bnb4', 'q2', 'q2f16', 'q1', 'q1f16'];

/**
 * Describe an embedding space without importing the optional runtime or downloading a model.
 * Pin revision to a full Hub commit SHA for reproducible collections: the default `main`
 * is mutable, as are branches/tags and local model directories. Changing model artifacts,
 * dtype, pooling, prefixes, or truncation requires an explicit collection re-embedding migration.
 * dtype is explicit because Transformers.js device defaults (including q8) are not universal.
 */
export function transformersEmbedder(options: Partial<TransformersConfig> & Pick<TransformersConfig, 'model'>): TransformersConfig {
  if (!options || typeof options !== 'object' || Array.isArray(options) || ![Object.prototype, null].includes(Object.getPrototypeOf(options))) invalid('Transformers options must be a plain object with a model.');
  const allowed = ['provider', 'model', 'revision', 'dtype', 'device', 'pooling', 'normalize', 'queryPrefix', 'documentPrefix', 'maxTokens', 'batchSize', 'batchTokens', 'localModelPath', 'allowRemoteModels', 'wasmPath'];
  for (const key of Object.keys(options)) if (!allowed.includes(key)) invalid(`Unknown Transformers option: ${key}.`);
  const config: TransformersConfig = {
    provider: 'transformers',
    model: options.model,
    revision: options.revision === undefined ? 'main' : options.revision,
    dtype: options.dtype === undefined ? 'fp32' : options.dtype,
    device: options.device === undefined ? 'wasm' : options.device,
    pooling: options.pooling === undefined ? 'mean' : options.pooling,
    normalize: options.normalize === undefined ? true : options.normalize,
    queryPrefix: options.queryPrefix === undefined ? '' : options.queryPrefix,
    documentPrefix: options.documentPrefix === undefined ? '' : options.documentPrefix,
    maxTokens: options.maxTokens === undefined ? 512 : options.maxTokens,
    batchSize: options.batchSize === undefined ? 16 : options.batchSize,
    batchTokens: options.batchTokens === undefined ? Math.max(8192, options.maxTokens ?? 512) : options.batchTokens,
    allowRemoteModels: options.allowRemoteModels === undefined ? true : options.allowRemoteModels,
  };
  if (options.provider !== undefined && options.provider !== 'transformers') invalid('Embedding provider must be transformers.');
  validName(config.model, 'Transformers model');
  validName(config.revision, 'Model revision');
  if (config.model !== config.model.trim() || config.revision !== config.revision.trim()) invalid('Model and revision must not have leading or trailing whitespace.');
  if (!DTYPES.includes(config.dtype)) invalid(`dtype must be an explicit supported artifact type (${DTYPES.join(', ')}); auto can change the embedding space across devices.`);
  if (!['wasm', 'webgpu', 'auto'].includes(config.device)) invalid('Transformers device must be wasm, webgpu, or auto.');
  if (!['mean', 'cls', 'last_token'].includes(config.pooling)) invalid('Pooling must be mean, cls, or last_token and must match the model card.');
  if (typeof config.normalize !== 'boolean' || typeof config.allowRemoteModels !== 'boolean') invalid('normalize and allowRemoteModels must be booleans.');
  for (const key of ['queryPrefix', 'documentPrefix'] as const) {
    if (typeof config[key] !== 'string' || config[key].length > 8192) invalid(`${key} must be a string of at most 8192 characters.`);
  }
  integer(config.maxTokens, 'maxTokens', 1, 32768);
  integer(config.batchSize, 'batchSize', 1, 256);
  integer(config.batchTokens, 'batchTokens', config.maxTokens, 1048576);
  for (const key of ['localModelPath', 'wasmPath'] as const) {
    if (options[key] === undefined) continue;
    const path = options[key];
    if (typeof path !== 'string' || !path.trim() || path !== path.trim() || path.length > 2048 || /[?#\u0000-\u001f]/.test(path)) invalid(`${key} must be a nonempty directory path or URL without a query or fragment.`);
    config[key] = path;
  }
  if (config.wasmPath) wasmPaths(config.wasmPath);
  return config;
}

type Tokenizer = ((texts: string[], options: Record<string, unknown>) => { input_ids?: unknown }) & {
  model_max_length?: number;
  padding_side?: string;
};
type Model = { dispose(): void | Promise<unknown> };
type EmbeddingPipeline = ((texts: string[], options: Record<string, unknown>) => Promise<unknown>) & {
  tokenizer: Tokenizer;
  dispose(): void | Promise<unknown>;
};
type WasmSettings = { wasmPaths?: string | { mjs: string; wasm: string }; numThreads?: number; wasmBinary?: unknown };
type Runtime = {
  env: Record<string, unknown> & { version?: string; backends?: { onnx?: { wasm?: WasmSettings } } };
  AutoTokenizer: { from_pretrained(model: string, options: Record<string, unknown>): Promise<Tokenizer> };
  AutoModel: { from_pretrained(model: string, options: Record<string, unknown>): Promise<Model> };
  FeatureExtractionPipeline: new (options: { task: string; tokenizer: Tokenizer; model: Model }) => EmbeddingPipeline;
};

export interface TransformersRuntimeInfo {
  provider: 'transformers';
  model: string;
  revision: string;
  revisionPinned: boolean;
  requestedDevice: TransformersConfig['device'];
  device: 'wasm' | 'webgpu' | null;
  dtype: string;
  version: string | null;
  loaded: boolean;
  warmed: boolean;
  disposed: boolean;
  dimension: number | null;
  effectiveMaxTokens: number | null;
  fallbackReason: string | null;
  warnings: string[];
}

async function defaultRuntime(): Promise<Runtime> {
  try {
    return await import('@huggingface/transformers') as unknown as Runtime;
  } catch (cause) {
    throw new SereneDBError('Unable to import optional @huggingface/transformers. Install a compatible Transformers.js 4.x runtime or configure its browser module resolver and deploy the matching WASM assets.', 'EMBEDDING_DEPENDENCY', { cause });
  }
}

function wasmPaths(path?: string): { mjs: string; wasm: string } {
  let base: URL;
  try {
    base = new URL(path ? `${path.replace(/\/$/, '')}/` : './vendor/', import.meta.url);
  } catch {
    return invalid('wasmPath must be a valid directory URL.');
  }
  if (!['http:', 'https:', 'file:'].includes(base.protocol)) invalid('wasmPath must use an HTTP(S) or local directory URL.');
  return {
    mjs: new URL('ort-wasm-simd-threaded.jsep.mjs', base).href,
    wasm: new URL('ort-wasm-simd-threaded.jsep.wasm', base).href,
  };
}

// ONNX backend initialization is realm-global and cannot be reset by disposing a model.
const wasmReservations = new WeakMap<WasmSettings, { mjs: string; wasm: string }>();
let loaderQueue = Promise.resolve();

function configureWasm(runtime: Runtime, config: TransformersConfig): void {
  const wasm = runtime.env.backends?.onnx?.wasm;
  if (!wasm) return;
  const paths = wasmPaths(config.wasmPath);
  const reserved = wasmReservations.get(wasm);
  if (reserved) {
    if (reserved.wasm !== paths.wasm || reserved.mjs !== paths.mjs) invalid('Conflicting wasmPath configurations share one ONNX runtime. Use the same WASM assets or separate workers.', 'EMBEDDING_ENV_CONFLICT');
    if (typeof wasm.wasmPaths !== 'object' || wasm.wasmPaths.wasm !== reserved.wasm) invalid('Shared ONNX WASM assets changed after adapter initialization. Isolate independently configured runtimes in separate workers.', 'EMBEDDING_ENV_CONFLICT');
    return;
  }
  const existing = wasm.wasmPaths;
  const libraryDefault = typeof existing === 'object' && /^https:\/\/cdn\.jsdelivr\.net\/npm\/onnxruntime-web@[^/]+\/dist\//.test(existing.wasm);
  const identical = typeof existing === 'object' && existing.mjs === paths.mjs && existing.wasm === paths.wasm;
  if (existing && !identical && (!libraryDefault || wasm.wasmBinary !== undefined)) invalid('An independently configured ONNX WASM runtime already uses different assets. Configure wasmPath consistently or use a separate worker.', 'EMBEDDING_ENV_CONFLICT');
  wasm.wasmPaths = paths;
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  wasm.numThreads = isolated ? (Number.isInteger(wasm.numThreads) && wasm.numThreads! > 0 ? wasm.numThreads : Math.max(1, Math.min(4, typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency || 1))) : 1;
  wasmReservations.set(wasm, { ...paths });
}

function loadWithEnvironment(runtime: Runtime, config: TransformersConfig, options: OperationOptions, load: () => Promise<EmbeddingPipeline>): Promise<EmbeddingPipeline> {
  const pending = loaderQueue.then(async () => {
    checkAbort(options.signal);
    configureWasm(runtime, config);
    const env = runtime.env;
    const settings: Record<string, unknown> = {
      // Local access must also be enabled for cache-only loading in Transformers.js 4.x.
      allowLocalModels: Boolean(config.localModelPath) || !config.allowRemoteModels,
      allowRemoteModels: config.allowRemoteModels,
      useBrowserCache: typeof caches !== 'undefined',
      useWasmCache: typeof caches !== 'undefined' || env.useWasmCache === true,
    };
    if (config.localModelPath) settings.localModelPath = `${config.localModelPath.replace(/\/$/, '')}/`;
    const changed: Array<{ key: string; previous: unknown; own: boolean; value: unknown }> = [];
    let pipeline: EmbeddingPipeline | undefined;
    try {
      for (const [key, value] of Object.entries(settings)) {
        const entry = { key, previous: env[key], own: Object.hasOwn(env, key), value };
        env[key] = value;
        changed.push(entry);
      }
      pipeline = await load();
      return pipeline;
    } finally {
      const conflicts = changed.filter(entry => env[entry.key] !== entry.value);
      for (const entry of changed.reverse()) {
        if (env[entry.key] !== entry.value) continue;
        if (entry.own) env[entry.key] = entry.previous;
        else delete env[entry.key];
      }
      if (conflicts.length) {
        await pipeline?.dispose();
        invalid(`Shared Transformers.js env changed during loading (${conflicts.map(entry => entry.key).join(', ')}). Serialize external model loaders too, or isolate them in separate workers.`, 'EMBEDDING_ENV_CONFLICT');
      }
    }
  });
  loaderQueue = pending.then(() => undefined, () => undefined);
  return pending;
}

function waitWithAbort<T>(pending: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return pending;
  return new Promise<T>((resolve, reject) => {
    const abort = () => {
      signal.removeEventListener('abort', abort);
      reject(new DOMException('Operation aborted.', 'AbortError'));
    };
    signal.addEventListener('abort', abort, { once: true });
    pending.then(value => {
      signal.removeEventListener('abort', abort);
      if (signal.aborted) abort();
      else resolve(value);
    }, error => {
      signal.removeEventListener('abort', abort);
      reject(error);
    });
    if (signal.aborted) abort();
  });
}

function report(options: OperationOptions, progress: ProgressEvent): void {
  if (!options.signal?.aborted) {
    // Observers must not invalidate model initialization or inference results.
    try { options.onProgress?.(progress); } catch { /* Progress is best-effort. */ }
  }
}

const yieldTurn = () => new Promise<void>(resolve => setTimeout(resolve, 0));

async function tensorRows(output: unknown, count: number, normalize: boolean): Promise<Float32Array[]> {
  if (!output || typeof output !== 'object') invalid('Feature extraction must return a pooled tensor of shape [batch, dimension].', 'EMBEDDING_OUTPUT');
  const tensor = output as { dims?: unknown; type?: unknown; data?: unknown; location?: string; getData?: () => Promise<unknown>; ort_tensor?: { getData?: () => Promise<unknown> } };
  const dims = tensor.dims;
  if (!Array.isArray(dims) || dims.length !== 2 || dims[0] !== count || !Number.isInteger(dims[1]) || dims[1] < 1 || dims[1] > 16384) invalid(`Expected exactly ${count} pooled tensor rows [batch, dimension] with dimension 1..16384; unpooled rank-3 tensors must not be flattened. Check model and pooling support.`, 'EMBEDDING_OUTPUT');
  if (tensor.type !== 'float32' && tensor.type !== 'float64') invalid('Pooled embeddings must have float32 or float64 outputs, not integer/packed/float16 data. Choose an artifact with CPU float outputs (usually fp32).', 'EMBEDDING_OUTPUT');
  let data: unknown;
  try {
    const storage = tensor.ort_tensor ?? tensor;
    data = tensor.location && !['cpu', 'cpu-pinned'].includes(tensor.location) && storage.getData ? await storage.getData() : tensor.data;
  } catch (cause) {
    throw new SereneDBError('Unable to materialize pooled embeddings on the CPU. Choose a supported feature-extraction model or the WASM device.', 'EMBEDDING_OUTPUT', { cause });
  }
  if (!(data instanceof Float32Array || data instanceof Float64Array || Array.isArray(data)) || data.length !== count * dims[1]) invalid('Pooled tensor storage must contain exactly batch * dimension floating-point values.', 'EMBEDDING_OUTPUT');
  const rows: Float32Array[] = [];
  for (let row = 0; row < count; row++) {
    const start = row * dims[1];
    const values = Array.isArray(data) ? data.slice(start, start + dims[1]) : data.subarray(start, start + dims[1]);
    if (!values.some(value => typeof value === 'number' && value !== 0)) invalid('Model returned an embedding with a zero norm.', 'EMBEDDING_OUTPUT');
    try {
      const vector = prepareVector(values, normalize ? 'cosine' : 'dot');
      if (!vector.some(value => value !== 0)) invalid('Embedding values underflowed to a zero float32 vector.');
      rows.push(vector);
    } catch (cause) {
      throw new SereneDBError('Model returned nonfinite, invalid, or unrepresentable float32 embedding values.', 'EMBEDDING_OUTPUT', { cause });
    }
  }
  return rows;
}

/**
 * Lazy, serialized text embedding adapter. Inputs are raw text; only the selected document
 * or query prefix is applied once. Each prefixed input is limited to 131072 UTF-16 code units;
 * chunk larger documents explicitly instead of allocating an unbounded tokenizer buffer.
 * Batches bound padded token slots, not just the sum of unpadded lengths. Model-card pooling
 * and prompt choices remain the caller's responsibility; equal dimensions do not imply equal spaces.
 * External Transformers.js loaders must coordinate global env writes or use separate workers.
 */
export class TransformersEmbedding {
  readonly config: Readonly<TransformersConfig>;
  private queue = Promise.resolve();
  private initializing?: Promise<EmbeddingPipeline>;
  private pipeline?: EmbeddingPipeline;
  private disposal?: Promise<void>;
  private disposed = false;
  private warmed = false;
  private device: 'wasm' | 'webgpu' | null = null;
  private version: string | null = null;
  private dimension: number | null = null;
  private effectiveMaxTokens: number | null = null;
  private fallbackReason: string | null = null;
  private readonly warnings: string[] = [];

  constructor(config: TransformersConfig, private readonly loadRuntime: () => Promise<Runtime> = defaultRuntime) {
    this.config = Object.freeze(transformersEmbedder(config));
    if (!/^[\da-f]{40}$/i.test(this.config.revision)) this.warnings.push(`Revision '${this.config.revision}' is not an immutable commit SHA. Pin a full model revision and explicitly re-embed collections when artifacts change.`);
    if (this.config.localModelPath) this.warnings.push('Transformers.js ignores revision for local artifacts. Deploy immutable, versioned local model directories and migrate collections when their contents change.');
  }

  runtimeInfo(): TransformersRuntimeInfo {
    return {
      provider: 'transformers', model: this.config.model, revision: this.config.revision,
      revisionPinned: /^[\da-f]{40}$/i.test(this.config.revision) && !this.config.localModelPath,
      requestedDevice: this.config.device, device: this.device, dtype: this.config.dtype,
      version: this.version, loaded: Boolean(this.pipeline), warmed: this.warmed,
      disposed: this.disposed, dimension: this.dimension, effectiveMaxTokens: this.effectiveMaxTokens,
      fallbackReason: this.fallbackReason, warnings: [...this.warnings],
    };
  }

  async encodeDocuments(texts: string[], options: OperationOptions = {}): Promise<Float32Array[]> {
    checkAbort(options.signal);
    if (!Array.isArray(texts)) invalid('Documents must be an array of strings.');
    const copy = Array.from(texts);
    return this.enqueue(() => this.encode(copy, this.config.documentPrefix, options), options);
  }

  async encodeQuery(text: string, options: OperationOptions = {}): Promise<Float32Array> {
    return this.enqueue(async () => (await this.encode([text], this.config.queryPrefix, options))[0], options);
  }

  async preload(options: OperationOptions = {}): Promise<void> {
    return this.enqueue(async () => {
      const pipeline = await this.getPipeline(options);
      if (this.warmed) return;
      report(options, { phase: 'embedding', completed: 0, total: 1, detail: { warming: true } });
      await this.infer(pipeline, [`${this.config.documentPrefix}SereneDB model preload.`], options);
      this.warmed = true;
      report(options, { phase: 'embedding', completed: 1, total: 1, detail: { warming: true } });
    }, options);
  }

  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.disposed = true;
    this.disposal = this.queue.then(async () => {
      const pipeline = this.pipeline;
      this.pipeline = undefined;
      this.initializing = undefined;
      this.warmed = false;
      await pipeline?.dispose();
    });
    this.queue = this.disposal.then(() => undefined, () => undefined);
    return this.disposal;
  }

  private assertOpen(): void {
    if (this.disposed) invalid('Embedding adapter is disposed.', 'CLOSED');
  }

  private enqueue<T>(work: () => Promise<T>, options: OperationOptions): Promise<T> {
    this.assertOpen();
    checkAbort(options.signal);
    const pending = this.queue.then(async () => {
      this.assertOpen();
      checkAbort(options.signal);
      return work();
    });
    // Aborting a caller must not free the queue while its native inference is still running.
    this.queue = pending.then(() => undefined, () => undefined);
    return waitWithAbort(pending, options.signal);
  }

  private async getPipeline(options: OperationOptions): Promise<EmbeddingPipeline> {
    this.assertOpen();
    checkAbort(options.signal);
    if (!this.initializing) {
      this.initializing = this.initialize(options).then(async pipeline => {
        if (this.disposed) {
          await pipeline.dispose();
          this.assertOpen();
        }
        this.pipeline = pipeline;
        return pipeline;
      }).catch(error => {
        this.initializing = undefined;
        throw error;
      });
    }
    const pipeline = await this.initializing;
    checkAbort(options.signal);
    this.assertOpen();
    return pipeline;
  }

  private async initialize(options: OperationOptions): Promise<EmbeddingPipeline> {
    this.fallbackReason = null;
    report(options, { phase: 'loading-model', completed: 0, total: 1, detail: { model: this.config.model, revision: this.config.revision, warnings: [...this.warnings] } });
    checkAbort(options.signal);
    const runtime = await this.loadRuntime();
    checkAbort(options.signal);
    this.version = runtime.env.version ?? null;
    let device: 'wasm' | 'webgpu' = 'wasm';
    if (this.config.device !== 'wasm') {
      try {
        const gpu = typeof navigator === 'undefined' ? undefined : (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
        if (gpu && await gpu.requestAdapter()) device = 'webgpu';
        else this.fallbackReason = 'WebGPU adapter unavailable; using WASM.';
      } catch (error) {
        this.fallbackReason = `WebGPU adapter request failed; using WASM: ${error instanceof Error ? error.message : String(error)}`;
      }
    }
    checkAbort(options.signal);
    const pipeline = await loadWithEnvironment(runtime, this.config, options, async () => {
      const progress_callback = (detail: unknown) => {
        const progress = detail as { loaded?: number; total?: number } | null;
        const total = typeof progress?.total === 'number' && Number.isFinite(progress.total) && progress.total > 0 ? progress.total : 1;
        const completed = typeof progress?.loaded === 'number' && Number.isFinite(progress.loaded) ? Math.max(0, Math.min(progress.loaded, total)) : 0;
        report(options, { phase: 'loading-model', completed, total, detail });
      };
      const load = async (selected: 'wasm' | 'webgpu'): Promise<EmbeddingPipeline> => {
        const settings = { revision: this.config.revision, dtype: this.config.dtype, device: selected, local_files_only: !this.config.allowRemoteModels, progress_callback };
        const raw = await runtime.AutoTokenizer.from_pretrained(this.config.model, settings);
        checkAbort(options.signal);
        this.assertOpen();
        if (typeof raw !== 'function') invalid('Model does not provide a callable text tokenizer for feature extraction.', 'EMBEDDING_MODEL');
        const limit = raw.model_max_length;
        this.effectiveMaxTokens = typeof limit === 'number' && Number.isFinite(limit) && limit > 0 ? Math.min(this.config.maxTokens, limit) : this.config.maxTokens;
        // v4's feature-extraction pipeline does not forward per-call max_length.
        const tokenizer = new Proxy(raw, { apply: (target, receiver, args) => {
          const encoding = Reflect.apply(target, receiver, [args[0], { ...args[1], truncation: true, max_length: this.config.maxTokens }]);
          if (args[1]?.return_tensor !== false) {
            const dims = encoding?.input_ids?.dims;
            if (!Array.isArray(dims) || dims.length !== 2 || dims[0] !== args[0].length || !Number.isInteger(dims[1]) || dims[1] < 1 || dims[1] > this.config.maxTokens || dims[0] * dims[1] > this.config.batchTokens) invalid('Tokenizer did not honor maxTokens/batchTokens or returned unsupported model inputs.', 'EMBEDDING_MODEL');
          }
          return encoding;
        } });
        const model = await runtime.AutoModel.from_pretrained(this.config.model, settings);
        try {
          checkAbort(options.signal);
          this.assertOpen();
          return new runtime.FeatureExtractionPipeline({ task: 'feature-extraction', tokenizer, model });
        } catch (error) {
          await model.dispose();
          throw error;
        }
      };
      if (device === 'webgpu') {
        try { return await load('webgpu'); }
        catch (error) {
          checkAbort(options.signal);
          this.assertOpen();
          this.fallbackReason = `WebGPU model load failed; retrying the same dtype on WASM: ${error instanceof Error ? error.message : String(error)}`;
          report(options, { phase: 'loading-model', completed: 0, total: 1, detail: { fallback: this.fallbackReason } });
          device = 'wasm';
        }
      }
      try { return await load('wasm'); }
      catch (cause) {
        checkAbort(options.signal);
        this.assertOpen();
        if (cause instanceof SereneDBError) throw cause;
        throw new SereneDBError(`Unable to load embedding model '${this.config.model}' at revision '${this.config.revision}' with dtype '${this.config.dtype}' on WASM. Check the model's ONNX artifact/dtype and feature-extraction support; q8/other quantizations are not available for every model. For offline use, preload or deploy all tokenizer, model, and matching WASM assets. ${cause instanceof Error ? cause.message : String(cause)}`, 'EMBEDDING_MODEL', { cause });
      }
    });
    this.device = device;
    report(options, { phase: 'loading-model', completed: 1, total: 1, detail: { device, dtype: this.config.dtype, version: this.version, fallback: this.fallbackReason } });
    return pipeline;
  }

  private async encode(texts: string[], prefix: string, options: OperationOptions): Promise<Float32Array[]> {
    for (let i = 0; i < texts.length; i++) {
      if (typeof texts[i] !== 'string' || texts[i].length + prefix.length > MAX_INPUT_CHARACTERS) invalid(`Text ${i} must be a string of at most ${MAX_INPUT_CHARACTERS} characters including its configured prefix; chunk larger documents before embedding.`);
      if (i > 0 && i % 64 === 0) { await yieldTurn(); checkAbort(options.signal); }
    }
    report(options, { phase: 'embedding', completed: 0, total: texts.length });
    checkAbort(options.signal);
    if (!texts.length) return [];
    const pipeline = await this.getPipeline(options);
    const rows: Float32Array[] = [];
    let batch: string[] = [];
    let longest = 0;
    let lengthsKnown = true;
    const flush = async () => {
      if (!batch.length) return;
      rows.push(...await this.infer(pipeline, batch, options));
      batch = [];
      longest = 0;
      lengthsKnown = true;
      report(options, { phase: 'embedding', completed: rows.length, total: texts.length });
      await yieldTurn();
      checkAbort(options.signal);
      this.assertOpen();
    };
    for (let i = 0; i < texts.length; i++) {
      checkAbort(options.signal);
      if (i > 0 && i % 16 === 0) { await yieldTurn(); checkAbort(options.signal); }
      const text = prefix + texts[i];
      const ids = pipeline.tokenizer([text], { padding: false, truncation: true, max_length: this.config.maxTokens, return_tensor: false }).input_ids;
      // If an alternate tokenizer cannot expose lengths, budget its entire truncation limit.
      let length = this.config.maxTokens;
      const known = Array.isArray(ids) && ids.length === 1 && Array.isArray(ids[0]);
      if (known) length = (ids as unknown[][])[0].length;
      if (!Number.isInteger(length) || length < 1 || length > this.config.maxTokens) invalid('Tokenizer returned an invalid or untruncated token sequence.', 'EMBEDDING_MODEL');
      const mixedLastToken = this.config.pooling === 'last_token' && pipeline.tokenizer.padding_side !== 'left' && batch.length > 0 && (!known || !lengthsKnown || length !== longest);
      if (batch.length >= this.config.batchSize || Math.max(longest, length) * (batch.length + 1) > this.config.batchTokens || mixedLastToken) await flush();
      batch.push(text);
      longest = Math.max(longest, length);
      lengthsKnown &&= known;
    }
    await flush();
    checkAbort(options.signal);
    return rows;
  }

  private async infer(pipeline: EmbeddingPipeline, texts: string[], options: OperationOptions): Promise<Float32Array[]> {
    checkAbort(options.signal);
    this.assertOpen();
    let output: unknown;
    try {
      // Normalize cloned CPU rows ourselves to avoid overflowing naive tensor L2 norms.
      output = await pipeline(texts, { pooling: this.config.pooling, normalize: false, truncation: true, max_length: this.config.maxTokens });
      checkAbort(options.signal);
      this.assertOpen();
      const rows = await tensorRows(output, texts.length, this.config.normalize);
      checkAbort(options.signal);
      this.assertOpen();
      if (this.dimension !== null && rows[0].length !== this.dimension) invalid(`Model embedding dimension changed from ${this.dimension} to ${rows[0].length}.`, 'SCHEMA_MISMATCH');
      this.dimension = rows[0].length;
      return rows;
    } catch (cause) {
      checkAbort(options.signal);
      if (cause instanceof SereneDBError) throw cause;
      throw new SereneDBError(`Feature extraction failed for '${this.config.model}' using '${this.config.pooling}' pooling. Check the model card and its Transformers.js feature-extraction/pooling support. ${cause instanceof Error ? cause.message : String(cause)}`, 'EMBEDDING_MODEL', { cause });
    } finally {
      if (output && typeof output === 'object' && typeof (output as { dispose?: unknown }).dispose === 'function') {
        try { await (output as { dispose(): unknown }).dispose(); }
        catch { if (!this.warnings.includes('Unable to release an inference output tensor.')) this.warnings.push('Unable to release an inference output tensor.'); }
      }
    }
  }
}
