import test from 'node:test';
import assert from 'node:assert/strict';
import { TransformersEmbedding, transformersEmbedder } from '../src/transformers.js';
import { SereneDBError } from '../src/errors.js';
import type { ProgressEvent, TransformersConfig } from '../src/types.js';

type Runtime = Awaited<ReturnType<NonNullable<ConstructorParameters<typeof TransformersEmbedding>[1]>>>;
type Pipeline = InstanceType<Runtime['FeatureExtractionPipeline']>;
type Tokenizer = Pipeline['tokenizer'];

function gate() {
  let open!: () => void;
  const promise = new Promise<void>(resolve => { open = resolve; });
  return { promise, open };
}

function fakeRuntime(hooks: {
  length?: (text: string) => number;
  modelMaxLength?: number;
  paddingSide?: string;
  unknownLengths?: boolean;
  tokenizer?: (model: string, options: Record<string, unknown>) => Promise<void>;
  model?: (model: string, options: Record<string, unknown>) => Promise<void>;
  output?: (texts: string[], options: Record<string, unknown>) => unknown | Promise<unknown>;
} = {}) {
  const loads: Array<{ kind: string; model: string; options: Record<string, unknown>; env: Record<string, unknown> }> = [];
  const batches: Array<{ texts: string[]; options: Record<string, unknown>; dims: number[] }> = [];
  const tokenizations: Array<{ texts: string[]; options: Record<string, unknown> }> = [];
  let runtimeLoads = 0;
  let pipelines = 0;
  let disposed = 0;
  let tensorsDisposed = 0;
  const env: Runtime['env'] = {
    version: '4.3.1-test', allowLocalModels: false, allowRemoteModels: true,
    localModelPath: '/original/', useBrowserCache: false, useWasmCache: false,
    remoteHost: 'https://huggingface.co/', remotePathTemplate: '{model}/resolve/{revision}/',
    backends: { onnx: { wasm: { numThreads: 8 } } },
  };
  const snapshot = () => Object.fromEntries(['allowLocalModels', 'allowRemoteModels', 'localModelPath', 'useBrowserCache', 'useWasmCache', 'remotePathTemplate'].map(key => [key, env[key]]));
  const runtime: Runtime = {
    env,
    AutoTokenizer: { from_pretrained: async (model, options) => {
      loads.push({ kind: 'tokenizer', model, options: { ...options }, env: snapshot() });
      await hooks.tokenizer?.(model, options);
      return Object.assign((texts: string[], options: Record<string, unknown>) => {
        tokenizations.push({ texts: [...texts], options: { ...options } });
        const max = Math.min(Number(options.max_length ?? Infinity), hooks.modelMaxLength ?? 512);
        const lengths = texts.map(text => Math.min(hooks.length?.(text) ?? Math.max(2, text.length + 2), options.truncation ? max : Infinity));
        if (options.return_tensor === false) return { input_ids: hooks.unknownLengths ? undefined : lengths.map(length => Array(length).fill(1)) };
        return { input_ids: { dims: [texts.length, Math.max(...lengths)] } };
      }, { model_max_length: hooks.modelMaxLength ?? 512, padding_side: hooks.paddingSide ?? 'right' }) as Tokenizer;
    } },
    AutoModel: { from_pretrained: async (model, options) => {
      loads.push({ kind: 'model', model, options: { ...options }, env: snapshot() });
      await hooks.model?.(model, options);
      return { dispose: async () => { disposed++; } };
    } },
    FeatureExtractionPipeline: function ({ tokenizer, model }: { tokenizer: Tokenizer; model: { dispose(): unknown } }) {
      pipelines++;
      return Object.assign(async (texts: string[], options: Record<string, unknown>) => {
        const encoding = tokenizer(texts, { padding: true, truncation: true });
        const dims = (encoding.input_ids as { dims: number[] }).dims;
        batches.push({ texts: [...texts], options: { ...options }, dims });
        if (hooks.output) return hooks.output(texts, options);
        return {
          dims: [texts.length, 2], type: 'float32', location: 'cpu',
          data: Float32Array.from(texts.flatMap(() => [3, 4])),
          dispose: () => { tensorsDisposed++; },
        };
      }, { tokenizer, dispose: async () => { await model.dispose(); } });
    } as unknown as Runtime['FeatureExtractionPipeline'],
  };
  return {
    runtime, loads, batches, tokenizations, snapshot,
    loader: async () => { runtimeLoads++; return runtime; },
    counts: () => ({ runtimeLoads, pipelines, disposed, tensorsDisposed }),
  };
}

function embedding(fake: ReturnType<typeof fakeRuntime>, options: Partial<TransformersConfig> = {}) {
  return new TransformersEmbedding(transformersEmbedder({ model: 'test/embedding-model', ...options }), fake.loader);
}

function code(expected: string) {
  return (error: unknown) => error instanceof SereneDBError && error.code === expected;
}

test('config is populated, serializable, and does not guess a quantized artifact', () => {
  const config = transformersEmbedder({ model: 'test/model' });
  assert.deepEqual(config, {
    provider: 'transformers', model: 'test/model', revision: 'main', dtype: 'fp32', device: 'wasm',
    pooling: 'mean', normalize: true, queryPrefix: '', documentPrefix: '', maxTokens: 512,
    batchSize: 16, batchTokens: 8192, allowRemoteModels: true,
  });
  assert.deepEqual(JSON.parse(JSON.stringify(config)), config);
  assert.deepEqual(structuredClone(config), config);
  assert.equal(transformersEmbedder({ model: 'test/model', maxTokens: 16384 }).batchTokens, 16384);
  const configured = transformersEmbedder({ ...config, revision: 'a'.repeat(40), dtype: 'q8', device: 'auto', pooling: 'cls', normalize: false, queryPrefix: 'query: ', documentPrefix: 'passage: ', localModelPath: '/models/v1/', wasmPath: '/runtime/', allowRemoteModels: false });
  assert.equal(configured.dtype, 'q8');
  assert.equal(configured.localModelPath, '/models/v1/');
  assert.equal(configured.allowRemoteModels, false);
});

test('config rejects malformed, unsupported, and unbounded options', () => {
  assert.throws(() => transformersEmbedder({} as never), code('INVALID_INPUT'));
  for (const options of [
    null, [], { model: '' }, { model: 12 }, { model: ' model' }, { revision: '' },
    { revision: null }, { provider: 'other' }, { dtype: 'auto' }, { dtype: 'typo' },
    { device: 'cpu' }, { pooling: 'none' }, { pooling: 'max' }, { normalize: 1 },
    { allowRemoteModels: 'false' }, { queryPrefix: null }, { documentPrefix: 2 },
    { queryPrefix: 'x'.repeat(8193) }, { batchSize: 0 }, { batchSize: 1.5 }, { batchSize: 257 },
    { maxTokens: 0 }, { maxTokens: 32769 }, { maxTokens: 2n }, { batchTokens: 511 }, { batchTokens: Infinity },
    { localModelPath: '' }, { wasmPath: '' }, { wasmPath: 'javascript:alert(1)' },
    { wasmPath: '/runtime/?v=2' }, { localModelPath: '/models/#tag' }, { unknown: true },
  ]) {
    const input = options && !Array.isArray(options) ? { model: 'test/model', ...options } : options;
    assert.throws(() => transformersEmbedder(input as never), code('INVALID_INPUT'));
  }
  assert.throws(() => transformersEmbedder(Object.assign(new Date(), { model: 'test/model' }) as never));
});

test('adapter remains lazy for empty, invalid, and pre-aborted input', async () => {
  const fake = fakeRuntime();
  const adapter = embedding(fake);
  assert.equal(adapter.runtimeInfo().loaded, false);
  assert.deepEqual(await adapter.encodeDocuments([]), []);
  await assert.rejects(adapter.encodeDocuments('text' as never), code('INVALID_INPUT'));
  await assert.rejects(adapter.encodeDocuments(['valid', 12 as never]), code('INVALID_INPUT'));
  await assert.rejects(adapter.encodeQuery('x'.repeat(131073)), /chunk larger documents/);
  await assert.rejects(adapter.encodeDocuments(['valid', , 'sparse'] as string[]), code('INVALID_INPUT'));
  const controller = new AbortController(); controller.abort();
  await assert.rejects(adapter.preload({ signal: controller.signal }), { name: 'AbortError' });
  await assert.rejects(adapter.encodeQuery('query', { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(fake.counts().runtimeLoads, 0);
  await adapter.dispose();
});

test('document/query prefixes are applied once and pooled CPU rows own their data', async () => {
  const data = new Float32Array([3, 4, 5, 12]);
  const fake = fakeRuntime({ output: texts => ({ dims: [texts.length, 2], type: 'float32', data: data.subarray(0, texts.length * 2) }) });
  const adapter = embedding(fake, { documentPrefix: 'passage: ', queryPrefix: 'query: ', maxTokens: 32 });
  const inputs = ['one', 'two'];
  const pending = adapter.encodeDocuments(inputs); inputs[0] = 'mutated';
  const rows = await pending;
  assert.deepEqual(fake.batches[0].texts, ['passage: one', 'passage: two']);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].length, 2);
  assert.ok(Math.abs(rows[0][0] - 0.6) < 1e-6);
  assert.ok(Math.abs(rows[1][0] - 5 / 13) < 1e-6);
  assert.notEqual(rows[0].buffer, rows[1].buffer);
  assert.notEqual(rows[0].buffer, data.buffer);
  rows[0][0] = 42;
  assert.equal(data[0], 3);
  const query = await adapter.encodeQuery('one');
  assert.deepEqual(fake.batches[1].texts, ['query: one']);
  assert.ok(query instanceof Float32Array);
  assert.equal(fake.batches[0].options.pooling, 'mean');
  assert.equal(fake.batches[0].options.normalize, false);
  assert.equal(adapter.runtimeInfo().dimension, 2);
  await adapter.dispose();
});

test('normalization is configurable and stable for very large/small finite outputs', async () => {
  for (const value of [1e200, 1e-200]) {
    const fake = fakeRuntime({ output: texts => ({ type: 'float64', dims: [texts.length, 1], data: new Float64Array([value]) }) });
    const adapter = embedding(fake);
    assert.deepEqual(await adapter.encodeQuery('value'), new Float32Array([1]));
    await adapter.dispose();
  }
  const fake = fakeRuntime();
  const adapter = embedding(fake, { normalize: false });
  assert.deepEqual(await adapter.encodeQuery('value'), new Float32Array([3, 4]));
  await adapter.dispose();
  for (const value of [1e200, 1e-200]) {
    const fake = fakeRuntime({ output: () => ({ type: 'float64', dims: [1, 1], data: new Float64Array([value]) }) });
    const adapter = embedding(fake, { normalize: false });
    await assert.rejects(adapter.encodeQuery('value'), code('EMBEDDING_OUTPUT'));
    await adapter.dispose();
  }
});

test('preload warms once and concurrent calls share a serialized singleton pipeline', async () => {
  const fake = fakeRuntime();
  const adapter = embedding(fake, { batchSize: 1 });
  const progress: ProgressEvent[] = [];
  await Promise.all([adapter.preload({ onProgress: event => progress.push(event) }), adapter.preload(), adapter.encodeQuery('hello'), adapter.encodeDocuments(['one', 'two'])]);
  assert.equal(fake.counts().runtimeLoads, 1);
  assert.equal(fake.counts().pipelines, 1);
  assert.equal(fake.batches.filter(batch => batch.texts[0].includes('model preload')).length, 1);
  assert.equal(fake.batches.length, 4);
  assert.equal(fake.counts().tensorsDisposed, 4);
  assert.ok(progress.some(event => event.phase === 'loading-model' && event.completed === 1));
  assert.ok(progress.some(event => event.phase === 'embedding' && event.completed === 1));
  assert.equal(adapter.runtimeInfo().warmed, true);
  const disposal = adapter.dispose(); assert.equal(adapter.dispose(), disposal);
  await disposal;
  assert.equal(fake.counts().disposed, 1);
  assert.equal(adapter.runtimeInfo().loaded, false);
  await assert.rejects(adapter.encodeQuery('closed'), code('CLOSED'));
});

test('real token lengths bound padded batch slots and enforce schema truncation', async () => {
  const fake = fakeRuntime({ length: text => text.includes('long') ? 100 : 2, modelMaxLength: 64 });
  const adapter = embedding(fake, { maxTokens: 8, batchTokens: 12, batchSize: 4, documentPrefix: 'd:' });
  const rows = await adapter.encodeDocuments(['a', 'b', 'long', 'longer']);
  assert.equal(rows.length, 4);
  assert.deepEqual(fake.batches.map(batch => batch.texts), [['d:a', 'd:b'], ['d:long'], ['d:longer']]);
  for (const batch of fake.batches) assert.ok(batch.dims[0] * batch.dims[1] <= 12);
  for (const call of fake.tokenizations) assert.equal(call.options.max_length, 8);
  assert.equal(adapter.runtimeInfo().effectiveMaxTokens, 8);
  await adapter.dispose();
  const short = fakeRuntime({ modelMaxLength: 4 });
  const limited = embedding(short, { maxTokens: 8, batchTokens: 16, batchSize: 2 });
  await limited.encodeDocuments(['long input', 'another', 'final']);
  assert.deepEqual(short.batches.map(batch => batch.texts.length), [2, 1]);
  assert.equal(limited.runtimeInfo().effectiveMaxTokens, 4);
  await limited.dispose();
});

test('unknown token lengths use conservative budgets instead of character heuristics', async () => {
  const fake = fakeRuntime({ unknownLengths: true });
  const adapter = embedding(fake, { maxTokens: 8, batchTokens: 8, batchSize: 8 });
  await adapter.encodeDocuments(['a', 'b', 'c']);
  assert.deepEqual(fake.batches.map(batch => batch.texts.length), [1, 1, 1]);
  await adapter.dispose();
});

test('last_token does not pool right-padding and left-padding can batch mixed lengths', async () => {
  for (const unknownLengths of [false, true]) {
    const fake = fakeRuntime({ length: text => text.length, unknownLengths });
    const adapter = embedding(fake, { pooling: 'last_token', maxTokens: 8, batchTokens: 32 });
    await adapter.encodeDocuments(['aa', 'bb', 'c', 'ddd']);
    assert.deepEqual(fake.batches.map(batch => batch.texts.length), unknownLengths ? [1, 1, 1, 1] : [2, 1, 1]);
    assert.ok(fake.batches.every(batch => batch.options.pooling === 'last_token'));
    await adapter.dispose();
  }
  const fake = fakeRuntime({ length: text => text.length, paddingSide: 'left' });
  const adapter = embedding(fake, { pooling: 'last_token', maxTokens: 8, batchTokens: 32 });
  await adapter.encodeDocuments(['a', 'longer']);
  assert.equal(fake.batches.length, 1);
  await adapter.dispose();
});

test('invalid tensor ranks, row counts, dtypes, values, and storage are rejected', async () => {
  const outputs = [
    null, new Float32Array([1, 2]), { data: new Float32Array([1, 2]) },
    { dims: [1, 1, 2], data: new Float32Array([1, 2]) }, { dims: [2] },
    { dims: [2, 2] }, { dims: [1, 0] }, { dims: [1, 16385] }, { dims: [1, 1.5] },
    { type: 'float32', dims: [1, 2], data: new Float32Array([1]) },
    { type: 'float32', dims: [1, 2], data: new Float32Array([1, NaN]) },
    { type: 'float32', dims: [1, 2], data: new Float32Array([Infinity, 1]) },
    { type: 'float32', dims: [1, 2], data: new Float32Array([0, 0]) },
    { type: 'float32', dims: [1, 2], data: ['1', 2] },
    { type: 'float32', dims: [1, 2], data: Array(2) },
    { type: 'float16', dims: [1, 2], data: new Uint16Array([1, 2]) },
    { type: 'int8', dims: [1, 2], data: new Int8Array([1, 2]) },
  ];
  for (const output of outputs) {
    const fake = fakeRuntime({ output: () => output });
    const adapter = embedding(fake);
    await assert.rejects(adapter.encodeQuery('invalid'), code('EMBEDDING_OUTPUT'));
    await adapter.dispose();
  }
  let read = false;
  const fake = fakeRuntime({ output: () => ({ type: 'float32', dims: [1, 2, 2], get data() { read = true; throw new Error('must not flatten'); } }) });
  const adapter = embedding(fake);
  await assert.rejects(adapter.encodeQuery('rank3'), code('EMBEDDING_OUTPUT'));
  assert.equal(read, false);
  await adapter.dispose();
});

test('pooled GPU storage is materialized on the CPU and disposed after copying', async () => {
  let fetched = 0;
  let disposed = 0;
  const data = new Float32Array([3, 4]);
  const fake = fakeRuntime({ output: () => ({
    type: 'float32', dims: [1, 2], location: 'gpu-buffer',
    get data() { throw new Error('GPU data is not directly accessible'); },
    ort_tensor: { getData: async () => { fetched++; return data; } },
    dispose: () => { disposed++; },
  }) });
  const adapter = embedding(fake);
  const row = await adapter.encodeQuery('gpu');
  assert.equal(fetched, 1);
  assert.equal(disposed, 1);
  data[0] = 0;
  assert.ok(Math.abs(row[0] - 0.6) < 1e-6);
  await adapter.dispose();
});

test('model output dimension cannot change across calls', async () => {
  let dimension = 2;
  const fake = fakeRuntime({ output: () => ({ type: 'float32', dims: [1, dimension], data: new Float32Array(dimension).fill(1) }) });
  const adapter = embedding(fake);
  await adapter.encodeQuery('first'); dimension = 3;
  await assert.rejects(adapter.encodeQuery('second'), code('SCHEMA_MISMATCH'));
  assert.equal(adapter.runtimeInfo().dimension, 2);
  await adapter.dispose();
});

test('abort rejects promptly, discards late inference, and keeps the queue occupied', async () => {
  const entered = gate(); const release = gate();
  let inference = 0;
  let disposed = 0;
  const fake = fakeRuntime({ output: async texts => {
    inference++;
    if (inference === 1) { entered.open(); await release.promise; }
    return { type: 'float32', dims: [texts.length, 2], data: new Float32Array([3, 4]), dispose: () => { disposed++; } };
  } });
  const adapter = embedding(fake, { batchSize: 1 });
  const controller = new AbortController();
  const pending = adapter.encodeDocuments(['first', 'never'], { signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  await entered.promise;
  const next = adapter.encodeQuery('next');
  controller.abort(); await rejected;
  assert.equal(fake.batches.length, 1);
  assert.equal(adapter.runtimeInfo().dimension, null);
  release.open(); await next;
  assert.deepEqual(fake.batches.map(batch => batch.texts), [['first'], ['next']]);
  assert.equal(disposed, 2);
  await adapter.dispose();
});

test('cooperative yields let timers cancel between embedding batches', async () => {
  const fake = fakeRuntime();
  const adapter = embedding(fake, { batchSize: 1 });
  const controller = new AbortController();
  await assert.rejects(adapter.encodeDocuments(['first', 'second'], {
    signal: controller.signal,
    onProgress: event => { if (event.phase === 'embedding' && event.completed === 1) setTimeout(() => controller.abort(), 0); },
  }), { name: 'AbortError' });
  await adapter.dispose();
  assert.equal(fake.batches.length, 1);
});

test('failed initialization is retryable with actionable model/dtype errors', async () => {
  let attempts = 0;
  const fake = fakeRuntime({ model: async () => { if (++attempts === 1) throw new Error('404 model_quantized.onnx'); } });
  const adapter = embedding(fake, { dtype: 'q8', revision: 'b'.repeat(40) });
  await assert.rejects(adapter.encodeQuery('first'), error => code('EMBEDDING_MODEL')(error) && /q8.*not available|not available.*every model/.test((error as Error).message) && /404/.test((error as Error).message));
  assert.equal(adapter.runtimeInfo().loaded, false);
  await adapter.encodeQuery('retry');
  assert.equal(fake.counts().pipelines, 1);
  assert.equal(adapter.runtimeInfo().revisionPinned, true);
  await adapter.dispose();
});

test('realm loader mutex scopes offline/local env and rejects conflicting WASM assets', async () => {
  const entered = gate(); const release = gate();
  const fake = fakeRuntime({ model: async model => { if (model === 'test/first') { entered.open(); await release.promise; } } });
  const original = fake.snapshot();
  const first = embedding(fake, { model: 'test/first', localModelPath: '/models/v1/', allowRemoteModels: false });
  const second = embedding(fake, { model: 'test/second', localModelPath: '/models/v2/' });
  const loading = first.preload();
  await entered.promise;
  const other = second.preload();
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  assert.equal(fake.loads.length, 2);
  release.open(); await Promise.all([loading, other]);
  assert.deepEqual(fake.loads.map(load => [load.model, load.env.allowRemoteModels, load.env.allowLocalModels, load.env.localModelPath, load.options.local_files_only]), [
    ['test/first', false, true, '/models/v1/', true], ['test/first', false, true, '/models/v1/', true],
    ['test/second', true, true, '/models/v2/', false], ['test/second', true, true, '/models/v2/', false],
  ]);
  assert.deepEqual(fake.snapshot(), original);
  const paths = fake.runtime.env.backends!.onnx!.wasm!.wasmPaths as { mjs: string; wasm: string };
  assert.ok(paths.mjs.endsWith('/vendor/ort-wasm-simd-threaded.jsep.mjs'));
  assert.ok(paths.wasm.endsWith('/vendor/ort-wasm-simd-threaded.jsep.wasm'));
  assert.equal(fake.runtime.env.backends!.onnx!.wasm!.numThreads, 1);
  const conflict = embedding(fake, { wasmPath: '/different-runtime/' });
  await assert.rejects(conflict.preload(), code('EMBEDDING_ENV_CONFLICT'));
  await Promise.all([first.dispose(), second.dispose(), conflict.dispose()]);
});

test('external env changes during loading are not overwritten and abort the load', async () => {
  const fake = fakeRuntime({ model: async () => { fake.runtime.env.allowRemoteModels = false; } });
  const adapter = embedding(fake);
  await assert.rejects(adapter.preload(), code('EMBEDDING_ENV_CONFLICT'));
  assert.equal(fake.runtime.env.allowRemoteModels, false);
  assert.equal(fake.counts().disposed, 1);
  assert.equal(adapter.runtimeInfo().loaded, false);
  await adapter.dispose();
});

test('same model/revision cannot reuse memoized configs from a different local root', async () => {
  const fake = fakeRuntime();
  const first = embedding(fake, { localModelPath: '/models/version-one/' });
  const second = embedding(fake, { localModelPath: '/models/version-two/' });
  await first.preload();
  await assert.rejects(second.preload(), code('EMBEDDING_ENV_CONFLICT'));
  assert.equal(fake.counts().pipelines, 1);
  await Promise.all([first.dispose(), second.dispose()]);
});

test('tokenizer metadata preflight stays pinned even when upstream omits revision', async () => {
  const revision = 'c'.repeat(40);
  const cached = new Set([`https://huggingface.co/test/embedding-model/resolve/${revision}/tokenizer_config.json`]);
  const probes: string[] = [];
  const fake = fakeRuntime({ tokenizer: async model => {
    // Transformers.js4.3.1 get_tokenizer_files probes metadata with default options {}.
    const template = (fake.runtime.env.remotePathTemplate as string).replaceAll('{model}', model).replaceAll('{revision}', 'main');
    const url = `${fake.runtime.env.remoteHost}${template}tokenizer_config.json`;
    probes.push(url);
    if (!cached.has(url)) throw new Error(`Offline cache miss: ${url}`);
  } });
  const original = fake.snapshot();
  const adapter = embedding(fake, { revision, allowRemoteModels: false });
  await adapter.preload();
  assert.deepEqual(probes, [...cached]);
  assert.deepEqual(fake.snapshot(), original);
  await adapter.dispose();
});

test('provenance reports mutable/local revisions and snapshots cannot mutate adapter config', async () => {
  const fake = fakeRuntime();
  const input = transformersEmbedder({ model: 'test/model' });
  const adapter = new TransformersEmbedding(input, fake.loader);
  input.queryPrefix = 'changed';
  const info = adapter.runtimeInfo();
  assert.equal(info.revisionPinned, false);
  assert.ok(info.warnings[0].includes('immutable commit SHA'));
  info.warnings.length = 0;
  assert.equal(adapter.runtimeInfo().warnings.length, 1);
  assert.equal(adapter.config.queryPrefix, '');
  assert.deepEqual(structuredClone(info), info);
  await adapter.dispose();
  const local = embedding(fake, { revision: 'a'.repeat(40), localModelPath: '/models/v1/' });
  assert.equal(local.runtimeInfo().revisionPinned, false);
  assert.ok(local.runtimeInfo().warnings[0].includes('ignores revision'));
  await local.dispose();
});

test('WebGPU adapter probing/load failure falls back to WASM without changing dtype', async () => {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { hardwareConcurrency: 8, gpu: { requestAdapter: async () => ({}) } } });
  try {
    const fake = fakeRuntime({ model: async (_model, options) => { if (options.device === 'webgpu') throw new Error('WebGPU unsupported operator'); } });
    const adapter = embedding(fake, { device: 'auto', dtype: 'q8' });
    await adapter.preload();
    assert.deepEqual(fake.loads.filter(load => load.kind === 'model').map(load => [load.options.device, load.options.dtype]), [['webgpu', 'q8'], ['wasm', 'q8']]);
    assert.equal(adapter.runtimeInfo().device, 'wasm');
    assert.ok(adapter.runtimeInfo().fallbackReason!.includes('unsupported operator'));
    await adapter.dispose();
    Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { gpu: { requestAdapter: async () => { throw new Error('GPU policy disabled'); } } } });
    const noGpu = fakeRuntime(); const fallback = embedding(noGpu, { device: 'auto' });
    await fallback.preload();
    assert.ok(fallback.runtimeInfo().fallbackReason!.includes('GPU policy disabled'));
    assert.equal(noGpu.loads.filter(load => load.kind === 'model')[0].options.device, 'wasm');
    await fallback.dispose();
  } finally {
    if (previous) Object.defineProperty(globalThis, 'navigator', previous);
    else Reflect.deleteProperty(globalThis, 'navigator');
  }
});

test('progress observer failures do not change loading or inference results', async () => {
  const fake = fakeRuntime();
  const adapter = embedding(fake);
  await adapter.preload({ onProgress: () => { throw new Error('consumer callback'); } });
  assert.equal(fake.counts().disposed, 0);
  assert.equal(adapter.runtimeInfo().loaded, true);
  await adapter.encodeQuery('query', { onProgress: () => { throw undefined; } });
  assert.equal(fake.counts().pipelines, 1);
  await adapter.dispose();
});

test('operation options are snapshotted so a caller cannot disable late abort checks', async () => {
  const entered = gate(); const release = gate();
  const fake = fakeRuntime({ output: async () => {
    entered.open(); await release.promise;
    return { type: 'float32', dims: [1, 2], data: new Float32Array([3, 4]) };
  } });
  const adapter = embedding(fake);
  const controller = new AbortController();
  const options = { signal: controller.signal as AbortSignal | undefined };
  const pending = adapter.encodeQuery('query', options);
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  await entered.promise; options.signal = undefined;
  controller.abort(); await rejected;
  release.open(); await adapter.dispose();
  assert.equal(adapter.runtimeInfo().dimension, null);
});

test('aborting model initialization releases late resources and a new call can retry', async () => {
  const entered = gate(); const release = gate();
  let attempt = 0;
  const fake = fakeRuntime({ model: async () => { if (++attempt === 1) { entered.open(); await release.promise; } } });
  const adapter = embedding(fake);
  const controller = new AbortController();
  const pending = adapter.preload({ signal: controller.signal });
  const rejected = assert.rejects(pending, { name: 'AbortError' });
  await entered.promise;
  controller.abort(); await rejected;
  const retry = adapter.preload(); release.open(); await retry;
  assert.equal(fake.counts().disposed, 1);
  assert.equal(fake.counts().pipelines, 1);
  assert.equal(adapter.runtimeInfo().warmed, true);
  await adapter.dispose();
});

test('tensor cleanup failures preserve successful output and invalid-output errors', async () => {
  let invalid = false;
  const fake = fakeRuntime({ output: () => ({ type: 'float32', dims: invalid ? [1, 1, 2] : [1, 2], data: new Float32Array([3, 4]), dispose: () => { throw new Error('cleanup failed'); } }) });
  const adapter = embedding(fake);
  assert.equal((await adapter.encodeQuery('valid')).length, 2);
  invalid = true;
  await assert.rejects(adapter.encodeQuery('invalid'), code('EMBEDDING_OUTPUT'));
  assert.equal(adapter.runtimeInfo().warnings.filter(warning => warning.includes('release an inference output tensor')).length, 1);
  await adapter.dispose();
});

test('dispose during model loading releases the late model and is terminal', async () => {
  const entered = gate(); const release = gate();
  const fake = fakeRuntime({ model: async () => { entered.open(); await release.promise; } });
  const adapter = embedding(fake);
  const loading = adapter.preload();
  const rejected = assert.rejects(loading, code('CLOSED'));
  await entered.promise;
  const disposal = adapter.dispose(); release.open();
  await Promise.all([disposal, rejected]);
  assert.equal(fake.counts().disposed, 1);
  assert.equal(fake.counts().pipelines, 0);
  await assert.rejects(adapter.preload(), code('CLOSED'));
});
