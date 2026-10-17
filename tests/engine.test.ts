import 'fake-indexeddb/auto';
import test from 'node:test';
import assert from 'node:assert/strict';
import { SereneDB } from '../src/index.js';
import type { DatabaseExport } from '../src/types.js';

const fresh = () => SereneDB.open(`engine-${crypto.randomUUID()}`, { worker: false, wasm: false });

test('atomic validation, persistent cold restart, and immutable typed vectors', async () => {
  const db = await fresh();
  const docs = await db.collection('docs');
  await assert.rejects(docs.upsert([{ id: 'a', vector: [1, 0, 0] }, { id: 'b', vector: [1, 0] }]));
  assert.equal((await docs.state()).schema.dimension, null);
  const input = new Float32Array([1, 0]);
  const saving = docs.upsert({ id: 'a', vector: input, metadata: { label: 'original' } });
  input[0] = 0; await saving;
  const output = (await docs.get('a'))!; output.vector[0] = 0; output.metadata.label = 'modified';
  assert.equal((await docs.get('a'))!.vector[0], 1);
  await docs.unload();
  assert.equal(await docs.count(), 1);
  await db.close();
  const reopened = await SereneDB.open(db.name, { worker: false, wasm: false });
  const restored = await reopened.collection('docs');
  assert.equal(await restored.count(), 1);
  assert.equal((await restored.search({ vector: [1, 0], mode: 'exact' })).results[0].id, 'a');
  await reopened.destroy();
});

test('SharedArrayBuffer vectors are copied at invocation rather than shared with queued work', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const shared = new Float32Array(new SharedArrayBuffer(8)); shared.set([1, 0]);
  const saving = docs.upsert({ id: 'a', vector: shared }); shared.set([0, 1]); await saving;
  assert.deepEqual((await docs.get('a'))!.vector, new Float32Array([1, 0]));
  shared.set([1, 0]); const searching = docs.search({ vector: shared }); shared.set([0, 1]);
  assert.equal((await searching).results[0].score, 1);
  await db.destroy();
});

test('cross connection schema and snapshots converge at each operation', async () => {
  const db = await fresh();
  const other = await SereneDB.open(db.name, { worker: false, wasm: false });
  const a = await db.collection('docs'), b = await other.collection('docs');
  await a.upsert({ id: 'a', vector: [1, 0] });
  await assert.rejects(b.upsert({ id: 'bad', vector: [1, 0, 0] }));
  await b.upsert({ id: 'b', vector: [0, 1] });
  assert.equal(await a.count(), 2);
  assert.equal((await a.delete('b')).deleted, 1);
  assert.equal(await b.has('b'), false);
  await other.close(); await db.destroy();
});

test('no-op delete never publishes a peer revision into an incomplete index', async () => {
  const db = await fresh(), other = await SereneDB.open(db.name, { worker: false, wasm: false });
  const a = await db.collection('docs'), b = await other.collection('docs');
  await a.upsert({ id: 'old', vector: [1, 0] });
  await a.search({ vector: [1, 0] });
  await b.upsert({ id: 'peer', vector: [0, 1] });
  assert.equal((await a.delete('absent')).deleted, 0);
  assert.deepEqual((await a.search({ vector: [1, 0], topK: 10 })).results.map(item => item.id), ['old', 'peer']);
  assert.equal(await a.count(), 2);
  await other.close(); await db.destroy();
});

test('drop invalidates old handles; destruction serializes writes', async () => {
  const db = await fresh();
  const old = await db.collection('docs');
  await old.upsert({ id: 'a', vector: [1, 0] });
  await old.drop();
  const current = await db.collection('docs', { dimension: 3 });
  await assert.rejects(old.get('a'));
  await current.upsert({ id: 'b', vector: [1, 0, 0] });
  const saving = current.upsert({ id: 'pending', vector: [1, 1, 0] });
  const destroying = db.destroy();
  await Promise.all([saving, destroying]);
  const reopened = await SereneDB.open(db.name, { worker: false, wasm: false });
  assert.deepEqual(await reopened.listCollections(), []);
  await reopened.destroy();
});

test('metrics cutoffs filters negative cosine and hybrid exact identifiers', async () => {
  const db = await fresh();
  const l2 = await db.collection('l2', { metric: 'euclidean' });
  await l2.upsert([{ id: 'exact', vector: [0, 0] }, { id: 'near', vector: [0.25, 0] }, { id: 'far', vector: [2, 0] }]);
  assert.deepEqual((await l2.search({ vector: [0, 0], maxDistance: 0.5 })).results.map(record => record.id), ['exact', 'near']);
  assert.equal((await l2.search({ vector: [0, 0], topK: 1, mmr: .5 })).results[0].id, 'exact');
  const docs = await db.collection('docs');
  await docs.upsert([
    { id: 'a', vector: [1, 0], text: 'Ticket ABC_123 database engine', metadata: { category: 'db' } },
    { id: 'b', vector: [0, 1], text: 'baking bread', metadata: { category: 'food' } },
    { id: 'c', vector: [-1, 0], metadata: { category: 'db' } },
  ]);
  assert.deepEqual((await docs.search({ vector: [1, 0], topK: 3 })).results.map(record => record.id), ['a', 'b', 'c']);
  assert.deepEqual((await docs.search({ vector: [1, 0], filter: { category: 'db' } })).results.map(record => record.id), ['a', 'c']);
  assert.equal((await docs.search({ text: 'ABC_123', mode: 'hybrid' })).results[0].id, 'a');
  await assert.rejects(docs.search({ vector: [1, 0, 0] }));
  await assert.rejects(docs.search({ vector: [1, 0], topK: -1 }));
  await db.destroy();
});

test('exact queries load packed rows without an unused HNSW build then upgrade lazily', async () => {
  const db = await fresh(), docs = await db.collection('docs', { dimension: 32, index: { annThreshold: 1 } });
  await docs.upsert(Array.from({ length: 30 }, (_, i) => ({ id: `r-${i}`, vector: Array.from({ length: 32 }, (_, j) => Math.sin(i * 32 + j)) })));
  await docs.search({ vector: Array.from({ length: 32 }, (_, i) => Math.cos(i)), mode: 'exact' });
  assert.equal((await docs.stats() as { graphReady: boolean }).graphReady, false);
  await docs.search({ vector: Array.from({ length: 32 }, (_, i) => Math.cos(i)), mode: 'ann' });
  assert.equal((await docs.stats() as { graphReady: boolean }).graphReady, true);
  await db.destroy();
});

test('import merge counts are derived and export is isolated', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  await docs.upsert({ id: 'a', vector: [1, 0] });
  const exported = await db.export();
  exported.collections[0].state.count = 999;
  exported.collections[0].records[0].id = 'b';
  await db.import(exported);
  assert.equal(await docs.count(), 2);
  exported.collections[0].records[0].vector = [NaN, 1];
  await assert.rejects(db.import(exported));
  assert.equal(await docs.count(), 2);
  const malformed = { ...exported, version: 1 } as unknown as DatabaseExport;
  await assert.rejects(db.import(malformed));
  await db.destroy();
});

test('import preflights record-inferred dimensions before committing any collection', async () => {
  const db = await fresh(), a = await db.collection('a'), b = await db.collection('b', { dimension: 3 });
  await a.upsert({ id: 'existing', vector: [1, 0] });
  const exported = await db.export();
  exported.collections[0].records[0].id = 'new';
  exported.collections[1].state.schema.dimension = null;
  exported.collections[1].records.push({ collection: 'b', id: 'bad', vector: [1, 0], metadata: {}, timestamp: 1 });
  await assert.rejects(db.import(exported));
  assert.equal(await a.has('new'), false); assert.equal(await b.count(), 0);
  await db.destroy();
});

test('import refuses a concurrently replaced destination even at the same dimension', async () => {
  const { Storage } = await import('../src/storage.js');
  const { collectionSchema } = await import('../src/validation.js');
  const db = await fresh(), docs = await db.collection('docs');
  await docs.upsert({ id: 'a', vector: [3, 4] });
  const exported = await db.export();
  const peer = await Storage.open(db.name), original = Storage.prototype.ensureCollection;
  let injected = false;
  Storage.prototype.ensureCollection = async function (name, schema, options) {
    if (!injected && options?.expectedGeneration) {
      injected = true; await peer.dropCollection(name);
      await original.call(peer, name, collectionSchema({ dimension: 2, metric: 'euclidean' }));
    }
    return original.call(this, name, schema, options);
  };
  try { await assert.rejects(db.import(exported)); }
  finally { Storage.prototype.ensureCollection = original; }
  const replacement = await peer.readCollection('docs');
  assert.equal(replacement.state.schema.metric, 'euclidean'); assert.equal(replacement.records.length, 0);
  peer.close(); await db.destroy();
});

test('derived index failure never changes successful durable deletion into rejection', async () => {
  const { VectorIndex } = await import('../src/vector-index.js');
  const db = await fresh(), docs = await db.collection('docs');
  await docs.upsert({ id: 'a', vector: [1, 0] }); await docs.search({ vector: [1, 0] });
  const original = VectorIndex.prototype.remove;
  VectorIndex.prototype.remove = () => { throw new Error('Injected derived failure'); };
  try { assert.equal((await docs.delete('a')).deleted, 1); }
  finally { VectorIndex.prototype.remove = original; }
  assert.equal(await docs.get('a'), null); await db.destroy();
});

test('pre-aborted operations do not mutate or poison queues', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const controller = new AbortController(); controller.abort();
  await assert.rejects(docs.upsert({ id: 'a', vector: [1] }, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(await docs.count(), 0);
  await docs.upsert({ id: 'b', vector: [1] });
  assert.equal(await docs.count(), 1);
  await assert.rejects(db.destroy({ signal: controller.signal }), { name: 'AbortError' });
  assert.equal(await docs.count(), 1);
  await db.destroy();
});

test('queued operation options are immutable snapshots of the cancellation contract', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const controller = new AbortController();
  const options: { signal?: AbortSignal } = { signal: controller.signal };
  const pending = docs.upsert({ id: 'a', vector: [1, 0] }, options);
  controller.abort(); delete options.signal;
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(await docs.count(), 0); await db.destroy();
});

test('terminal close drains an otherwise-full queue', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const queued = Array.from({ length: 128 }, () => docs.state());
  const closing = db.close();
  await Promise.all(queued); await closing;
  await assert.rejects(docs.state());
  const reopened = await SereneDB.open(db.name, { worker: false, wasm: false }); await reopened.destroy();
});

test('empty hybrid cutoffs and lexical-only cutoff applicability are explicit', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  assert.deepEqual((await docs.search({ vector: [1], mode: 'hybrid', minScore: 0 })).results, []);
  await docs.upsert({ id: 'a', vector: [1], text: 'ticket abc' });
  await assert.rejects(docs.search({ text: 'ticket', mode: 'hybrid', minScore: 999 }));
  const iterator = docs.iterate({ signal: AbortSignal.abort() }); await assert.rejects(iterator.next());
  await db.destroy();
});

test('stream ingestion has bounded independent commits and invalid batch sizes fail', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const input = Array.from({ length: 15 }, (_, i) => ({ id: `r-${i}`, vector: [1, i] }));
  const result = await docs.ingest(input, { batchSize: 4 });
  assert.equal(result.upserted, 15); assert.equal(await docs.count(), 15);
  await assert.rejects(docs.ingest(input, { batchSize: 0 }));
  await docs.delete(['r-1', 'r-2']);
  assert.equal(await docs.count(), 13);
  assert.equal((await docs.search({ vector: [1, 0], topK: 15 })).results.length, 13);
  await db.destroy();
});

test('streaming snapshots reusable producer records before advancing the iterator', async () => {
  const db = await fresh(), docs = await db.collection('docs', { metric: 'dot' });
  const reusable = { id: '', vector: [1] };
  function* source() { for (let i = 0; i < 3; i++) { reusable.id = `r${i}`; reusable.vector[0] = i + 1; yield reusable; } }
  await docs.ingest(source(), { batchSize: 3 });
  assert.equal(await docs.count(), 3);
  for (let i = 0; i < 3; i++) assert.equal((await docs.get(`r${i}`))!.vector[0], i + 1);
  await db.destroy();
});

test('same-dimensional model-space differences reject without inference', async () => {
  const { transformersEmbedder } = await import('../src/transformers.js');
  const db = await fresh();
  const config = transformersEmbedder({ model: 'test/model', revision: 'a'.repeat(40) });
  const docs = await db.collection('embedded', { dimension: 2, embedder: config });
  const space = (await docs.state()).schema.embedding!.fingerprint;
  await assert.rejects(docs.upsert({ id: 'bad', vector: [1, 0] }));
  await docs.upsert({ id: 'valid', vector: [1, 0], space });
  assert.equal((await docs.search({ vector: [1, 0], space })).results[0].id, 'valid');
  await assert.rejects(docs.search({ vector: [1, 0], space: 'other' }));
  await assert.rejects(db.collection('embedded', { embedder: { ...config, documentPrefix: 'passage: ' } }));
  assert.equal(await docs.count(), 1); await db.destroy();
});

test('model residency waits for native disposal before loading a replacement', async () => {
  const { TransformersEmbedding, transformersEmbedder } = await import('../src/transformers.js');
  const preload = TransformersEmbedding.prototype.preload, dispose = TransformersEmbedding.prototype.dispose;
  let live = 0, maximum = 0, release: (() => void) | undefined;
  const loaded = new Set<TransformersEmbedding>();
  TransformersEmbedding.prototype.preload = async function () { if (!loaded.has(this)) { loaded.add(this); live++; maximum = Math.max(maximum, live); } };
  TransformersEmbedding.prototype.dispose = async function () {
    if (!loaded.has(this)) return;
    if (release === undefined) await new Promise<void>(resolve => { release = resolve; });
    loaded.delete(this); live--;
  };
  const db = await fresh();
  try {
    const config = transformersEmbedder({ model: 'test/model', revision: 'a'.repeat(40) });
    const a = await db.collection('a', { embedder: config }), b = await db.collection('b', { embedder: config });
    await a.preload(); const second = b.preload();
    while (!release) await new Promise(resolve => setTimeout(resolve, 0));
    assert.equal(live, 1); release(); await second;
    assert.equal(maximum, 1); await db.destroy(); assert.equal(live, 0);
  } finally { TransformersEmbedding.prototype.preload = preload; TransformersEmbedding.prototype.dispose = dispose; }
});

test('legacy migration is explicit and does not touch the source', async () => {
  const { migrateLegacy } = await import('../src/legacy.js');
  const source = `legacy-${crypto.randomUUID()}`, target = `target-${crypto.randomUUID()}`;
  const raw = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(source, 1);
    request.onupgradeneeded = () => request.result.createObjectStore('vectors', { keyPath: 'id' }).put({ id: 'old', vector: [1, 0], metadata: { source: true } });
    request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error);
  }); raw.close();
  assert.equal((await migrateLegacy(source, target, 'cosine', { worker: false, wasm: false })).count, 1);
  const reopened = await SereneDB.open(target, { worker: false, wasm: false });
  assert.equal((await (await reopened.collection()).get('old'))!.vector[0], 1); await reopened.destroy();
  const legacy = await new Promise<IDBDatabase>(resolve => { const request = indexedDB.open(source); request.onsuccess = () => resolve(request.result); });
  assert.equal(legacy.version, 1); assert.equal(legacy.objectStoreNames.contains('vectors'), true); legacy.close();
  await new Promise<void>(resolve => { const request = indexedDB.deleteDatabase(source); request.onsuccess = () => resolve(); });
});
