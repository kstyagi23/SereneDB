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

test('pre-aborted operations do not mutate or poison queues', async () => {
  const db = await fresh(), docs = await db.collection('docs');
  const controller = new AbortController(); controller.abort();
  await assert.rejects(docs.upsert({ id: 'a', vector: [1] }, { signal: controller.signal }), { name: 'AbortError' });
  assert.equal(await docs.count(), 0);
  await docs.upsert({ id: 'b', vector: [1] });
  assert.equal(await docs.count(), 1);
  await db.destroy();
});
