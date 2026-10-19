import test from 'node:test';
import assert from 'node:assert/strict';
import { VectorIndex } from '../src/vector-index.js';
import type { VectorDistanceKernel, VectorDistanceKernelFactory } from '../src/vector-index.js';
import { SereneDBError } from '../src/errors.js';
import { prepareVector } from '../src/validation.js';
import type { IndexConfig, IndexSnapshot, Metric } from '../src/types.js';

const config = (overrides: Partial<IndexConfig> = {}): IndexConfig => ({ type: 'hnsw', m: 16, efConstruction: 128, efSearch: 64, annThreshold: 0, ...overrides });
const vector = (values: number[], metric: Metric = 'cosine') => prepareVector(values, metric);
const identifiers = (matches: { id: string }[]) => matches.map(match => match.id);
const approximately = (actual: number, expected: number, epsilon = 1e-6) => assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} != ${expected}`);

function random(seed: number): () => number {
  return () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
}

function normalizedRandom(dimension: number, next: () => number, center?: Float32Array, noise = 1): Float32Array {
  const values = new Float32Array(dimension);
  for (let i = 0; i < dimension; i++) values[i] = (center?.[i] ?? 0) + (next() * 2 - 1) * noise;
  return prepareVector(values, 'cosine');
}

test('exact cosine math, zero and negative scores, deterministic ID ties', async () => {
  const index = new VectorIndex(2, 'cosine', config({ type: 'flat' }));
  index.add('negative', vector([-1, 0]));
  index.add('zero', vector([0, 1]));
  index.add('b', vector([1, 0]));
  index.add('a', vector([1, 0]));
  const results = await index.search(vector([1, 0]), { topK: 10 });
  assert.deepEqual(results, [
    { id: 'a', score: 1 }, { id: 'b', score: 1 }, { id: 'zero', score: 0 }, { id: 'negative', score: -1 },
  ]);
  assert.deepEqual(identifiers(await index.search(vector([1, 0]), { topK: 1 })), ['a']);
  assert.deepEqual(identifiers(await index.search(vector([1, 0]), { topK: 10, minScore: 0 })), ['a', 'b', 'zero']);
  assert.deepEqual(await index.search(vector([1, 0]), { topK: 0 }), []);
  const angled = new VectorIndex(2, 'cosine', config());
  angled.add('angled', vector([3, 4]));
  approximately((await angled.search(vector([1, 0]), { topK: 1, exact: true }))[0].score, 0.6);
});

test('exact dot product is not normalized and preserves negative scores', async () => {
  const index = new VectorIndex(2, 'dot', config());
  index.add('a', vector([2, 3], 'dot'));
  index.add('b', vector([-2, -3], 'dot'));
  index.add('zero', vector([0, 0], 'dot'));
  assert.deepEqual(await index.search(vector([4, 5], 'dot'), { topK: 3, exact: true }), [
    { id: 'a', score: 23 }, { id: 'zero', score: 0 }, { id: 'b', score: -23 },
  ]);
  assert.deepEqual(identifiers(await index.search(vector([4, 5], 'dot'), { topK: 3, minScore: -10, exact: true })), ['a', 'zero']);
});

test('Euclidean squared ranking returns true distance and negative score', async () => {
  const index = new VectorIndex(2, 'euclidean', config());
  index.add('far', vector([3, 4], 'euclidean'));
  index.add('near', vector([0.25, 0], 'euclidean'));
  index.add('exact', vector([0, 0], 'euclidean'));
  assert.deepEqual(await index.search(vector([0, 0], 'euclidean'), { topK: 3, exact: true }), [
    { id: 'exact', score: 0, distance: 0 }, { id: 'near', score: -0.25, distance: 0.25 }, { id: 'far', score: -5, distance: 5 },
  ]);
  for (const exact of [true, false]) {
    assert.deepEqual(identifiers(await index.search(vector([0, 0], 'euclidean'), { topK: 3, maxDistance: 0.25, exact })), ['exact', 'near']);
    assert.deepEqual(identifiers(await index.search(vector([0, 0], 'euclidean'), { topK: 3, minScore: -0.25, exact })), ['exact', 'near']);
    assert.deepEqual(identifiers(await index.search(vector([0, 0], 'euclidean'), { topK: 3, minScore: 0, exact })), ['exact']);
    assert.deepEqual(await index.search(vector([0, 0], 'euclidean'), { topK: 3, minScore: 0.1, exact }), []);
  }
});

test('bounded top-k heap agrees with independent full-sort oracle', async () => {
  const next = random(12345);
  for (const metric of ['cosine', 'dot', 'euclidean'] as const) {
    const index = new VectorIndex(7, metric, config({ type: 'flat' }));
    const rows: { id: string; vector: Float32Array }[] = [];
    for (let i = 0; i < 257; i++) {
      const values = Array.from({ length: 7 }, () => Math.round((next() * 2 - 1) * 3));
      if (values.reduce((sum, value) => sum + Math.abs(value), 0) === 0) values[0] = 1;
      const row = { id: `row-${String(256 - i).padStart(3, '0')}`, vector: vector(values, metric) };
      rows.push(row); index.add(row.id, row.vector);
    }
    const query = vector([1, -2, 3, 0, 2, -1, 4], metric);
    const expected = rows.map(row => {
      let value = 0;
      for (let i = 0; i < query.length; i++) value += metric === 'euclidean' ? (query[i] - row.vector[i]) ** 2 : query[i] * row.vector[i];
      return { id: row.id, score: metric === 'euclidean' ? -Math.sqrt(value) : metric === 'cosine' ? Math.max(-1, Math.min(1, value)) : value };
    }).sort((a, b) => b.score - a.score || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const k of [1, 2, 3, 7, 16, 64, 256, 257, 1000]) {
      const result = await index.search(query, { topK: k });
      assert.deepEqual(identifiers(result), identifiers(expected.slice(0, k)), `${metric}, k=${k}`);
      for (let i = 0; i < result.length; i++) approximately(result[i].score, expected[i].score);
    }
  }
});

test('packed contiguous growth, cloning, upserts, deletion and tombstone stats', async () => {
  const index = new VectorIndex(2, 'dot', config());
  const input = new Float32Array([1, 0]);
  index.add('a', input); input[0] = 100;
  const copy = index.get('a')!; copy[0] = 100;
  assert.deepEqual(index.get('a'), new Float32Array([1, 0]));
  for (let i = 0; i < 70; i++) index.add(`row-${i}`, new Float32Array([i, 1]));
  index.add('a', new Float32Array([-1, 0]));
  assert.equal(index.dimension, 2); assert.equal(index.size, 71); assert.equal(index.physicalCount, 72);
  assert.equal(index.has('a'), true); assert.equal(index.has('missing'), false); assert.equal(index.get('missing'), undefined);
  assert.equal(index.remove('row-69'), true); assert.equal(index.remove('row-69'), false);
  assert.equal(index.get('row-69'), undefined);
  const packed = index.getPackedVectors();
  assert.equal(packed.count, index.physicalCount); assert.equal(packed.vectors.length, packed.count * 2);
  assert.equal(packed.active.length, packed.count); assert.equal(packed.ids.length, packed.count);
  assert.equal(packed.active[0], 0); assert.equal(packed.active[packed.count - 1], 1);
  for (const exact of [true, false]) {
    const results = await index.search(new Float32Array([1, 0]), { topK: 100, exact, efSearch: 128 });
    assert.equal(results.length, index.size); assert.equal(new Set(identifiers(results)).size, index.size);
    assert.equal(results.some(match => match.id === 'row-69'), false);
    assert.equal(results.find(match => match.id === 'a')?.score, -1);
  }
  for (let i = 0; i < 65; i++) index.remove(`row-${i}`);
  assert.equal(index.stats().needsRebuild, true);
  const before = await index.search(new Float32Array([1, 0]), { topK: 100, exact: true });
  await index.compact();
  assert.deepEqual(await index.search(new Float32Array([1, 0]), { topK: 100, exact: true }), before);
  assert.equal(index.stats().tombstones, 0); assert.equal(index.physicalCount, index.size); assert.equal(index.stats().capacity, index.size);
  for (const id of index.getPackedVectors().ids.slice()) index.remove(id);
  assert.deepEqual(await index.search(new Float32Array([1, 0]), { topK: 10 }), []);
  assert.equal(index.stats().needsRebuild, true);
  index.add('fresh', new Float32Array([2, 0]));
  assert.equal(index.physicalCount, 1);
  assert.deepEqual(await index.search(new Float32Array([1, 0]), { topK: 1 }), [{ id: 'fresh', score: 2 }]);
});

test('deleted high-level entry points route but never return; updates survive reload', async () => {
  const next = random(77);
  const index = new VectorIndex(12, 'cosine', config({ m: 8 }));
  for (let i = 0; i < 500; i++) index.add(`row-${i}`, normalizedRandom(12, next));
  const old = index.exportSnapshot();
  const entryId = old.ids[old.entryPoint];
  index.remove(entryId);
  const query = normalizedRandom(12, next);
  for (let i = 0; i < 150; i++) index.add('updated', normalizedRandom(12, next));
  const replacement = vector(Array.from({ length: 12 }, (_, i) => i + 1));
  index.add('updated', replacement);
  const restored = VectorIndex.fromSnapshot(index.exportSnapshot());
  const results = await restored.search(replacement, { topK: 10, efSearch: 500 });
  assert.equal(results[0].id, 'updated'); assert.equal(results.filter(match => match.id === 'updated').length, 1);
  assert.equal(results.some(match => match.id === entryId), false);
  assert.deepEqual(await restored.search(query, { topK: 20, exact: true }), await index.search(query, { topK: 20, exact: true }));
});

test('filter is exact even with efSearch=1 and selective distant matches', async () => {
  const index = new VectorIndex(2, 'euclidean', config({ m: 4, efSearch: 1 }));
  for (let i = 0; i < 200; i++) index.add(`row-${i}`, new Float32Array([i, 0]));
  let calls = 0;
  const results = await index.search(new Float32Array([0, 0]), {
    topK: 3, efSearch: 1, exact: false,
    filter: id => { calls++; return Number(id.slice(4)) >= 150; },
  });
  assert.deepEqual(identifiers(results), ['row-150', 'row-151', 'row-152']);
  assert.equal(calls, index.size);
  assert.deepEqual(await index.search(new Float32Array([0, 0]), { topK: 5, filter: () => false }), []);
  await assert.rejects(index.search(new Float32Array([0, 0]), { topK: 1, filter: () => { throw new Error('filter failure'); } }), /filter failure/);
  assert.equal((await index.search(new Float32Array([0, 0]), { topK: 1 }))[0].id, 'row-0');
});

test('bulk-kernel distances use physical rows and enforce meaningful score contracts', async () => {
  const index = new VectorIndex(2, 'dot', config());
  index.add('deleted', new Float32Array([100, 0]));
  index.add('a', new Float32Array([4, 0])); index.add('b', new Float32Array([-2, 0])); index.remove('deleted');
  assert.deepEqual(await index.search(new Float32Array([1, 0]), { topK: 2, scores: new Float32Array([-100, -4, 2]) }), [
    { id: 'a', score: 4 }, { id: 'b', score: -2 },
  ]);
  await assert.rejects(index.search(new Float32Array([1, 0]), { topK: 1, scores: new Float32Array(2) }));
  await assert.rejects(index.search(new Float32Array([1, 0]), { topK: 1, scores: new Float32Array([0, NaN, 2]) }));
  const l2 = new VectorIndex(2, 'euclidean', config());
  l2.add('a', new Float32Array([3, 4]));
  assert.deepEqual(await l2.search(new Float32Array([0, 0]), { topK: 1, exact: true, scores: new Float32Array([25]) }), [{ id: 'a', score: -5, distance: 5 }]);
  await assert.rejects(l2.search(new Float32Array([0, 0]), { topK: 1, scores: new Float32Array([-1]) }));
});

test('snapshot ownership, empty snapshots and deterministic continuation', async () => {
  const next = random(990);
  const index = new VectorIndex(16, 'cosine', config());
  for (let i = 0; i < 180; i++) index.add(`row-${i}`, normalizedRandom(16, next));
  index.remove('row-17'); index.add('row-23', normalizedRandom(16, next));
  const snapshot = index.exportSnapshot();
  const loaded = VectorIndex.fromSnapshot(snapshot);
  assert.deepEqual(loaded.exportSnapshot(), snapshot);
  snapshot.ids[0] = 'changed'; snapshot.vectors[0] = 100; snapshot.active[0] = 0; snapshot.neighbors[0][0].length = 0;
  assert.equal(index.has('row-0'), true); assert.equal(loaded.has('row-0'), true);
  assert.deepEqual(loaded.get('row-0'), index.get('row-0'));
  for (let i = 180; i < 220; i++) { const row = normalizedRandom(16, next); index.add(`row-${i}`, row); loaded.add(`row-${i}`, row); }
  assert.deepEqual(loaded.exportSnapshot(), index.exportSnapshot());
  const query = normalizedRandom(16, next);
  assert.deepEqual(await loaded.search(query, { topK: 20, efSearch: 128 }), await index.search(query, { topK: 20, efSearch: 128 }));
  for (const type of ['flat', 'hnsw'] as const) {
    const empty = new VectorIndex(2, 'dot', config({ type }));
    const restored = VectorIndex.fromSnapshot(empty.exportSnapshot());
    assert.equal(restored.size, 0); assert.deepEqual(await restored.search(new Float32Array([0, 0]), { topK: 1 }), []);
  }
});

test('snapshot rejects malformed scalar, matrix, active and graph shapes', () => {
  const next = random(42);
  const index = new VectorIndex(4, 'cosine', config({ m: 4 }));
  for (let i = 0; i < 30; i++) index.add(`row-${i}`, normalizedRandom(4, next));
  const corruptions: ((snapshot: IndexSnapshot) => void)[] = [
    snapshot => { snapshot.format = 999; },
    snapshot => { snapshot.dimension = 0; },
    snapshot => { snapshot.metric = 'bad' as Metric; },
    snapshot => { snapshot.options.m = 1; },
    snapshot => { snapshot.ids = new Array(1_000_001); },
    snapshot => { snapshot.vectors = snapshot.vectors.slice(1); },
    snapshot => { snapshot.vectors[0] = Infinity; },
    snapshot => { snapshot.vectors.fill(0, 0, 4); },
    snapshot => { snapshot.active = new Uint8Array(1); },
    snapshot => { snapshot.active[0] = 2; },
    snapshot => { snapshot.ids[0] = ''; },
    snapshot => { snapshot.ids[1] = snapshot.ids[0]; },
    snapshot => { snapshot.levels[0] = 33; },
    snapshot => { snapshot.neighbors.length--; },
    snapshot => { snapshot.neighbors[0] = []; },
    snapshot => { snapshot.neighbors[0][0] = [0]; },
    snapshot => { snapshot.neighbors[0][0] = [1, 1]; },
    snapshot => { snapshot.neighbors[0][0] = [snapshot.ids.length]; },
    snapshot => { snapshot.neighbors[0][0] = [0.5]; },
    snapshot => { snapshot.neighbors[0][0] = new Array(9).fill(1); },
    snapshot => { snapshot.entryPoint = -1; },
    snapshot => { snapshot.maxLevel++; },
    snapshot => { snapshot.randomState = 0; },
  ];
  for (const corrupt of corruptions) { const snapshot = index.exportSnapshot(); corrupt(snapshot); assert.throws(() => VectorIndex.fromSnapshot(snapshot)); }
  const low = index.exportSnapshot();
  const upperNode = low.levels.findIndex(level => level > 0);
  const bottomNode = low.levels.findIndex(level => level === 0);
  assert.ok(upperNode >= 0 && bottomNode >= 0);
  low.neighbors[upperNode][1] = [bottomNode];
  assert.throws(() => VectorIndex.fromSnapshot(low));
});

test('finite float32 prepared contract, extreme numbers and normalization boundary', async () => {
  const cosine = new VectorIndex(3, 'cosine', config());
  const large = prepareVector([1e300, -1e300, 1e-300], 'cosine');
  cosine.add('large', large);
  approximately((await cosine.search(large, { topK: 1, exact: true }))[0].score, 1);
  cosine.add('tiny', prepareVector([1e-300, 2e-300, 3e-300], 'cosine'));
  for (const input of [new Float32Array([NaN, 1, 0]), new Float32Array([Infinity, 1, 0]), new Float32Array([0, 0, 0]), new Float32Array([1, 0])]) {
    assert.throws(() => cosine.add('bad', input)); await assert.rejects(cosine.search(input, { topK: 1 }));
  }
  assert.throws(() => cosine.add('bad', [1, 0, 0] as unknown as Float32Array));
  assert.equal(cosine.size, 2);
  assert.throws(() => prepareVector([3.5e38], 'dot'));
  const l2 = new VectorIndex(1, 'euclidean', config());
  l2.add('large', new Float32Array([3e38]));
  const result = (await l2.search(new Float32Array([-3e38]), { topK: 1, exact: true }))[0];
  assert.ok(Number.isFinite(result.distance)); approximately(result.distance! / 6e38, 1);
  const rounded = new VectorIndex(2, 'cosine', config());
  const q = vector([1, 1]); rounded.add('self', q);
  assert.ok((await rounded.search(q, { topK: 1 }))[0].score <= 1);
});

test('constructor and search bounds reject malformed options without altering rows', async () => {
  for (const options of [config({ m: 1 }), config({ m: 65 }), config({ efConstruction: 15 }), config({ efSearch: 0 }), config({ annThreshold: -1 })]) assert.throws(() => new VectorIndex(2, 'cosine', options));
  assert.throws(() => new VectorIndex(0, 'cosine', config())); assert.throws(() => new VectorIndex(16385, 'cosine', config()));
  const index = new VectorIndex(2, 'cosine', config()); index.add('a', vector([1, 0]));
  for (const options of [{ topK: -1 }, { topK: 1.5 }, { topK: 1, efSearch: 0 }, { topK: 1, minScore: NaN }, { topK: 1, maxDistance: 1 }]) await assert.rejects(index.search(vector([1, 0]), options));
  assert.throws(() => index.add('', vector([1, 0]))); assert.equal(index.size, 1);
  const before = index.exportSnapshot();
  assert.throws(() => index.add('a', new Float32Array([NaN, 0])));
  assert.deepEqual(index.exportSnapshot(), before);
});

test('atomic rebuild rejects invalid source rows and cancellation without losing old graph', async () => {
  const index = new VectorIndex(8, 'cosine', config());
  const next = random(101);
  for (let i = 0; i < 200; i++) index.add(`row-${i}`, normalizedRandom(8, next));
  const before = index.exportSnapshot();
  await assert.rejects(index.rebuild([{ id: 'valid', vector: normalizedRandom(8, next) }, { id: 'bad', vector: new Float32Array(8) }]));
  assert.deepEqual(index.exportSnapshot(), before);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 0);
  try { await assert.rejects(index.compact({ signal: controller.signal }), { name: 'AbortError' }); } finally { clearTimeout(timer); }
  assert.deepEqual(index.exportSnapshot(), before);
  await index.rebuild([{ id: 'only', vector: normalizedRandom(8, next) }]);
  assert.equal(index.size, 1); assert.equal(index.has('only'), true); assert.equal(index.physicalCount, 1);
});

test('cooperative exact/ANN cancellation yields macrotasks and does not poison workspaces', async () => {
  const dimension = 64;
  const index = new VectorIndex(dimension, 'cosine', config({ m: 8, efConstruction: 64 }));
  const next = random(202);
  for (let i = 0; i < 2000; i++) index.add(`row-${i}`, normalizedRandom(dimension, next));
  const query = normalizedRandom(dimension, next);
  for (const exact of [true, false]) {
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(index.search(query, { topK: 10, exact, signal: controller.signal }), { name: 'AbortError' });
    const active = new AbortController();
    let ticked = false;
    const timer = setTimeout(() => { ticked = true; active.abort(); }, 0);
    try { await assert.rejects(index.search(query, { topK: 10, exact, efSearch: 2000, signal: active.signal }), { name: 'AbortError' }); } finally { clearTimeout(timer); }
    assert.equal(ticked, true);
  }
  const [a, b] = await Promise.all([index.search(query, { topK: 10, exact: true }), index.search(query, { topK: 10, efSearch: 2000 })]);
  assert.deepEqual(a, b);
});

test('cooperative searches and rebuilds reject revision mixing', async () => {
  const index = new VectorIndex(8, 'dot', config({ type: 'flat' }));
  for (let i = 0; i < 1000; i++) index.add(`row-${i}`, new Float32Array(8).fill(i));
  const signal = new AbortController().signal;
  const search = index.search(new Float32Array(8).fill(1), { topK: 10, exact: true, signal });
  index.remove('row-0');
  await assert.rejects(search, (error: unknown) => error instanceof SereneDBError && error.code === 'INDEX_CHANGED');
  const rebuild = index.compact({ signal });
  index.add('new', new Float32Array(8).fill(1));
  await assert.rejects(rebuild, (error: unknown) => error instanceof SereneDBError && error.code === 'INDEX_CHANGED');
  assert.equal(index.has('new'), true);
});

function mockKernel(options: { failure?: 'append' | 'distance' | 'between' | 'score'; cleanupThrows?: boolean } = {}) {
  const states: { rows: Float32Array[]; disposed: number; distanceCalls: number; betweenCalls: number }[] = [];
  const factory: VectorDistanceKernelFactory = (dimension, metric) => {
    const state = { rows: [] as Float32Array[], disposed: 0, distanceCalls: 0, betweenCalls: 0 };
    states.push(state);
    const rank = (a: Float32Array, b: Float32Array) => {
      let value = 0;
      for (let i = 0; i < dimension; i++) value += metric === 'euclidean' ? (a[i] - b[i]) ** 2 : a[i] * b[i];
      return metric === 'euclidean' ? value : -value;
    };
    return {
      append(row, values) { if (options.failure === 'append') throw new Error('append failure'); assert.equal(row, state.rows.length); state.rows.push(values.slice()); },
      distance(query, row) { state.distanceCalls++; return options.failure === 'distance' ? Infinity : rank(query, state.rows[row]); },
      between(a, b) { state.betweenCalls++; return options.failure === 'between' ? NaN : rank(state.rows[a], state.rows[b]); },
      async score(query) { return options.failure === 'score' ? new Float32Array(state.rows.length).fill(NaN) : Float32Array.from(state.rows, row => rank(query, row)); },
      dispose() { state.disposed++; state.rows = []; if (options.cleanupThrows) throw new Error('cleanup failure'); },
    } satisfies VectorDistanceKernel;
  };
  return { factory, states };
}

test('optional distance kernel uses physical rows, graph distances and bulk exact scores', async () => {
  for (const metric of ['cosine', 'dot', 'euclidean'] as const) {
    const mock = mockKernel();
    const native = new VectorIndex(12, metric, config({ m: 8 }), mock.factory);
    const baseline = new VectorIndex(12, metric, config({ m: 8 }));
    const next = random(456);
    for (let i = 0; i < 180; i++) {
      const row = metric === 'cosine' ? normalizedRandom(12, next) : vector(Array.from({ length: 12 }, () => next() * 2 - 1), metric);
      native.add(`row-${i}`, row); baseline.add(`row-${i}`, row);
    }
    native.remove('row-17'); baseline.remove('row-17');
    const query = metric === 'cosine' ? normalizedRandom(12, next) : new Float32Array(12).fill(0.25);
    const scores = await native.precomputeDistances(query);
    assert.equal(scores?.length, native.physicalCount);
    const expected = await baseline.search(query, { topK: 10, exact: true });
    assert.deepEqual(identifiers(await native.search(query, { topK: 10, exact: true, scores })), identifiers(expected));
    assert.deepEqual(identifiers(await native.search(query, { topK: 10, efSearch: 180 })), identifiers(expected));
    assert.ok(mock.states[0].distanceCalls > 0); assert.ok(mock.states[0].betweenCalls > 0);
    const reloaded = VectorIndex.fromSnapshot(native.exportSnapshot(), mock.factory);
    assert.equal(mock.states[1].rows.length, native.physicalCount);
    assert.deepEqual(identifiers(await reloaded.search(query, { topK: 10, exact: true })), identifiers(expected));
    native.dispose(); reloaded.dispose(); assert.equal(mock.states[0].disposed, 1); assert.equal(mock.states[1].disposed, 1);
  }
});

test('optional kernel failures disable acceleration without invalidating results or updates', async () => {
  for (const failure of ['append', 'distance', 'between', 'score'] as const) {
    const mock = mockKernel({ failure, cleanupThrows: true });
    const index = new VectorIndex(2, 'euclidean', config({ m: 4 }), mock.factory);
    for (let i = 0; i < 30; i++) index.add(`row-${i}`, new Float32Array([i, 0]));
    if (failure === 'score') assert.equal(await index.precomputeDistances(new Float32Array([0, 0])), undefined);
    assert.deepEqual(identifiers(await index.search(new Float32Array([0, 0]), { topK: 3, exact: true })), ['row-0', 'row-1', 'row-2']);
    index.add('row-0', new Float32Array([100, 0]));
    assert.equal((await index.search(new Float32Array([0, 0]), { topK: 1, exact: true }))[0].id, 'row-1');
    assert.equal(mock.states[0].disposed, 1);
    index.dispose(); assert.equal(mock.states[0].disposed, 1);
  }
  const index = new VectorIndex(2, 'dot', config(), () => { throw new Error('unavailable'); });
  index.add('a', new Float32Array([2, 0]));
  assert.deepEqual(await index.search(new Float32Array([1, 0]), { topK: 1 }), [{ id: 'a', score: 2 }]);
});

test('kernel rebuild cancellation releases temporary stores and successful rebuild transfers ownership', async () => {
  const mock = mockKernel();
  const index = new VectorIndex(8, 'cosine', config(), mock.factory);
  const next = random(808);
  for (let i = 0; i < 300; i++) index.add(`row-${i}`, normalizedRandom(8, next));
  const before = index.exportSnapshot();
  const controller = new AbortController();
  const canceled = index.compact({ signal: controller.signal }); controller.abort();
  await assert.rejects(canceled, { name: 'AbortError' });
  assert.equal(mock.states[0].disposed, 0); assert.equal(mock.states[1].disposed, 1); assert.deepEqual(index.exportSnapshot(), before);
  await assert.rejects(index.rebuild([{ id: 'bad', vector: new Float32Array(8) }]));
  assert.equal(mock.states[2].disposed, 1); assert.equal(mock.states[0].disposed, 0);
  index.remove('row-1'); await index.compact();
  assert.equal(mock.states[0].disposed, 1); assert.equal(mock.states[3].disposed, 0);
  assert.equal(mock.states[3].rows.length, index.size);
  index.dispose(); index.dispose(); assert.equal(mock.states[3].disposed, 1);
  assert.throws(() => index.add('a', normalizedRandom(8, next)), (error: unknown) => error instanceof SereneDBError && error.code === 'INDEX_DISPOSED');
  await assert.rejects(index.search(normalizedRandom(8, next), { topK: 1 }), (error: unknown) => error instanceof SereneDBError && error.code === 'INDEX_DISPOSED');
  assert.throws(() => index.exportSnapshot());
});

test('empty-index reuse resets native addressing and invalid snapshots never allocate kernels', () => {
  const mock = mockKernel();
  const index = new VectorIndex(2, 'dot', config(), mock.factory);
  index.add('a', new Float32Array([1, 0])); index.remove('a'); index.add('b', new Float32Array([2, 0]));
  assert.equal(mock.states[0].disposed, 1); assert.equal(mock.states[1].rows.length, 1);
  assert.equal(index.physicalCount, 1);
  const snapshot = index.exportSnapshot(); snapshot.vectors[0] = NaN;
  assert.throws(() => VectorIndex.fromSnapshot(snapshot, mock.factory)); assert.equal(mock.states.length, 2);
  index.dispose();
});

test('bulk score calls serialize reusable native scratch buffers and cancellation rejects promptly', async () => {
  let release!: () => void;
  let entered!: () => void;
  const started = new Promise<void>(resolve => { entered = resolve; });
  const held = new Promise<void>(resolve => { release = resolve; });
  let calls = 0;
  let concurrent = 0;
  let maximum = 0;
  const mock = mockKernel();
  const factory: VectorDistanceKernelFactory = (dimension, metric) => {
    const kernel = mock.factory(dimension, metric);
    return {
      ...kernel,
      async score(query) {
        calls++; concurrent++; maximum = Math.max(maximum, concurrent);
        if (calls === 1) { entered(); await held; }
        try { return await kernel.score!(query); } finally { concurrent--; }
      },
    };
  };
  const index = new VectorIndex(2, 'dot', config(), factory);
  index.add('a', new Float32Array([2, 0])); index.add('b', new Float32Array([0, 3]));
  const controller = new AbortController();
  const first = index.precomputeDistances(new Float32Array([1, 0]), controller.signal);
  await started;
  const distanceCalls = mock.states[0].distanceCalls;
  assert.deepEqual(await index.search(new Float32Array([0, 1]), { topK: 1, exact: true }), [{ id: 'b', score: 3 }]);
  assert.equal(mock.states[0].distanceCalls, distanceCalls);
  const second = index.precomputeDistances(new Float32Array([0, 1]));
  controller.abort();
  await assert.rejects(first, { name: 'AbortError' });
  assert.equal(calls, 1);
  release();
  assert.deepEqual(await second, new Float32Array([-0, -3]));
  assert.equal(maximum, 1); assert.equal(calls, 2);
  assert.equal(mock.states[0].disposed, 0);
  index.dispose();
});

test('disposing an index rejects an in-flight bulk score even if kernel ignores cancellation', async () => {
  let release!: () => void;
  let entered!: () => void;
  const held = new Promise<void>(resolve => { release = resolve; });
  const started = new Promise<void>(resolve => { entered = resolve; });
  const mock = mockKernel();
  const factory: VectorDistanceKernelFactory = (dimension, metric) => {
    const kernel = mock.factory(dimension, metric);
    return { ...kernel, async score() { entered(); await held; return new Float32Array(1); } };
  };
  const index = new VectorIndex(2, 'dot', config(), factory); index.add('a', new Float32Array([1, 0]));
  const scoring = index.precomputeDistances(new Float32Array([1, 0]));
  await started; index.dispose();
  await assert.rejects(scoring, { name: 'AbortError' });
  assert.equal(mock.states[0].disposed, 1);
  release();
  await new Promise<void>(resolve => setTimeout(resolve, 0));
});

test('input and query typed views own only their row, including shared buffers', async () => {
  const index = new VectorIndex(2, 'dot', config({ type: 'flat' }));
  const shared = new Float32Array(new SharedArrayBuffer(8)); shared.set([1, 0]);
  index.add('a', shared); shared.set([0, 1]);
  assert.deepEqual(index.get('a'), new Float32Array([1, 0]));
  for (let i = 0; i < 300; i++) index.add(`row-${i}`, new Float32Array([-1, 0]));
  shared.set([1, 0]);
  const searching = index.search(shared, { topK: 1, exact: true, signal: new AbortController().signal });
  shared.set([0, 1]);
  assert.deepEqual(await searching, [{ id: 'a', score: 1 }]);
  const snapshot = index.exportSnapshot();
  snapshot.vectors = new Float32Array(new SharedArrayBuffer(snapshot.vectors.byteLength));
  assert.throws(() => VectorIndex.fromSnapshot(snapshot));
});

test('deterministic multilayer HNSW degree bounds and seeded clustered recall', async t => {
  const dimension = 64;
  const next = random(303);
  const centers = Array.from({ length: 16 }, () => normalizedRandom(dimension, next));
  const index = new VectorIndex(dimension, 'cosine', config({ m: 16, efConstruction: 128, efSearch: 128 }));
  for (let i = 0; i < 3000; i++) index.add(`row-${String(i).padStart(4, '0')}`, normalizedRandom(dimension, next, centers[i % centers.length], 0.08));
  const snapshot = index.exportSnapshot();
  assert.ok(snapshot.maxLevel >= 2); assert.ok(snapshot.levels.some(level => level > 0));
  for (const layers of snapshot.neighbors) for (let level = 0; level < layers.length; level++) assert.ok(layers[level].length <= (level === 0 ? 32 : 16));
  let hits = 0;
  for (let i = 0; i < 24; i++) {
    const query = normalizedRandom(dimension, next, centers[i % centers.length], 0.08);
    const truth = new Set(identifiers(await index.search(query, { topK: 10, exact: true })));
    const result = await index.search(query, { topK: 10, efSearch: 128 });
    hits += result.filter(match => truth.has(match.id)).length;
  }
  const recall = hits / 240;
  t.diagnostic(`clustered N=3000,D=64,M=16,efSearch=128 recall@10=${recall.toFixed(4)}`);
  assert.ok(recall >= 0.95, `Clustered recall ${recall} < 0.95`);
});

test('HNSW seeded random 10k x 384 recall and cooperative search latency', { timeout: 240000, skip: process.env.SERENEDB_DENSE_TEST !== '1' }, async t => {
  const dimension = 384;
  const next = random(0xabc123);
  const index = new VectorIndex(dimension, 'cosine', config({ m: 16, efConstruction: 128, efSearch: 2048 }));
  const start = performance.now();
  const records = Array.from({ length: 10000 }, (_, i) => ({ id: `row-${String(i).padStart(5, '0')}`, vector: normalizedRandom(dimension, next) }));
  await index.rebuild(records);
  const buildMs = performance.now() - start;
  let hits = 0;
  const beams = [128, 256, 512, 1024, 2048];
  const beamHits = new Map(beams.map(ef => [ef, 0]));
  const beamMs = new Map(beams.map(ef => [ef, 0]));
  let exactMs = 0;
  let annMs = 0;
  for (let i = 0; i < 12; i++) {
    const query = normalizedRandom(dimension, next);
    let startQuery = performance.now();
    const truth = new Set(identifiers(await index.search(query, { topK: 10, exact: true })));
    exactMs += performance.now() - startQuery;
    for (const ef of beams) {
      startQuery = performance.now();
      const result = await index.search(query, { topK: 10, efSearch: ef });
      const elapsed = performance.now() - startQuery;
      const found = result.filter(match => truth.has(match.id)).length;
      beamHits.set(ef, beamHits.get(ef)! + found); beamMs.set(ef, beamMs.get(ef)! + elapsed);
      if (ef === 2048) { annMs += elapsed; hits += found; }
    }
  }
  const recall = hits / 120;
  t.diagnostic(`random N=10000,D=384,M=16,efConstruction=128,efSearch=2048 recall@10=${recall.toFixed(4)} build=${buildMs.toFixed(0)}ms exact=${(exactMs / 12).toFixed(2)}ms ANN=${(annMs / 12).toFixed(2)}ms`);
  for (const ef of beams) t.diagnostic(`efSearch=${ef} recall@10=${(beamHits.get(ef)! / 120).toFixed(4)} ANN=${(beamMs.get(ef)! / 12).toFixed(2)}ms`);
  assert.ok(recall >= 0.95, `Random recall ${recall} < 0.95`);
  assert.ok(beamHits.get(512)! / 120 >= 0.95, 'efSearch=512 must reach the seeded high-dimensional recall target.');
  assert.equal(index.stats().tombstones, 0);
  assert.equal(index.stats().capacity, index.size);
});
