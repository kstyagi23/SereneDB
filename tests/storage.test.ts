import 'fake-indexeddb/auto';
import { forceCloseDatabase, IDBDatabase, IDBObjectStore } from 'fake-indexeddb';
import assert from 'node:assert/strict';
import { test, type TestContext } from 'node:test';
import { Storage } from '../src/storage.js';
import type { IndexSnapshot, OpenOptions, PreparedRecord } from '../src/types.js';
import { collectionSchema } from '../src/validation.js';

let sequence = 0;
const schema = () => collectionSchema({ metric: 'dot', index: 'flat' });
const record = (id: string, vector = [1, 0]): PreparedRecord => ({ id, vector: new Float32Array(vector), metadata: {} });
const code = (expected: string) => (error: unknown) => !!error && typeof error === 'object' && 'code' in error && error.code === expected;

async function storage(t: TestContext, options: OpenOptions = {}): Promise<Storage> {
  const db = await Storage.open(`storage-test-${++sequence}`, options);
  t.after(() => db.close());
  return db;
}

function rawOpen(name: string, version?: number, upgrade?: (db: IDBDatabase) => void): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = version === undefined ? indexedDB.open(name) : indexedDB.open(name, version);
    request.onupgradeneeded = () => upgrade?.(request.result);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

function rawWrite(db: IDBDatabase, stores: string[], write: (tx: IDBTransaction) => void): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(stores, 'readwrite');
    tx.oncomplete = () => resolve();
    tx.onabort = () => reject(tx.error);
    try { write(tx); } catch (error) { tx.abort(); reject(error); }
  });
}

function rawRead(db: IDBDatabase, store: string, key: IDBValidKey): Promise<any> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const request = tx.objectStore(store).get(key);
    let value: unknown;
    request.onsuccess = () => { value = request.result; };
    tx.oncomplete = () => resolve(value);
    tx.onabort = () => reject(tx.error);
  });
}

function snapshot(): IndexSnapshot {
  return {
    format: 1, dimension: 2, metric: 'dot', options: schema().index,
    ids: ['a'], vectors: new Float32Array([1, 0]), active: new Uint8Array([1]),
    levels: [0], neighbors: [[[]]], entryPoint: 0, maxLevel: 0, randomState: 1,
  };
}

test('version 2 stores and a cold restart are independent of localStorage and caller-owned values', async t => {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get: () => { throw new Error('localStorage must not be consulted'); } });
  try {
    const db = await storage(t);
    const state = await db.ensureCollection('docs', schema());
    assert.equal(state.schema.dimension, null);
    const input = record('a', [1, 2, 3]);
    input.metadata = { source: 'original', nested: { value: 1 } };
    input.text = 'document';
    input.timestamp = 0;
    const pending = db.mutate('docs', { upserts: [input] }, state.generation);
    input.vector[0] = 99;
    input.metadata.source = 'changed';
    const result = await pending;
    result.upserted[0].vector[0] = 88;
    result.upserted[0].metadata.source = 'returned value changed';
    result.state.schema.metric = 'cosine';
    const rootGeneration = db.generation;
    db.close();
    const reopened = await Storage.open(db.name);
    t.after(() => reopened.close());
    assert.equal(reopened.generation, rootGeneration);
    const loaded = await reopened.readCollection('docs');
    assert.equal(loaded.state.generation, state.generation);
    assert.equal(loaded.state.schema.metric, 'dot');
    assert.equal(loaded.state.schema.index.type, 'flat');
    assert.equal(loaded.state.schema.dimension, 3);
    assert.equal(loaded.state.count, 1);
    assert.equal(loaded.state.revision, 1);
    assert.deepEqual(Array.from(loaded.records[0].vector), [1, 2, 3]);
    assert.equal(loaded.records[0].metadata.source, 'original');
    assert.equal(loaded.records[0].text, 'document');
    assert.equal(loaded.records[0].timestamp, 0);
    loaded.records[0].vector[1] = 44;
    assert.deepEqual(Array.from((await reopened.get('docs', 'a'))!.vector), [1, 2, 3]);
    const raw = await rawOpen(db.name);
    t.after(() => raw.close());
    assert.equal(raw.version, 2);
    assert.deepEqual(Array.from(raw.objectStoreNames), ['collections', 'indexes', 'records', 'system']);
    const tx = raw.transaction('records');
    assert.deepEqual(tx.objectStore('records').keyPath, ['collection', 'id']);
    assert.equal(tx.objectStore('records').index('collection').keyPath, 'collection');
    assert.equal((await rawRead(raw, 'system', 'root')).status, 'active');
  } finally {
    if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);
    else Reflect.deleteProperty(globalThis, 'localStorage');
  }
});

test('explicit schema compatibility uses persisted state and atomically pins a requested dimension', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  assert.equal((await db.ensureCollection('docs', collectionSchema(), { explicit: {} })).schema.metric, 'dot');
  await assert.rejects(db.ensureCollection('docs', collectionSchema()), code('SCHEMA_MISMATCH'));
  await assert.rejects(db.ensureCollection('docs', collectionSchema(), { explicit: { metric: true } }), code('SCHEMA_MISMATCH'));
  await assert.rejects(db.ensureCollection('docs', collectionSchema(), { explicit: { index: true } }), code('SCHEMA_MISMATCH'));
  const pinned = await db.ensureCollection('docs', collectionSchema({ dimension: 2, metric: 'dot', index: 'flat' }), { explicit: { dimension: true } });
  assert.equal(pinned.schema.dimension, 2);
  assert.equal(pinned.revision, 1);
  assert.equal((await db.ensureCollection('docs', schema())).schema.dimension, 2);
  await assert.rejects(db.ensureCollection('docs', collectionSchema({ dimension: 3, metric: 'dot', index: 'flat' })), code('SCHEMA_MISMATCH'));
  await assert.rejects(db.mutate('docs', { upserts: [record('a', [1, 2, 3])] }), code('SCHEMA_MISMATCH'));
  assert.equal((await db.refreshState('docs')).count, 0);
});

test('synchronous enqueue DataCloneError rolls back earlier puts, count, revision, and inferred dimension', async t => {
  const db = await storage(t);
  const state = await db.ensureCollection('docs', schema());
  const original = IDBObjectStore.prototype.put;
  const enqueued: string[] = [];
  IDBObjectStore.prototype.put = function (value: any, key?: IDBValidKey): IDBRequest<IDBValidKey> {
    if (this.name === 'records') {
      if (value.id === 'bad') throw new DOMException('Uncloneable second write', 'DataCloneError');
      enqueued.push(value.id);
    }
    return Reflect.apply(original, this, key === undefined ? [value] : [value, key]);
  };
  try {
    await assert.rejects(db.mutate('docs', { upserts: [record('first'), record('bad')] }, state.generation), { name: 'DataCloneError' });
  } finally { IDBObjectStore.prototype.put = original; }
  assert.deepEqual(enqueued, ['first']);
  const after = await db.readCollection('docs');
  assert.equal(after.state.count, 0);
  assert.equal(after.state.revision, 0);
  assert.equal(after.state.schema.dimension, null);
  assert.deepEqual(after.records, []);
  const valid = await db.mutate('docs', { upserts: [record('valid', [1, 2, 3])] }, state.generation);
  assert.equal(valid.state.schema.dimension, 3);
});

test('mixed dimensions and uncloneable metadata never establish an empty collection dimension', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  await assert.rejects(db.mutate('docs', { upserts: [record('a'), record('b', [1, 2, 3])] }), code('SCHEMA_MISMATCH'));
  const bad = record('bad');
  bad.metadata = { callback: (() => undefined) as any };
  await assert.rejects(db.mutate('docs', { upserts: [record('a'), bad] }));
  const state = await db.refreshState('docs');
  assert.equal(state.schema.dimension, null);
  assert.equal(state.count, 0);
  assert.equal(state.revision, 0);
});

test('mutations acknowledge transaction completion, not request success', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const original = IDBObjectStore.prototype.put;
  let requestSucceeded = false;
  let committed = false;
  IDBObjectStore.prototype.put = function (value: any, key?: IDBValidKey): IDBRequest<IDBValidKey> {
    const request = Reflect.apply(original, this, key === undefined ? [value] : [value, key]);
    if (this.name === 'records') {
      this.transaction.addEventListener('complete', () => { committed = true; });
      request.addEventListener('success', () => { requestSucceeded = true; assert.equal(committed, false); });
    }
    return request;
  };
  try {
    await db.mutate('docs', { upserts: [record('a')] });
    assert.equal(requestSucceeded, true);
    assert.equal(committed, true);
  } finally { IDBObjectStore.prototype.put = original; }
});

test('cancellation at the transaction completion boundary cannot reject an already committed mutation', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const original = IDBDatabase.prototype.transaction;
  const controller = new AbortController();
  IDBDatabase.prototype.transaction = function (...args: any[]): IDBTransaction {
    const tx = Reflect.apply(original, this, args);
    if (args[1] === 'readwrite') tx.addEventListener('complete', () => { controller.abort(); db.close(); }, { once: true });
    return tx;
  };
  let result;
  try { result = await db.mutate('docs', { upserts: [record('a')] }, undefined, controller.signal); }
  finally { IDBDatabase.prototype.transaction = original; }
  assert.equal(controller.signal.aborted, true);
  assert.equal(result.state.count, 1);
  const reopened = await Storage.open(db.name);
  t.after(() => reopened.close());
  assert.equal((await reopened.refreshState('docs')).count, 1);
  assert.ok(await reopened.get('docs', 'a'));
});

test('failed IDB requests abort even when a request listener suppresses the default abort', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const original = IDBObjectStore.prototype.put;
  let failed = false;
  IDBObjectStore.prototype.put = function (value: any, key?: IDBValidKey): IDBRequest<IDBValidKey> {
    if (this.name === 'records' && value.id === 'bad') {
      const request = this.add({ ...value, id: 'first' });
      request.addEventListener('error', event => { failed = true; event.preventDefault(); });
      return request;
    }
    return Reflect.apply(original, this, key === undefined ? [value] : [value, key]);
  };
  try { await assert.rejects(db.mutate('docs', { upserts: [record('first'), record('bad')] }), { name: 'ConstraintError' }); }
  finally { IDBObjectStore.prototype.put = original; }
  assert.equal(failed, true);
  assert.deepEqual((await db.readCollection('docs')).records, []);
  assert.equal((await db.refreshState('docs')).schema.dimension, null);
});

test('AbortSignal aborts an active transaction even after a record request succeeds', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const original = IDBObjectStore.prototype.put;
  const controller = new AbortController();
  IDBObjectStore.prototype.put = function (value: any, key?: IDBValidKey): IDBRequest<IDBValidKey> {
    const request = Reflect.apply(original, this, key === undefined ? [value] : [value, key]);
    if (this.name === 'records') request.addEventListener('success', () => controller.abort(), { once: true });
    return request;
  };
  try {
    await assert.rejects(db.mutate('docs', { upserts: [record('a'), record('b')] }, undefined, controller.signal), { name: 'AbortError' });
  } finally { IDBObjectStore.prototype.put = original; }
  const after = await db.readCollection('docs');
  assert.equal(after.state.count, 0);
  assert.equal(after.state.revision, 0);
  assert.equal(after.state.schema.dimension, null);
  assert.deepEqual(after.records, []);
  await assert.rejects(db.mutate('docs', { upserts: [record('c')] }, undefined, controller.signal), { name: 'AbortError' });
});

test('closing a connection aborts its active mutation and never writes back returned data', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const pending = db.mutate('docs', { upserts: [record('a')] });
  db.close();
  await assert.rejects(pending, code('DATABASE_CLOSED'));
  const reopened = await Storage.open(db.name);
  t.after(() => reopened.close());
  assert.equal((await reopened.refreshState('docs')).count, 0);
  assert.equal((await reopened.refreshState('docs')).schema.dimension, null);
});

test('concurrent connections cannot commit conflicting inferred dimensions', async t => {
  const first = await storage(t);
  const state = await first.ensureCollection('docs', schema());
  const second = await Storage.open(first.name);
  t.after(() => second.close());
  const results = await Promise.allSettled([
    first.mutate('docs', { upserts: [record('two')] }, state.generation),
    second.mutate('docs', { upserts: [record('three', [1, 2, 3])] }, state.generation),
  ]);
  assert.equal(results.filter(result => result.status === 'fulfilled').length, 1);
  const rejection = results.find(result => result.status === 'rejected') as PromiseRejectedResult;
  assert.equal(rejection.reason.code, 'SCHEMA_MISMATCH');
  const read = await second.readCollection('docs');
  assert.equal(read.records.length, 1);
  assert.equal(read.state.count, 1);
  assert.equal(read.state.revision, 1);
  assert.equal(read.records[0].vector.length, read.state.schema.dimension);
  assert.equal((await first.refreshState('docs')).schema.dimension, read.state.schema.dimension);
});

test('duplicate IDs, deletes, upserts, and clears maintain exact counts with last-write-wins semantics', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  let result = await db.mutate('docs', { upserts: [record('a'), record('b'), record('a', [0, 1])] });
  assert.equal(result.state.count, 2);
  assert.equal(result.upserted.length, 2);
  assert.equal(result.state.revision, 1);
  assert.deepEqual(Array.from((await db.get('docs', 'a'))!.vector), [0, 1]);
  result = await db.mutate('docs', { deleteIds: ['a', 'a', 'missing'], upserts: [record('a')] });
  assert.equal(result.deleted, 1);
  assert.equal(result.state.count, 2);
  assert.equal(result.state.revision, 2);
  result = await db.mutate('docs', { deleteIds: ['b', 'b', 'missing'] });
  assert.equal(result.deleted, 1);
  assert.equal(result.state.count, 1);
  assert.equal(result.state.revision, 3);
  result = await db.mutate('docs', { deleteIds: ['b', 'missing'] });
  assert.equal(result.deleted, 0);
  assert.equal(result.state.revision, 3);
  result = await db.mutate('docs', { clear: true, upserts: [record('c'), record('c'), record('d')] });
  assert.equal(result.deleted, 1);
  assert.equal(result.state.count, 2);
  assert.equal(result.state.revision, 4);
  assert.equal(await db.get('docs', 'a'), undefined);
  result = await db.mutate('docs', { clear: true });
  assert.equal(result.deleted, 2);
  assert.equal(result.state.count, 0);
  assert.equal(result.state.revision, 5);
  assert.equal((await db.mutate('docs', { clear: true })).state.revision, 5);
  assert.equal((await db.refreshState('docs')).schema.dimension, 2);
});

test('clear and drop ranges isolate collections including prefix names and arbitrary Unicode IDs', async t => {
  const db = await storage(t);
  const first = await db.ensureCollection('a', schema());
  await db.ensureCollection('aa', schema());
  await db.mutate('a', { upserts: [record('id\uffff'), record('id\udbff\udfff')] });
  await db.mutate('aa', { upserts: [record('other')] });
  await db.mutate('a', { clear: true });
  assert.equal((await db.readCollection('a')).records.length, 0);
  assert.equal((await db.readCollection('aa')).records.length, 1);
  await db.mutate('a', { upserts: [record('again')] });
  await db.dropCollection('a', first.generation);
  await assert.rejects(db.refreshState('a'), code('COLLECTION_NOT_FOUND'));
  assert.equal((await db.readCollection('aa')).records.length, 1);
  await db.dropCollection('missing');
});

test('dropped/recreated collections reject stale mutations and stale drops', async t => {
  const db = await storage(t);
  const first = await db.ensureCollection('docs', schema());
  await db.dropCollection('docs', first.generation);
  const replacement = await db.ensureCollection('docs', schema());
  assert.notEqual(replacement.generation, first.generation);
  await assert.rejects(db.mutate('docs', { upserts: [record('stale')] }, first.generation), code('COLLECTION_STALE'));
  await assert.rejects(db.dropCollection('docs', first.generation), code('COLLECTION_STALE'));
  assert.equal((await db.refreshState('docs')).count, 0);
  assert.equal(await db.checkpoint('docs', first.generation, first.revision, snapshot()), false);
});

test('checkpoints require the same collection generation/revision and are invalidated atomically by writes', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const mutation = await db.mutate('docs', { upserts: [record('a')] });
  const payload = snapshot();
  const pending = db.checkpoint('docs', mutation.state.generation, mutation.state.revision, payload);
  payload.vectors[0] = 99;
  assert.equal(await pending, true);
  const read = await db.readCollection('docs');
  assert.equal(read.snapshot?.revision, read.state.revision);
  assert.equal(read.snapshot?.payload.vectors[0], 1);
  assert.equal(await db.checkpoint('docs', mutation.state.generation, mutation.state.revision - 1, snapshot()), false);
  const next = await db.mutate('docs', { upserts: [record('b')] });
  assert.equal((await db.readCollection('docs')).snapshot, undefined);
  assert.equal(await db.checkpoint('docs', mutation.state.generation, mutation.state.revision, snapshot()), false);
  const wrong = snapshot();
  wrong.dimension = 3;
  await assert.rejects(db.checkpoint('docs', next.state.generation, next.state.revision, wrong), code('SCHEMA_MISMATCH'));
});

test('readCollection returns a consistent readonly state/records/checkpoint snapshot across writers', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const mutation = await db.mutate('docs', { upserts: [record('a')] });
  await db.checkpoint('docs', mutation.state.generation, mutation.state.revision, snapshot());
  const peer = await Storage.open(db.name);
  t.after(() => peer.close());
  const [read, write] = await Promise.all([db.readCollection('docs'), peer.mutate('docs', { upserts: [record('b')] })]);
  assert.equal(read.records.length, read.state.count);
  assert.equal(read.state.count, 1);
  assert.equal(read.snapshot?.revision, read.state.revision);
  assert.equal(write.state.count, 2);
  assert.equal((await db.readCollection('docs')).snapshot, undefined);
});

test('every transaction gates root identity and closes stale connections', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const raw = await rawOpen(db.name);
  t.after(() => raw.close());
  const root = await rawRead(raw, 'system', 'root');
  await rawWrite(raw, ['system'], tx => tx.objectStore('system').put({ ...root, generation: crypto.randomUUID() }));
  await assert.rejects(db.listCollections(), code('DATABASE_STALE'));
  await assert.rejects(db.mutate('docs', { upserts: [record('a')] }), code('DATABASE_STALE'));
});

test('a missing authoritative root is an error, never an automatic recovery or reset', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  const raw = await rawOpen(db.name);
  t.after(() => raw.close());
  await rawWrite(raw, ['system'], tx => tx.objectStore('system').delete('root'));
  await assert.rejects(db.refreshState('docs'), code('STORAGE_CORRUPT'));
  await assert.rejects(Storage.open(db.name), code('STORAGE_CORRUPT'));
  assert.equal((await rawRead(raw, 'collections', 'docs')).count, 0);
});

test('versionchange closes peer Storage connections so destruction is not indefinitely blocked', async t => {
  const first = await storage(t);
  await first.ensureCollection('docs', schema());
  await first.mutate('docs', { upserts: [record('a')] });
  const peer = await Storage.open(first.name);
  t.after(() => peer.close());
  const previousGeneration = first.generation;
  await first.destroy();
  await assert.rejects(peer.listCollections(), code('DATABASE_CLOSED'));
  const replacement = await Storage.open(first.name);
  t.after(() => replacement.close());
  assert.notEqual(replacement.generation, previousGeneration);
  assert.deepEqual(await replacement.listCollections(), []);
  await assert.rejects(first.ensureCollection('docs', schema()), code('DATABASE_CLOSED'));
});

test('blocked destruction times out only after an atomic terminal reset, and eventual deletion is safe', async t => {
  const db = await storage(t, { blockedTimeoutMs: 20 });
  await db.ensureCollection('docs', schema());
  await db.mutate('docs', { upserts: [record('a')] });
  const raw = await rawOpen(db.name);
  t.after(() => raw.close());
  const previousGeneration = db.generation;
  try {
    await assert.rejects(db.destroy(), code('STORAGE_BLOCKED'));
    await assert.rejects(db.listCollections(), code('DATABASE_CLOSED'));
    const root = await rawRead(raw, 'system', 'root');
    assert.equal(root.status, 'destroying');
    assert.notEqual(root.generation, previousGeneration);
    assert.equal(await rawRead(raw, 'collections', 'docs'), undefined);
    assert.equal(await rawRead(raw, 'records', ['docs', 'a']), undefined);
  } finally { raw.close(); }
  const fresh = await Storage.open(db.name);
  t.after(() => fresh.close());
  assert.notEqual(fresh.generation, previousGeneration);
  assert.deepEqual(await fresh.listCollections(), []);
});

test('aborting the destructive transaction rolls back the root and all data and leaves the connection usable', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  await db.mutate('docs', { upserts: [record('a')] });
  const original = IDBObjectStore.prototype.put;
  const controller = new AbortController();
  const generation = db.generation;
  IDBObjectStore.prototype.put = function (value: any, key?: IDBValidKey): IDBRequest<IDBValidKey> {
    const request = Reflect.apply(original, this, key === undefined ? [value] : [value, key]);
    if (this.name === 'system' && value.status === 'destroying') request.addEventListener('success', () => controller.abort(), { once: true });
    return request;
  };
  try { await assert.rejects(db.destroy(controller.signal), { name: 'AbortError' }); }
  finally { IDBObjectStore.prototype.put = original; }
  assert.equal(db.generation, generation);
  assert.equal((await db.readCollection('docs')).state.count, 1);
  assert.ok(await db.get('docs', 'a'));
  await assert.rejects(db.destroy(controller.signal), { name: 'AbortError' });
  assert.equal((await db.refreshState('docs')).count, 1);
});

test('abort after the terminal reset stops waiting but cannot cancel pending physical deletion', async t => {
  const db = await storage(t);
  await db.ensureCollection('docs', schema());
  await db.mutate('docs', { upserts: [record('a')] });
  const raw = await rawOpen(db.name);
  t.after(() => raw.close());
  const controller = new AbortController();
  raw.onversionchange = () => controller.abort();
  try {
    await assert.rejects(db.destroy(controller.signal), { name: 'AbortError' });
    assert.equal((await rawRead(raw, 'system', 'root')).status, 'destroying');
    await assert.rejects(db.listCollections(), code('DATABASE_CLOSED'));
  } finally { raw.close(); }
  const fresh = await Storage.open(db.name);
  t.after(() => fresh.close());
  assert.deepEqual(await fresh.listCollections(), []);
});

test('unexpected IndexedDB close is reported explicitly', async t => {
  const name = `storage-test-${++sequence}`;
  const original = IDBDatabase.prototype.transaction;
  let connection: IDBDatabase | undefined;
  IDBDatabase.prototype.transaction = function (...args: any[]): IDBTransaction {
    if (this.name === name) connection = this;
    return Reflect.apply(original, this, args);
  };
  let db: Storage;
  try { db = await Storage.open(name); }
  finally { IDBDatabase.prototype.transaction = original; }
  t.after(() => db.close());
  assert.ok(connection);
  const closed = new Promise<void>(resolve => connection!.addEventListener('close', () => resolve(), { once: true }));
  forceCloseDatabase(connection!);
  await closed;
  await assert.rejects(db.listCollections(), error => code('DATABASE_CLOSED')(error) && /unexpectedly/.test((error as Error).message));
});

test('strict durability falls back only when transaction options are unsupported', async t => {
  const db = await storage(t);
  const original = IDBDatabase.prototype.transaction;
  let optionAttempts = 0;
  let plainAttempts = 0;
  IDBDatabase.prototype.transaction = function (...args: any[]): IDBTransaction {
    if (args[1] === 'readwrite') {
      if (args.length === 3) { optionAttempts += 1; assert.equal(args[2].durability, 'strict'); throw new TypeError('Transaction options unsupported'); }
      plainAttempts += 1;
    }
    return Reflect.apply(original, this, args);
  };
  try {
    await db.ensureCollection('docs', schema());
    await db.mutate('docs', { upserts: [record('a')] });
  } finally { IDBDatabase.prototype.transaction = original; }
  assert.equal(optionAttempts, 1);
  assert.equal(plainAttempts, 2);
  assert.equal((await db.refreshState('docs')).count, 1);
  const strict = await storage(t);
  let calls = 0;
  IDBDatabase.prototype.transaction = function (...args: any[]): IDBTransaction {
    if (args[1] === 'readwrite') { calls += 1; throw new DOMException('Quota failure', 'QuotaExceededError'); }
    return Reflect.apply(original, this, args);
  };
  try { await assert.rejects(strict.ensureCollection('docs', schema()), { name: 'QuotaExceededError' }); }
  finally { IDBDatabase.prototype.transaction = original; }
  assert.equal(calls, 1);
  assert.deepEqual(await strict.listCollections(), []);
});

test('legacy upgrades fail without deleting, converting, or changing original records', async t => {
  const name = `storage-test-${++sequence}`;
  const legacy = await rawOpen(name, 1, db => db.createObjectStore('vectors', { keyPath: 'id' }));
  await rawWrite(legacy, ['vectors'], tx => tx.objectStore('vectors').put({ id: 'original', vector: [1, 2] }));
  legacy.close();
  await assert.rejects(Storage.open(name), code('LEGACY_DATA'));
  const unchanged = await rawOpen(name);
  t.after(() => unchanged.close());
  assert.equal(unchanged.version, 1);
  assert.deepEqual(Array.from(unchanged.objectStoreNames), ['vectors']);
  assert.deepEqual(await rawRead(unchanged, 'vectors', 'original'), { id: 'original', vector: [1, 2] });
});

test('blocked opens are bounded and their later upgrade never creates replacement stores', async t => {
  const name = `storage-test-${++sequence}`;
  const legacy = await rawOpen(name, 1, db => db.createObjectStore('vectors', { keyPath: 'id' }));
  try { await assert.rejects(Storage.open(name, { blockedTimeoutMs: 20 }), code('STORAGE_BLOCKED')); }
  finally { legacy.close(); }
  const unchanged = await rawOpen(name);
  t.after(() => unchanged.close());
  assert.equal(unchanged.version, 1);
  assert.deepEqual(Array.from(unchanged.objectStoreNames), ['vectors']);
});
