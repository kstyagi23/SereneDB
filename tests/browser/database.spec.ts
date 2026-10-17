import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => { await page.goto('/test.html'); await page.waitForFunction(() => typeof window.SereneDB === 'function'); });

test('worker CRUD survives cold page reload and localStorage denial', async ({ page }) => {
  const name = `browser-${crypto.randomUUID()}`;
  await page.evaluate(async name => {
    const db = await window.SereneDB.open(name);
    const docs = await db.collection('docs');
    await docs.upsert([{ id: 'a', vector: new Float32Array([1, 0]), metadata: { category: 'db' } }, { id: 'b', vector: [0, 1] }]);
    await db.close();
    localStorage.clear();
  }, name);
  await page.reload();
  const result = await page.evaluate(async name => {
    Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('Denied', 'SecurityError'); }, configurable: true });
    const db = await window.SereneDB.open(name);
    const docs = await db.collection('docs');
    const response = await docs.search({ vector: [1, 0], mode: 'exact', includeVectors: true });
    const count = await docs.count();
    await db.destroy();
    return { count, ids: response.results.map(item => item.id), score: response.results[0].score };
  }, name);
  expect(result).toEqual({ count: 2, ids: ['a', 'b'], score: 1 });
});

test('two pages see committed writes and reject schema conflicts', async ({ page, context }) => {
  const name = `concurrent-${crypto.randomUUID()}`;
  const other = await context.newPage(); await other.goto('/test.html');
  await Promise.all([page, other].map(p => p.evaluate(async name => {
    window.db = await window.SereneDB.open(name); window.docs = await window.db.collection('docs');
  }, name)));
  await page.evaluate(() => window.docs.upsert({ id: 'a', vector: [1, 0] }));
  const rejected = await other.evaluate(async () => { try { await window.docs.upsert({ id: 'wrong', vector: [1, 0, 0] }); return false; } catch { return true; } });
  expect(rejected).toBe(true);
  await other.evaluate(() => window.docs.upsert({ id: 'b', vector: [0, 1] }));
  expect(await page.evaluate(() => window.docs.count())).toBe(2);
  await page.evaluate(() => window.docs.delete('b'));
  expect(await other.evaluate(() => window.docs.has('b'))).toBe(false);
  await other.evaluate(() => window.db.close()); await page.evaluate(() => window.db.destroy()); await other.close();
});

test('destroy waits for queued writes and invalidates open peers', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const name = `destroy-${crypto.randomUUID()}`;
    const first = await window.SereneDB.open(name), second = await window.SereneDB.open(name);
    const docs = await first.collection('docs'), peer = await second.collection('docs');
    const write = docs.upsert({ id: 'a', vector: [1, 0] });
    await Promise.all([write, first.destroy()]);
    let peerRejected = false;
    try { await peer.count(); } catch { peerRejected = true; }
    try { await second.close(); } catch {}
    const reopened = await window.SereneDB.open(name);
    const collections = await reopened.listCollections(); await reopened.destroy();
    return { peerRejected, collections };
  });
  expect(result).toEqual({ peerRejected: true, collections: [] });
});

test('aborted request and malformed batch leave durable state unchanged', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const db = await window.SereneDB.open(`abort-${crypto.randomUUID()}`);
    const docs = await db.collection('docs');
    let errors = 0;
    const controller = new AbortController(); controller.abort();
    try { await docs.upsert({ id: 'a', vector: [1, 0] }, { signal: controller.signal }); } catch { errors++; }
    try { await docs.upsert([{ id: 'a', vector: [1, 0, 0] }, { id: 'b', vector: [1, 0] }]); } catch { errors++; }
    const before = await docs.state();
    await docs.upsert({ id: 'valid', vector: [1, 0] });
    const after = await docs.state(); await db.destroy();
    return { errors, beforeCount: before.count, beforeDimension: before.schema.dimension, afterCount: after.count, afterDimension: after.schema.dimension };
  });
  expect(result).toEqual({ errors: 2, beforeCount: 0, beforeDimension: null, afterCount: 1, afterDimension: 2 });
});

test('filtered nearest neighbors, L2 cutoffs, and hybrid identifiers', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const db = await window.SereneDB.open(`metrics-${crypto.randomUUID()}`);
    const docs = await db.collection('docs');
    await docs.upsert([
      { id: 'a', vector: [1, 0], text: 'ticket ABC_123 vector database', metadata: { category: 'db' } },
      { id: 'b', vector: [0, 1], text: 'baking bread', metadata: { category: 'food' } },
      { id: 'c', vector: [-1, 0], metadata: { category: 'db' } },
    ]);
    const filtered = await docs.search({ vector: [1, 0], filter: { category: 'db' }, topK: 3 });
    const hybrid = await docs.search({ text: 'ABC_123', mode: 'hybrid' });
    const l2 = await db.collection('distance', { metric: 'euclidean' });
    await l2.upsert([{ id: 'exact', vector: [0, 0] }, { id: 'near', vector: [0.25, 0] }, { id: 'far', vector: [2, 0] }]);
    const nearby = await l2.search({ vector: [0, 0], maxDistance: 0.5 });
    await db.destroy();
    return { filtered: filtered.results.map(item => item.id), mode: filtered.mode, hybrid: hybrid.results[0].id, nearby: nearby.results.map(item => item.id) };
  });
  expect(result).toEqual({ filtered: ['a', 'c'], mode: 'exact', hybrid: 'a', nearby: ['exact', 'near'] });
});

test('checkpoint corruption rebuilds from committed records', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const name = `recovery-${crypto.randomUUID()}`;
    let db = await window.SereneDB.open(name), docs = await db.collection('docs');
    await docs.upsert([{ id: 'a', vector: [1, 0] }, { id: 'b', vector: [0, 1] }]); await docs.checkpoint(); await db.close();
    const raw = await new Promise<IDBDatabase>((resolve, reject) => { const request = indexedDB.open(name); request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error); });
    await new Promise<void>((resolve, reject) => {
      const tx = raw.transaction('indexes', 'readwrite'), store = tx.objectStore('indexes'), request = store.get('docs');
      request.onsuccess = () => { const record = request.result; record.payload = { format: -1 }; store.put(record); };
      tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error);
    }); raw.close();
    db = await window.SereneDB.open(name); docs = await db.collection('docs');
    const response = await docs.search({ vector: [1, 0], mode: 'exact' }); await db.destroy();
    return response.results.map(item => item.id);
  });
  expect(result).toEqual(['a', 'b']);
});

test('SIMD exact reference and HNSW recall stay correct off the UI thread', async ({ page }) => {
  const result = await page.evaluate(async () => {
    const db = await window.SereneDB.open(`recall-${crypto.randomUUID()}`), docs = await db.collection('docs', { dimension: 64, index: { efConstruction: 160, efSearch: 128 } });
    let seed = 7;
    const random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296) - 0.5;
    const records = Array.from({ length: 1500 }, (_, i) => ({ id: `r-${i}`, vector: Array.from({ length: 64 }, random) }));
    let beats = 0;
    const timer = setInterval(() => beats++, 10);
    await docs.upsert(records);
    let recall = 0;
    const times: number[] = [];
    for (let i = 0; i < 12; i++) {
      const vector = Array.from({ length: 64 }, random);
      const exact = await docs.search({ vector, mode: 'exact', topK: 10 });
      const start = performance.now();
      const ann = await docs.search({ vector, mode: 'ann', topK: 10, efSearch: 128 }); times.push(performance.now() - start);
      const ids = new Set(exact.results.map(item => item.id)); recall += ann.results.filter(item => ids.has(item.id)).length / 10;
    }
    clearInterval(timer);
    const stats = await docs.stats() as { kernel: string };
    await db.destroy(); times.sort((a, b) => a - b);
    return { recall: recall / 12, beats, kernel: stats.kernel, annP95: times[Math.floor(times.length * 0.95)] };
  });
  console.log(`Browser benchmark ${test.info().project.name}: ${JSON.stringify(result)}`);
  expect(result.recall).toBeGreaterThanOrEqual(0.95);
  expect(result.beats).toBeGreaterThan(5);
  expect(result.kernel).toBe('wasm-simd');
});
