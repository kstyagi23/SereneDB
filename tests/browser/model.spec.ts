import { test, expect } from '@playwright/test';

test('real model embeddings, schema, cached assets, and offline worker restart', async ({ page, context }) => {
  test.setTimeout(300000);
  await page.goto('/test.html');
  const name = `model-${crypto.randomUUID()}`;
  const result = await page.evaluate(async name => {
    const db = await window.SereneDB.open(name);
    const config = window.transformersEmbedder({ model: 'Xenova/all-MiniLM-L6-v2', dtype: 'q8', device: 'wasm', batchSize: 4 });
    const docs = await db.collection('docs', { embedder: config });
    await docs.preload();
    await docs.upsert([
      { id: 'database', text: 'Persist and search vector embeddings offline in a browser using IndexedDB.' },
      { id: 'baking', text: 'Bake bread with flour and butter in an oven.' },
      { id: 'astronomy', text: 'Jupiter has many moons and is the largest planet.' },
    ]);
    const response = await docs.search({ text: 'How can I store a vector database locally in a web app?' });
    let rejected = false;
    try { await docs.upsert({ id: 'bad', vector: new Float32Array(384).fill(1) }); } catch { rejected = true; }
    const state = await docs.state(); await db.close();
    return { first: response.results[0].id, dimension: state.schema.dimension, rejected, score: response.results[0].score };
  }, name);
  expect(result.first).toBe('database'); expect(result.dimension).toBe(384); expect(result.rejected).toBe(true);
  const remote: string[] = [];
  await context.route('**/*', async route => { const url = route.request().url(); if (!url.startsWith('http://127.0.0.1:4173')) { remote.push(url); await route.abort(); } else await route.continue(); });
  await page.reload();
  const offline = await page.evaluate(async name => {
    const db = await window.SereneDB.open(name), docs = await db.collection('docs');
    const response = await docs.search({ text: 'How can I store a vector database locally in a web app?' });
    await db.destroy(); return response.results[0].id;
  }, name);
  expect(offline).toBe('database'); expect(remote).toEqual([]);
});
