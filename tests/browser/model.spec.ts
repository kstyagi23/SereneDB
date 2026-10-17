import { test, expect } from '@playwright/test';

test('real model embeddings, schema, cached assets, and offline worker restart', async ({ page, context }) => {
  test.setTimeout(300000);
  await page.goto('/test.html');
  const name = `model-${crypto.randomUUID()}`;
  const result = await page.evaluate(async name => {
    const db = await window.SereneDB.open(name);
    const config = window.transformersEmbedder({ model: 'Xenova/all-MiniLM-L6-v2', revision: '751bff37182d3f1213fa05d7196b954e230abad9', dtype: 'q8', device: 'wasm', batchSize: 4 });
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
  console.log('Model cache:', await page.evaluate(async () => {
    const result: Record<string, string[]> = {};
    for (const key of await caches.keys()) result[key] = (await (await caches.open(key)).keys()).map(request => request.url);
    return result;
  }));
  const remote: string[] = [];
  await context.route('**/*', async route => { const url = route.request().url(); if (!url.startsWith('http://127.0.0.1:4173')) { remote.push(url); console.log('Unexpected offline request:', url); await route.abort(); } else await route.continue(); });
  await page.reload();
  const offline = await page.evaluate(async name => {
    const db = await window.SereneDB.open(name), docs = await db.collection('docs');
    const response = await docs.search({ text: 'How can I store a vector database locally in a web app?' });
    await db.destroy(); return response.results[0].id;
  }, name);
  expect(offline).toBe('database'); expect(remote).toEqual([]);
});

test('studio download, semantic search, unload and cache-only reload preserve documents', async ({ page, context }) => {
  test.setTimeout(300000);
  await page.goto('/test.html');
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await expect(page.locator('#ingest')).toBeDisabled();
  await page.getByLabel('Allow downloading model files from Hugging Face').check();
  await page.getByRole('button', { name: 'Download & load model' }).click();
  await expect(page.locator('#model-state')).toContainText('Ready on CPU / WASM', { timeout: 180000 });
  await expect(page.locator('#model-state')).toContainText('384 dimensions');
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed 4 documents');
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
  await page.evaluate(async () => {
    const { Collection } = await import('/dist/index.js');
    const original = Collection.prototype.unload;
    Collection.prototype.unload = async function () { Collection.prototype.unload = original; throw new Error('Injected unload failure'); };
  });
  await page.getByLabel('Allow downloading model files from Hugging Face').uncheck();
  await expect(page.locator('#status')).toContainText('Injected unload failure');
  await expect(page.locator('#search')).toBeDisabled();
  await expect(page.locator('#ingest')).toBeDisabled();
  await page.getByRole('button', { name: 'Unload from memory' }).click();
  await expect(page.locator('#status')).toContainText('Cached files and saved documents are kept');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await expect(page.locator('#search')).toBeDisabled();
  await page.getByLabel('Allow downloading model files from Hugging Face').uncheck();
  const remote: string[] = [];
  await context.route('**/*', async route => {
    const url = route.request().url();
    if (!url.startsWith('http://127.0.0.1:4173')) { remote.push(url); await route.abort(); }
    else await route.continue();
  });
  await page.getByRole('button', { name: 'Load cached model' }).click();
  await expect(page.locator('#model-state')).toContainText('Ready on CPU / WASM', { timeout: 60000 });
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
  await page.reload();
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await expect(page.getByLabel('Allow downloading model files from Hugging Face')).not.toBeChecked();
  await page.getByRole('button', { name: 'Load cached model' }).click();
  await expect(page.locator('#model-state')).toContainText('Ready on CPU / WASM', { timeout: 60000 });
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
  await page.getByLabel('Embedding model').selectOption('custom');
  await page.getByLabel('Hugging Face repository').fill('https://huggingface.co/Xenova/all-MiniLM-L6-v2');
  await page.getByLabel('Revision', { exact: true }).fill('751bff37182d3f1213fa05d7196b954e230abad9');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Load cached model' }).click();
  await expect(page.locator('#model-state')).toContainText('Ready on CPU / WASM', { timeout: 60000 });
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
  expect(remote).toEqual([]);
});
