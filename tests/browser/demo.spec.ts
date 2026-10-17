import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/test.html');
  await page.waitForFunction(() => typeof window.SereneDB === 'function');
});

test('keyword workflow is immediate, private, deduplicated and keyboard friendly', async ({ page, context }) => {
  const requests: string[] = [];
  await context.route('https://huggingface.co/**', route => { requests.push(route.request().url()); return route.abort(); });
  await expect(page.locator('#model-settings')).toBeHidden();
  await expect(page.locator('#search')).toBeDisabled();
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed 4 documents');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await expect(page.locator('#search')).toBeEnabled();
  await page.getByRole('textbox', { name: 'Search query' }).press('Enter');
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
  await expect(page.locator('.result').first()).toContainText('RRF');
  await expect(page.locator('#search-metrics')).toContainText('ms');
  expect(requests).toEqual([]);
});

test('semantic mode requires a deliberate load and Hugging Face consent is off by default', async ({ page, context }) => {
  const requests: string[] = [];
  await context.route('https://huggingface.co/**', route => { requests.push(route.request().url()); return route.abort(); });
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await expect(page.locator('#model-settings')).toBeVisible();
  await expect(page.getByLabel('Allow downloading model files from Hugging Face')).not.toBeChecked();
  await expect(page.getByRole('button', { name: 'Load cached model' })).toBeEnabled();
  await expect(page.locator('#ingest')).toBeDisabled();
  await expect(page.locator('#unload-model')).toBeDisabled();
  await page.getByLabel('Allow downloading model files from Hugging Face').check();
  await expect(page.getByRole('button', { name: 'Download & load model' })).toBeEnabled();
  await page.getByLabel('Allow downloading model files from Hugging Face').uncheck();
  await expect(page.getByRole('button', { name: 'Load cached model' })).toBeEnabled();
  expect(requests).toEqual([]);
});

test('custom models expose repository, revision and model-card settings without requesting files', async ({ page, context }) => {
  const requests: string[] = [];
  await context.route('https://huggingface.co/**', route => { requests.push(route.request().url()); return route.abort(); });
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await page.getByLabel('Embedding model').selectOption('custom');
  await page.getByLabel('Hugging Face repository').fill('https://huggingface.co/example/embedding-model');
  await page.getByLabel('Revision', { exact: true }).fill('test-revision');
  await page.getByLabel('Pooling', { exact: true }).selectOption('cls');
  await page.getByLabel('Query prefix').fill('query: ');
  await page.getByLabel('Document prefix').fill('passage: ');
  await expect(page.locator('#model-link')).toHaveAttribute('href', 'https://huggingface.co/example/embedding-model');
  await expect(page.locator('#ingest')).toBeDisabled();
  expect(requests).toEqual([]);
  await page.getByLabel('Hugging Face repository').fill('not-a-repository');
  await page.getByRole('button', { name: 'Load cached model' }).click();
  await expect(page.locator('#status')).toContainText('organization/model-name');
  await expect(page.locator('#cancel')).toBeHidden();
  expect(requests).toEqual([]);
});

test('cache-only loading fails helpfully without requesting missing files from Hugging Face', async ({ page, context }) => {
  const requests: string[] = [];
  await context.route('https://huggingface.co/**', route => { requests.push(route.request().url()); return route.abort(); });
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await page.getByLabel('Embedding model').selectOption('custom');
  await page.getByLabel('Hugging Face repository').fill('serenedb-test/no-cached-model');
  await page.getByRole('button', { name: 'Load cached model' }).click();
  await expect(page.locator('#status')).toContainText('could not load from cache');
  await expect(page.locator('#ingest')).toBeDisabled();
  await expect(page.locator('#cancel')).toBeHidden();
  expect(requests).toEqual([]);
});

test('model downloads expose progress and can be canceled without saving documents', async ({ page, context }) => {
  const pending: Array<() => void> = [];
  let firstRequest: (() => void) | undefined;
  const intercepted = new Promise<void>(resolve => { firstRequest = resolve; });
  await context.route('https://huggingface.co/**', async route => {
    firstRequest?.();
    await new Promise<void>(resolve => { pending.push(resolve); });
    await route.abort();
  });
  try {
    await page.getByRole('radio', { name: 'Semantic search' }).check();
    await page.getByLabel('Allow downloading model files from Hugging Face').check();
    await page.getByRole('button', { name: 'Download & load model' }).click();
    await intercepted;
    await expect(page.locator('#operation-progress')).toBeVisible();
    await expect(page.locator('#cancel')).toBeEnabled();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.locator('#status')).toContainText('Canceled');
    await expect(page.locator('#collection-count')).toHaveText('0 saved documents');
    await expect(page.locator('#search')).toBeDisabled();
    await expect(page.locator('#cancel')).toBeHidden();
  } finally { for (const release of pending) release(); await context.unrouteAll({ behavior: 'ignoreErrors' }); }
});

test('clear is confirmed and canceling it preserves documents', async ({ page }) => {
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Clear collection', exact: true }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('button', { name: 'Keep it' }).click();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Clear collection', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Clear collection' }).click();
  await expect(page.locator('#status')).toContainText('Collection cleared and committed');
  await expect(page.locator('#collection-count')).toHaveText('0 saved documents');
  await expect(page.locator('#search')).toBeDisabled();
});

test('reload restores stored text and text files stay local until saved', async ({ page }) => {
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed');
  await page.reload();
  await expect(page.locator('#status')).toContainText('Restored 4 saved documents');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.locator('#text-file').setInputFiles({ name: 'notes.txt', mimeType: 'text/plain', buffer: Buffer.from('Local imported note about penguins.') });
  await expect(page.locator('#documents')).toHaveValue('Local imported note about penguins.');
  await expect(page.locator('#draft-count')).toHaveText('1 document');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#collection-count')).toHaveText('5 saved documents');
  await page.getByRole('textbox', { name: 'Search query' }).fill('no-match-in-this-collection');
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('#results')).toContainText('No matches this time');
});

test('older demo document IDs are reused instead of duplicating restored text', async ({ page }) => {
  await page.evaluate(async () => {
    const db = await window.SereneDB.open('SereneDB-Lab');
    const docs = await db.collection('lexical', { dimension: 8, index: 'flat' });
    await docs.upsert([
      { id: 'document-1', text: 'Legacy browser database note.', vector: [1, 0, 0, 0, 0, 0, 0, 0] },
      { id: 'document-2', text: 'Legacy embedding model note.', vector: [1, 0, 0, 0, 0, 0, 0, 0] },
    ]); await db.close();
  });
  await page.reload();
  await expect(page.locator('#status')).toContainText('Restored 2 saved documents');
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed 2 documents');
  await expect(page.locator('#collection-count')).toHaveText('2 saved documents');
});

test('Back/Forward page restoration reconnects local collection handles', async ({ page }) => {
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed');
  await page.evaluate(() => {
    dispatchEvent(new PageTransitionEvent('pagehide', { persisted: true }));
    dispatchEvent(new PageTransitionEvent('pageshow', { persisted: true }));
  });
  await expect(page.locator('#status')).toContainText('Local collection reconnected');
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result').first()).toContainText('IndexedDB');
});

test('enabled form boundaries maintain non-text contrast without relying on focus', async ({ page }) => {
  const ratios = await page.evaluate(() => {
    const luminance = (rgb: string) => {
      const channels = rgb.match(/[0-9.]+/g)!.slice(0, 3).map(value => Number(value) / 255).map(value => value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4);
      return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
    };
    return ['#documents', '#query', '#top-k'].map(selector => {
      const input = document.querySelector(selector)!;
      const a = luminance(getComputedStyle(input).borderColor);
      const b = luminance(getComputedStyle(input.closest('section')!).backgroundColor);
      return (Math.max(a, b) + .05) / (Math.min(a, b) + .05);
    });
  });
  expect(ratios.every(ratio => ratio >= 3)).toBe(true);
});

test('cache removal is scoped to a selected model revision and never clears vectors', async ({ page }) => {
  await page.evaluate(async () => {
    const cache = await caches.open('transformers-cache');
    for (const url of [
      'https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/config.json',
      'https://huggingface.co/another/model/resolve/main/config.json',
      `${location.origin}/dist/vendor/ort-wasm-simd-threaded.jsep.wasm`,
    ]) await cache.put(url, new Response('{}'));
  });
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await expect(page.locator('#cache-state')).toContainText('1 model file cached');
  await page.getByRole('button', { name: 'Remove cached files' }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Remove cached files' }).click();
  await expect(page.locator('#status')).toContainText('Removed 1 cached model file');
  const urls = await page.evaluate(async () => (await (await caches.open('transformers-cache')).keys()).map(request => request.url));
  expect(urls).toHaveLength(2);
  expect(urls.some(url => url.includes('another/model'))).toBe(true);
  expect(urls.some(url => url.includes('/dist/vendor/'))).toBe(true);
  await page.getByRole('radio', { name: 'Keyword search' }).check();
  await expect(page.locator('#collection-count')).toHaveText('4 saved documents');
});

test('documents and highlighted queries are rendered as text, never HTML', async ({ page }) => {
  await page.locator('#documents').fill('<img src=x onerror="window.__unsafe = true"> needle');
  await page.getByRole('button', { name: 'Save documents' }).click();
  await expect(page.locator('#status')).toContainText('Committed');
  await page.getByRole('textbox', { name: 'Search query' }).fill('needle');
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await expect(page.locator('.result')).toContainText('<img');
  await expect(page.locator('.result mark')).toHaveText('needle');
  await expect(page.locator('.result img')).toHaveCount(0);
  expect(await page.evaluate(() => window['__unsafe'])).toBeUndefined();
});

test('mobile semantic controls and modal stay inside the viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole('radio', { name: 'Semantic search' }).check();
  await page.getByLabel('Embedding model').selectOption('custom');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await expect(page.getByLabel('Hugging Face repository')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Load cached model' })).toBeVisible();
});
