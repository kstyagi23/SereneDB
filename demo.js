import { SereneDB } from './dist/index.js';
import { transformersEmbedder } from './dist/transformers.js';

// The browser regression suite also exercises the public package from this page.
window.SereneDB = SereneDB;
window.transformersEmbedder = transformersEmbedder;

const $ = id => document.getElementById(id);
const DEFAULT_MODEL = 'Xenova/all-MiniLM-L6-v2';
const DEFAULT_REVISION = '751bff37182d3f1213fa05d7196b954e230abad9';
const SAMPLE = $('documents').value;
const session = { db: null, collection: null, mode: 'lexical', count: 0, savedCount: 0, modelReady: false, modelTouched: false, runtime: null, cacheEntries: 0, busy: null, configVersion: 0, cacheVersion: 0, modelError: '', confirming: false, editorVersion: 0 };
let cacheTimer;
let opening;
let suspended = false;
let closing = Promise.resolve();

async function database() {
  if (suspended) throw new DOMException('Page is suspended.', 'AbortError');
  if (session.db) return session.db;
  opening ??= SereneDB.open('SereneDB-Lab').then(db => { session.db = db; return db; }).catch(error => { opening = null; throw error; });
  return opening;
}

function setStatus(message, state = 'idle') {
  $('status').textContent = message;
  document.querySelector('.activity-panel').dataset.state = state;
}

function draftDocuments() { return $('documents').value.split(/\r?\n\s*\r?\n/).map(text => text.trim()).filter(Boolean); }

function updateDraft() {
  const count = draftDocuments().length;
  $('draft-count').textContent = `${count} document${count === 1 ? '' : 's'}`;
  updateControls();
}

function readConfig() {
  const custom = $('model-preset').value === 'custom';
  let model = custom ? $('model-id').value.trim() : DEFAULT_MODEL;
  if (model.startsWith('https://')) {
    let url;
    try { url = new URL(model); } catch { throw new Error('Enter a Hugging Face repository ID or model-page URL.'); }
    if (url.hostname !== 'huggingface.co') throw new Error('Model URLs must point to huggingface.co.');
    const parts = url.pathname.split('/').filter(Boolean);
    if (parts.length !== 2) throw new Error('Paste the model page URL, not a file or collection URL.');
    model = parts.join('/');
  }
  if (!/^[\w.-]+\/[\w.-]+$/.test(model)) throw new Error('Enter a public repository in organization/model-name format.');
  return transformersEmbedder({
    model,
    revision: custom ? $('model-revision').value.trim() : DEFAULT_REVISION,
    dtype: $('model-dtype').value,
    device: $('device').value,
    pooling: $('model-pooling').value,
    normalize: $('model-normalize').checked,
    maxTokens: Number($('model-tokens').value),
    queryPrefix: $('query-prefix').value,
    documentPrefix: $('document-prefix').value,
    allowRemoteModels: false,
  });
}

function configForSpace(config) {
  const { device: _device, allowRemoteModels: _remote, batchSize: _size, batchTokens: _budget, localModelPath: _local, wasmPath: _wasm, ...space } = config;
  return space;
}

async function digest(value) {
  const bytes = new TextEncoder().encode(value);
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), byte => byte.toString(16).padStart(2, '0')).join('');
}

async function selectedCollectionName(config) {
  return `semantic-${await digest(JSON.stringify(configForSpace(config)))}`;
}

function showEmpty(title = 'A good answer starts here.', message = 'Save a few documents, then search. Everything stays in this browser.') {
  const empty = document.createElement('div'); empty.className = 'empty-state';
  const mark = document.createElement('span'); mark.className = 'empty-mark'; mark.textContent = '\u2315'; mark.setAttribute('aria-hidden', 'true');
  const heading = document.createElement('strong'); heading.textContent = title;
  const copy = document.createElement('p'); copy.textContent = message;
  empty.append(mark, heading, copy); $('results').replaceChildren(empty);
  $('search-metrics').textContent = 'Ready when you are';
}

function updateControls() {
  const busy = Boolean(session.busy);
  const semantic = session.mode === 'semantic';
  for (const control of document.querySelectorAll('input[name="mode"], #model-settings input, #model-settings select, #model-advanced, #top-k, #query')) {
    if ('disabled' in control) control.disabled = busy;
  }
  $('sample').disabled = busy;
  for (const suggestion of document.querySelectorAll('[data-query]')) suggestion.disabled = busy;
  $('import-text').disabled = busy;
  $('ingest').disabled = busy || draftDocuments().length === 0 || semantic && !session.modelReady;
  $('search').disabled = busy || session.count === 0 || !$('query').value.trim() || semantic && !session.modelReady;
  $('retrieval').disabled = busy || !semantic;
  $('load-model').disabled = busy || session.modelReady;
  $('load-model').textContent = session.modelReady ? 'Model ready' : $('download-consent').checked ? 'Download & load model' : 'Load cached model';
  $('unload-model').disabled = busy || !session.modelTouched;
  $('remove-model').disabled = busy || session.cacheEntries === 0;
  $('reset').disabled = busy || session.count === 0;
  $('export').disabled = busy || session.savedCount === 0;
  $('cancel').hidden = !busy;
  $('cancel').disabled = busy && !session.busy.cancellable;
  $('documents').disabled = busy;
  $('results').setAttribute('aria-busy', session.busy?.kind === 'search' ? 'true' : 'false');
  $('model-settings').hidden = !semantic;
  $('collection-count').textContent = `${session.count} saved document${session.count === 1 ? '' : 's'}`;
  $('engine-badge').textContent = !semantic ? 'No model needed' : session.modelReady ? 'Model ready' : 'Model optional';
  $('search-mode-label').textContent = semantic ? 'Semantic' : 'Keyword';
  $('save-help').textContent = semantic && !session.modelReady ? 'Load a cached model or opt in to a download before saving or searching. Stored documents stay available after unloading.' : 'New documents are added in one transaction. Identical text is saved only once; existing documents are kept.';
  let modelText = 'Not loaded. Downloads are off.';
  if (session.modelReady) modelText = `Ready on ${session.runtime?.device === 'webgpu' ? 'WebGPU' : 'CPU / WASM'}${session.runtime?.dimension ? ` / ${session.runtime.dimension} dimensions` : ''}.`;
  else if (session.busy?.kind === 'model') modelText = 'Loading model files. Documents stay on your device.';
  else if (session.modelError) modelText = session.modelError;
  else if ($('download-consent').checked) modelText = 'Not loaded. Download permission is on for this session.';
  $('model-state').textContent = modelText;
  $('model-state').dataset.state = session.modelReady ? 'loaded' : session.modelError ? 'error' : 'idle';
}

function updateModelSummary() {
  const custom = $('model-preset').value === 'custom';
  $('custom-model-fields').hidden = !custom;
  const defaultSettings = !custom && $('model-dtype').value === 'q8' && $('model-pooling').value === 'mean';
  $('model-size').textContent = !custom && $('model-dtype').value === 'q8' ? '~23 MB weights' : !custom && $('model-dtype').value === 'fp32' ? '~90 MB weights' : 'Size depends on repository';
  $('model-shape').textContent = defaultSettings ? '384 dimensions' : 'Check model compatibility';
  try {
    const config = readConfig();
    $('model-link').href = `https://huggingface.co/${config.model}`;
    $('model-link').hidden = false;
  } catch { $('model-link').hidden = true; }
}

function updateStrategy() {
  const semantic = session.mode === 'semantic';
  const choices = semantic ? [['hybrid', 'Hybrid / vector + keywords'], ['auto', 'Vector / automatic'], ['exact', 'Vector / exact']] : [['hybrid', 'Keyword / BM25']];
  $('retrieval').replaceChildren(...choices.map(([value, text]) => new Option(text, value)));
}

async function cacheRequests(config) {
  if (!('caches' in window)) return [];
  const cache = await caches.open('transformers-cache');
  const prefix = `https://huggingface.co/${config.model}/resolve/${encodeURIComponent(config.revision)}/`;
  return (await cache.keys()).filter(request => request.url.startsWith(prefix));
}

async function refreshCache(version = session.configVersion) {
  const inspection = ++session.cacheVersion;
  try {
    const config = readConfig();
    const requests = await cacheRequests(config);
    if (inspection !== session.cacheVersion || version !== session.configVersion) return;
    session.cacheEntries = requests.length;
    $('cache-state').textContent = requests.length ? `${requests.length} model file${requests.length === 1 ? '' : 's'} cached on this device` : 'No files cached for this revision';
  } catch (error) {
    if (inspection !== session.cacheVersion || version !== session.configVersion) return;
    session.cacheEntries = 0;
    $('cache-state').textContent = 'Cache unavailable or model settings incomplete';
  }
  updateControls();
}

async function inspectSelection(version = session.configVersion) {
  try {
    if (!session.db) return;
    const config = session.mode === 'semantic' ? readConfig() : null;
    const name = config ? await selectedCollectionName(config) : 'lexical';
    const states = await session.db.listCollections();
    if (version !== session.configVersion) return;
    session.savedCount = states.reduce((sum, state) => sum + state.count, 0);
    session.count = states.find(state => state.name === name)?.count ?? 0;
  } catch (error) {
    if (version !== session.configVersion) return;
    session.count = 0;
  }
  updateControls();
}

function settingsChanged() {
  session.configVersion++;
  session.cacheVersion++;
  const previous = session.collection;
  session.collection = null;
  if (previous) void previous.unload().catch(() => {});
  session.modelReady = false; session.modelTouched = false; session.runtime = null; session.modelError = ''; session.count = 0;
  session.cacheEntries = 0;
  $('cache-state').textContent = 'Checking local cache...';
  updateModelSummary(); updateControls();
  showEmpty('A fresh embedding space.', 'Each model configuration uses its own collection. Existing documents are kept.');
  setStatus(session.mode === 'semantic' ? 'Model settings changed. Load this model before saving or searching.' : 'Keyword search is ready. No model download is needed.');
  clearTimeout(cacheTimer);
  cacheTimer = setTimeout(() => { void inspectSelection(); void refreshCache(); }, 180);
}

async function ensureCollection(options = {}) {
  await database();
  const config = session.mode === 'semantic' ? readConfig() : null;
  const name = config ? await selectedCollectionName(config) : 'lexical';
  const previous = session.collection;
  if (previous && previous.name !== name) await previous.unload();
  const collection = await session.db.collection(name, config ? { embedder: config } : { dimension: 8, index: 'flat' }, options);
  session.collection = collection;
  session.count = await collection.count();
  return collection;
}

function reportProgress(progress) {
  if (!session.busy || session.busy.controller.signal.aborted) return;
  $('operation-progress').hidden = false;
  const detail = progress.detail ?? {};
  let label = '';
  if (progress.phase === 'loading-model') {
    const file = typeof detail.file === 'string' ? detail.file.split('/').pop() : '';
    label = file ? `${file}${progress.total > 1 ? ` / ${(100 * progress.completed / progress.total).toFixed(0)}%` : ''}` : detail.fallback ? 'Using the CPU fallback' : 'Preparing tokenizer and model weights';
  } else if (progress.phase === 'embedding') label = detail.warming ? 'Warming up the local model' : `Embedding ${progress.completed} / ${progress.total} documents`;
  else if (progress.phase === 'indexing') label = `Loading ${progress.completed} / ${progress.total} vectors`;
  else label = `${progress.phase}: ${progress.completed} / ${progress.total}`;
  $('progress-label').textContent = label;
  if (progress.total > 1 || progress.phase !== 'loading-model') {
    $('progress-bar').max = Math.max(1, progress.total);
    $('progress-bar').value = Math.max(0, progress.completed);
  } else $('progress-bar').removeAttribute('value');
}

function operationOptions() { return { signal: session.busy.controller.signal, onProgress: reportProgress }; }

function checkCancelled() { if (session.busy.controller.signal.aborted) throw new DOMException('Operation canceled.', 'AbortError'); }

function errorMessage(error, kind) {
  if (error?.name === 'AbortError') return 'Canceled. Previously committed documents and completed cached downloads are kept.';
  if (kind === 'model' && !$('download-consent').checked) return `The model could not load from cache. Enable Hugging Face downloads or check the selected repository, revision, and weights. ${error.message}`;
  if (kind === 'model') return `Could not load the model. Check its public ONNX artifacts, revision, pooling, and browser compatibility. ${error.message}`;
  return error?.message ?? String(error);
}

async function run(kind, message, action) {
  if (session.busy || session.confirming || suspended) return;
  const operation = { kind, controller: new AbortController(), cancellable: true };
  let finish;
  operation.finished = new Promise(resolve => { finish = resolve; });
  session.busy = operation;
  setStatus(message, 'busy'); $('operation-progress').hidden = false; $('progress-bar').removeAttribute('value'); $('progress-label').textContent = 'Preparing...';
  updateControls();
  try { await action(); }
  catch (error) {
    const message = errorMessage(error, kind);
    if (kind === 'model') { session.modelReady = false; session.modelError = error.name === 'AbortError' ? '' : 'Load failed. Your documents are unchanged.'; }
    setStatus(message, error.name === 'AbortError' ? 'idle' : 'error');
  } finally {
    if (session.busy === operation) session.busy = null;
    $('operation-progress').hidden = true;
    updateControls();
    finish();
  }
}

async function loadModel() {
  if (session.mode !== 'semantic') return;
  await run('model', $('download-consent').checked ? 'Downloading the selected model from Hugging Face. Text stays on this device.' : 'Loading the selected model from local cache. Remote model access is disabled.', async () => {
    // Check configuration before opening a model collection or requesting runtime files.
    readConfig();
    session.modelError = '';
    const collection = await ensureCollection(operationOptions());
    session.modelTouched = true;
    updateControls();
    try {
      const info = await collection.preload({ ...operationOptions(), runtime: { allowRemoteModels: $('download-consent').checked, device: $('device').value } });
      checkCancelled();
      session.busy.cancellable = false;
      session.modelReady = true; session.runtime = info;
      updateControls();
      setStatus(`Model ready on ${info.device === 'webgpu' ? 'WebGPU' : 'CPU / WASM'}. ${info.dimension} dimensions. ${info.fallbackReason ?? 'You can now save documents and search by meaning.'}`, 'success');
    } finally { await refreshCache(); await inspectSelection(); }
  });
}

async function changedPermission() {
  updateControls();
  if (session.busy) return;
  if (session.modelTouched && session.collection) {
    session.modelReady = false; session.runtime = null;
    updateControls();
    await run('permission', 'Applying model download permission...', async () => {
      await session.collection.unload();
      session.modelReady = false; session.modelTouched = false; session.runtime = null;
      setStatus('Download permission changed. Load the model again to apply it; stored documents are unchanged.');
    });
  }
  updateControls();
}

function confirmAction(title, description, label) {
  session.confirming = true;
  $('confirm-title').textContent = title; $('confirm-description').textContent = description; $('confirm-action').textContent = label;
  const dialog = $('confirm-dialog'); dialog.returnValue = '';
  return new Promise(resolve => {
    dialog.addEventListener('close', () => { session.confirming = false; resolve(dialog.returnValue === 'confirm'); }, { once: true });
    dialog.showModal();
  });
}

function addHighlightedText(element, text, query) {
  const words = Array.from(new Set(query.toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [])).filter(word => word.length > 1).slice(0, 30);
  if (!words.length) { element.textContent = text; return; }
  const expression = new RegExp(`(?<![\\p{L}\\p{N}_])(?:${words.sort((a, b) => b.length - a.length).map(word => word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})(?![\\p{L}\\p{N}_])`, 'giu');
  let last = 0;
  for (const match of text.matchAll(expression)) {
    element.append(document.createTextNode(text.slice(last, match.index)));
    const mark = document.createElement('mark'); mark.textContent = match[0]; element.append(mark); last = match.index + match[0].length;
  }
  element.append(document.createTextNode(text.slice(last)));
}

function renderResults(response, query) {
  if (!response.results.length) {
    showEmpty('No matches this time.', session.mode === 'lexical' ? 'Try words that appear in a document, or load a semantic model to search by meaning.' : 'Try a different query or add more documents to this model collection.');
    $('search-metrics').textContent = '0 matches'; return;
  }
  const cards = response.results.map((result, i) => {
    const card = document.createElement('article'); card.className = 'result';
    const rank = document.createElement('span'); rank.className = 'result-rank'; rank.textContent = String(i + 1).padStart(2, '0');
    const content = document.createElement('div');
    const heading = document.createElement('div'); heading.className = 'result-heading';
    const title = document.createElement('strong'); title.textContent = result.metadata?.title || `Document ${result.id.slice(-8)}`;
    const score = document.createElement('span'); score.className = 'result-score'; score.textContent = `${response.mode === 'hybrid' ? 'RRF' : 'Cosine'} ${result.score.toFixed(4)}`;
    if (response.mode === 'hybrid') score.title = 'Reciprocal rank score. Keyword matches use BM25 retrieval; this number is a rank-fusion score, not a similarity percentage.';
    heading.append(title, score);
    const text = result.text ?? '';
    const excerpt = document.createElement('p'); addHighlightedText(excerpt, text.length > 560 ? `${text.slice(0, 560)}...` : text, query);
    content.append(heading, excerpt);
    if (text.length > 560) {
      const details = document.createElement('details'), summary = document.createElement('summary'), full = document.createElement('p'); summary.textContent = 'Read full document'; full.textContent = text; details.append(summary, full); content.append(details);
    }
    card.append(rank, content); return card;
  });
  $('results').replaceChildren(...cards);
  $('search-metrics').textContent = `${response.results.length} matches / ${response.elapsedMs.toFixed(1)} ms`;
}

async function saveDocuments() {
  if (session.mode === 'semantic' && !session.modelReady) { setStatus('Load the selected model before saving documents.', 'error'); return; }
  await run('save', 'Preparing documents for a committed local save...', async () => {
    const texts = draftDocuments();
    if (!texts.length) throw new Error('Add at least one nonempty document.');
    if ($('documents').value.length > 2000000 || texts.length > 1000) throw new Error('Keep the demo editor under 2 million characters and 1,000 documents. Use the streaming API for larger datasets.');
    const collection = await ensureCollection(operationOptions());
    const existing = session.count ? await collection.scan({ limit: 100000, filter: { text: { $in: texts } } }, operationOptions()) : [];
    const existingIds = new Map(existing.filter(record => typeof record.text === 'string').map(record => [record.text, record.id]));
    const records = await Promise.all(texts.map(async text => ({
      id: existingIds.get(text) ?? `text-${await digest(text)}`,
      text,
      metadata: { source: 'studio', title: text.split('\n')[0].slice(0, 68) },
      ...(session.mode === 'lexical' ? { vector: [1, 0, 0, 0, 0, 0, 0, 0] } : {}),
    })));
    checkCancelled();
    const result = await collection.upsert(records, operationOptions());
    // Once storage acknowledges, present its commit even if cancellation arrived late.
    session.busy.cancellable = false;
    session.count = result.count;
    await inspectSelection();
    const distinct = new Set(records.map(record => record.id)).size;
    setStatus(`Committed ${distinct} document${distinct === 1 ? '' : 's'}. ${result.count} saved in this collection / revision ${result.revision}.`, 'success');
    showEmpty('Your collection is ready.', 'Search with keywords, or ask a question in semantic mode.');
  });
}

async function searchDocuments() {
  if (session.count === 0 || session.mode === 'semantic' && !session.modelReady) return;
  const query = $('query').value.trim();
  if (!query) { $('query').focus(); return; }
  await run('search', 'Searching your local collection...', async () => {
    const collection = await ensureCollection(operationOptions());
    const response = await collection.search({ text: query, mode: session.mode === 'lexical' ? 'hybrid' : $('retrieval').value, topK: Number($('top-k').value) }, operationOptions());
    checkCancelled(); renderResults(response, query);
    setStatus(`${response.results.length} result${response.results.length === 1 ? '' : 's'} in ${response.elapsedMs.toFixed(1)} ms / ${session.mode === 'lexical' ? 'BM25 keyword retrieval' : response.mode} / revision ${response.revision}.`, 'success');
  });
}

$('documents').addEventListener('input', () => { session.editorVersion++; updateDraft(); });
$('query').addEventListener('input', updateControls);
$('search-form').addEventListener('submit', event => { event.preventDefault(); void searchDocuments(); });
$('ingest').addEventListener('click', () => { void saveDocuments(); });
$('load-model').addEventListener('click', () => { void loadModel(); });
$('cancel').addEventListener('click', () => {
  if (session.busy?.cancellable) { session.busy.controller.abort(); session.busy.cancellable = false; $('cancel').disabled = true; setStatus('Canceling. In-flight native inference or file requests may finish before cleanup.', 'busy'); }
});
$('download-consent').addEventListener('change', () => { void changedPermission(); });
$('sample').addEventListener('click', () => { $('documents').value = SAMPLE; session.editorVersion++; updateDraft(); setStatus('Sample documents are in the editor. Save them to update the collection.'); });
$('import-text').addEventListener('click', () => { $('text-file').click(); });
$('text-file').addEventListener('change', () => {
  const file = $('text-file').files[0]; $('text-file').value = '';
  if (!file) return;
  void run('file', `Reading ${file.name} locally...`, async () => {
    if (file.size > 2 * 1024 * 1024) throw new Error('Choose a text file smaller than 2 MiB for this demo.');
    const text = await file.text(); checkCancelled();
    if (text.includes('\u0000') || !text.trim()) throw new Error('Choose a nonempty plain text or Markdown file.');
    $('documents').value = text; session.editorVersion++; updateDraft();
    setStatus(`${file.name} loaded into the editor. Save documents to persist it.`, 'success');
  });
});

for (const radio of document.querySelectorAll('input[name="mode"]')) radio.addEventListener('change', () => {
  session.mode = radio.value;
  updateStrategy(); settingsChanged();
});
for (const input of document.querySelectorAll('#model-settings input:not(#download-consent), #model-settings select')) input.addEventListener(input.tagName === 'INPUT' && input.type !== 'checkbox' ? 'input' : 'change', () => {
  if (input.id === 'model-preset' && input.value === 'minilm') {
    $('model-dtype').value = 'q8'; $('model-pooling').value = 'mean'; $('model-tokens').value = '512'; $('model-normalize').checked = true; $('query-prefix').value = ''; $('document-prefix').value = '';
  }
  if (input.id === 'model-preset' && input.value === 'custom') $('model-advanced').open = true;
  settingsChanged();
});

for (const button of document.querySelectorAll('[data-query]')) button.addEventListener('click', () => {
  if (session.busy) return;
  $('query').value = button.dataset.query; updateControls();
  if (!$('search').disabled) void searchDocuments(); else $('query').focus();
});

$('unload-model').addEventListener('click', () => { void run('unload', 'Releasing the model from memory...', async () => {
  if (session.collection) await session.collection.unload();
  session.modelReady = false; session.modelTouched = false; session.runtime = null; session.modelError = '';
  setStatus('Model unloaded from memory. Cached files and saved documents are kept. Load it again to resume semantic search.', 'success');
  await refreshCache();
}); });

$('remove-model').addEventListener('click', async () => {
  if (session.busy || session.confirming) return;
  if (!await confirmAction('Remove this model from cache?', 'Only Hugging Face files for this model and revision will be removed. Documents, vectors, other models, and shared runtime files are kept.', 'Remove cached files')) return;
  void run('remove', 'Removing cached model files...', async () => {
    const config = readConfig();
    if (session.collection) await session.collection.unload();
    session.modelReady = false; session.modelTouched = false; session.runtime = null;
    const requests = await cacheRequests(config); checkCancelled();
    session.busy.cancellable = false; updateControls();
    const cache = await caches.open('transformers-cache');
    await Promise.all(requests.map(request => cache.delete(request)));
    await refreshCache();
    setStatus(`Removed ${requests.length} cached model file${requests.length === 1 ? '' : 's'}. Stored documents and vectors are unchanged.`, 'success');
  });
});

$('reset').addEventListener('click', async () => {
  if (session.busy || session.confirming) return;
  if (!await confirmAction('Clear this collection?', `Delete all ${session.count} saved documents and vectors in the selected collection? The editor, model cache, and other collections are kept.`, 'Clear collection')) return;
  void run('clear', 'Clearing the selected collection...', async () => {
    const collection = await ensureCollection(operationOptions());
    const result = await collection.clear(operationOptions()); session.busy.cancellable = false;
    session.count = result.count; await inspectSelection(); showEmpty();
    setStatus('Collection cleared and committed. Model files and other collections are unchanged.', 'success');
  });
});

$('export').addEventListener('click', () => { void run('export', 'Reading a backup from committed local storage...', async () => {
  const backup = await session.db.export(operationOptions()); checkCancelled();
  const url = URL.createObjectURL(new Blob([JSON.stringify(backup)], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'serenedb-backup.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setStatus('Backup exported. It includes saved documents and vectors from all studio collections, not model weights.', 'success');
}); });

window.addEventListener('pagehide', () => {
  suspended = true;
  session.configVersion++;
  clearTimeout(cacheTimer);
  session.busy?.controller.abort();
  const db = session.db;
  session.db = null; opening = null; session.collection = null;
  session.modelReady = false; session.modelTouched = false; session.runtime = null;
  closing = Promise.all([db?.close().catch(() => {}), session.busy?.finished]).then(() => {});
});

window.addEventListener('pageshow', event => {
  if (!event.persisted) return;
  void (async () => {
    await closing;
    suspended = false;
    await run('restore', 'Reconnecting to local storage...', async () => {
      await database(); await inspectSelection(); await refreshCache();
      setStatus(session.mode === 'semantic' ? 'Local collection reconnected. Load the cached model to resume semantic search.' : 'Local collection reconnected. Your saved documents are ready.', 'success');
    });
  })();
});

async function initialize() {
  updateStrategy(); updateDraft(); updateModelSummary();
  const version = session.configVersion;
  try {
    await database();
    await inspectSelection();
    const state = (await session.db.listCollections()).find(collection => collection.name === 'lexical');
    if (state?.count && version === session.configVersion && session.mode === 'lexical' && session.editorVersion === 0 && !session.busy) {
      const collection = await session.db.collection('lexical');
      const records = await collection.scan({ limit: 1000 });
      if (version === session.configVersion && session.editorVersion === 0 && !session.busy) {
        $('documents').value = records.map(record => record.text).filter(Boolean).join('\n\n');
        updateDraft(); setStatus(`Restored ${state.count} saved document${state.count === 1 ? '' : 's'} from this browser. No model download needed.`, 'success');
      }
    }
    const storage = await SereneDB.storageInfo();
    if (storage.usage !== undefined) $('storage-status').textContent = `${(storage.usage / 1024 / 1024).toFixed(1)} MB origin storage`;
  } catch (error) { setStatus(`Local database unavailable: ${error.message}`, 'error'); }
  await refreshCache(); updateControls();
}

void initialize();
