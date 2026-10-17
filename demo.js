import { SereneDB } from './dist/index.js';
import { transformersEmbedder } from './dist/transformers.js';

window.SereneDB = SereneDB;
window.transformersEmbedder = transformersEmbedder;
const status = document.getElementById('status');
let db, collection, currentMode;
const busy = async action => {
  for (const button of document.querySelectorAll('button')) button.disabled = true;
  try { await action(); } catch (error) { status.textContent = `${error.name}: ${error.message}`; }
  finally { for (const button of document.querySelectorAll('button')) button.disabled = false; }
};
async function open() {
  const mode = document.getElementById('mode').value;
  if (!db) db = await SereneDB.open('SereneDB-Lab');
  if (mode !== currentMode) {
    collection = await db.collection(mode === 'semantic' ? 'semantic' : 'lexical', mode === 'semantic' ? {
      embedder: transformersEmbedder({ model: 'Xenova/all-MiniLM-L6-v2', dtype: 'q8', device: 'auto', pooling: 'mean', normalize: true }),
    } : { dimension: 8, index: 'flat' });
    currentMode = mode;
  }
  return collection;
}
document.getElementById('ingest').onclick = () => busy(async () => {
  const documents = document.getElementById('documents').value.split(/\n\s*\n/).map(text => text.trim()).filter(Boolean);
  await open();
  status.textContent = 'Preparing documents...';
  const inputs = documents.map((text, i) => ({ id: `document-${i + 1}`, text, metadata: { source: 'lab' }, ...(currentMode === 'lexical' ? { vector: [1, 0, 0, 0, 0, 0, 0, 0] } : {}) }));
  const result = await collection.upsert(inputs, { onProgress: progress => { status.textContent = `${progress.phase}: ${progress.completed}/${progress.total}`; } });
  await collection.checkpoint();
  status.textContent = `Committed ${result.upserted} records. Collection has ${result.count} records.\nRevision ${result.revision}. Workers enabled; documents remain local.`;
});
document.getElementById('search').onclick = () => busy(async () => {
  await open();
  const text = document.getElementById('query').value;
  const mode = currentMode === 'lexical' ? 'hybrid' : document.getElementById('retrieval').value;
  const response = await collection.search({ text, mode, topK: 5 });
  const results = document.getElementById('results'); results.replaceChildren();
  for (const result of response.results) {
    const card = document.createElement('div'); card.className = 'result';
    const title = document.createElement('strong'); title.textContent = `${result.id} / ${result.score.toFixed(4)}`;
    const paragraph = document.createElement('p'); paragraph.textContent = result.text;
    card.append(title, paragraph); results.append(card);
  }
  status.textContent = `${response.results.length} results / ${response.mode} / ${response.elapsedMs.toFixed(2)} ms\nSnapshot revision ${response.revision}.`;
});
document.getElementById('reset').onclick = () => busy(async () => { await open(); await collection.clear(); status.textContent = 'Collection cleared and committed.'; document.getElementById('results').replaceChildren(); });
document.getElementById('export').onclick = () => busy(async () => {
  await open();
  const url = URL.createObjectURL(new Blob([JSON.stringify(await db.export())], { type: 'application/json' }));
  const link = document.createElement('a'); link.href = url; link.download = 'serenedb-backup.json'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  status.textContent = 'Backup exported from committed storage.';
});
