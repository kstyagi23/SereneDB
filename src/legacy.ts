import { SereneDBError } from './errors.js';
import { SereneDB } from './index.js';
import type { Metric, OpenOptions } from './types.js';

// Legacy databases had no reliable persisted metric. Migration requires an explicit choice.
export async function migrateLegacy(sourceName: string, destinationName: string, metric: Metric, options: OpenOptions = {}): Promise<{ count: number }> {
  if (sourceName === destinationName) throw new SereneDBError('Legacy migration requires a different destination name. The original database is never modified.', 'LEGACY_DATA');
  const source = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(sourceName);
    request.onupgradeneeded = () => { request.transaction!.abort(); reject(new SereneDBError('Legacy source does not exist.', 'LEGACY_DATA')); };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new SereneDBError('Legacy source is blocked.', 'BLOCKED'));
  });
  let records: unknown[];
  try {
    if (!source.objectStoreNames.contains('vectors')) throw new SereneDBError('Source has no legacy vector store.', 'LEGACY_DATA');
    records = await new Promise((resolve, reject) => {
      const tx = source.transaction('vectors', 'readonly');
      const request = tx.objectStore('vectors').getAll();
      tx.oncomplete = () => resolve(request.result);
      tx.onabort = () => reject(tx.error);
    });
  } finally { source.close(); }
  const target = await SereneDB.open(destinationName, options);
  try {
    const collection = await target.collection('default', { metric });
    await collection.upsert(records as Parameters<typeof collection.upsert>[0]);
    await collection.checkpoint();
    return { count: await collection.count() };
  } finally { await target.close(); }
}
