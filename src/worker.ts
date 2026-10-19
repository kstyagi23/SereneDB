import { Engine } from './engine.js';
import type { OpenOptions, SerializedError } from './types.js';

const scope = globalThis as unknown as DedicatedWorkerGlobalScope;
let engine: Engine | null = null;
const controllers = new Map<number, AbortController>();
scope.onmessage = event => {
  const { id, method, args, cancel } = event.data as { id: number; method: string; args: unknown[]; cancel?: number };
  if (cancel !== undefined) { controllers.get(cancel)?.abort(); return; }
  if (!Number.isSafeInteger(id) || !Array.isArray(args) || typeof method !== 'string') return;
  const controller = new AbortController();
  controllers.set(id, controller);
  const execute = async () => {
    if (method === 'open') {
      if (engine) throw new Error('Worker already owns a database.');
      engine = await Engine.open(args[0] as string, args[1] as OpenOptions);
      return;
    }
    if (!engine) throw new Error('Database worker has not initialized.');
    return engine.call(method, args, { signal: controller.signal, onProgress: progress => scope.postMessage({ id, progress }) });
  };
  execute().then(result => {
    const buffers = new Set<ArrayBuffer>();
    const collect = (value: unknown, depth = 0) => {
      if (depth > 8 || value === null || typeof value !== 'object') return;
      if (value instanceof Float32Array) { if (value.buffer instanceof ArrayBuffer) buffers.add(value.buffer); return; }
      if (Array.isArray(value)) for (const entry of value) collect(entry, depth + 1);
      else for (const [key, entry] of Object.entries(value)) if (key === 'vector' || key === 'results') collect(entry, depth + 1);
    };
    collect(result);
    scope.postMessage({ id, result }, Array.from(buffers));
  }).catch(error => {
    const value: SerializedError = { name: error?.name ?? 'Error', message: error?.message ?? String(error), ...(error?.code ? { code: error.code } : {}) };
    try { scope.postMessage({ id, error: value }); } catch { scope.close(); }
  }).finally(() => controllers.delete(id));
};
