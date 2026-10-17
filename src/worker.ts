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
  execute().then(result => scope.postMessage({ id, result }), error => {
    const value: SerializedError = { name: error?.name ?? 'Error', message: error?.message ?? String(error), ...(error?.code ? { code: error.code } : {}) };
    scope.postMessage({ id, error: value });
  }).finally(() => controllers.delete(id));
};
