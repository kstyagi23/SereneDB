import { Engine } from './engine.js';
import { checkAbort, SereneDBError } from './errors.js';
import type { OpenOptions, OperationOptions, ProgressEvent, SerializedError } from './types.js';

export interface Connection {
  call<T>(method: string, args?: unknown[], options?: OperationOptions): Promise<T>;
  terminate(): void;
}

export class LocalConnection implements Connection {
  private constructor(private engine: Engine) {}
  static async open(name: string, options: OpenOptions): Promise<LocalConnection> { return new LocalConnection(await Engine.open(name, options)); }
  call<T>(method: string, args: unknown[] = [], options: OperationOptions = {}): Promise<T> { return this.engine.call(method, args, options) as Promise<T>; }
  terminate(): void {}
}

interface Pending {
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
  options: OperationOptions;
  abort?: () => void;
}

export class WorkerConnection implements Connection {
  private nextId = 0;
  private pending = new Map<number, Pending>();
  private closed = false;
  private constructor(private worker: Worker) {
    worker.onmessage = event => {
      const message = event.data as { id: number; result?: unknown; error?: SerializedError; progress?: ProgressEvent };
      const pending = this.pending.get(message.id);
      if (!pending) return;
      if (message.progress) { try { pending.options.onProgress?.(message.progress); } catch { /* Progress is an observer. */ } return; }
      this.pending.delete(message.id);
      if (pending.abort) pending.options.signal?.removeEventListener('abort', pending.abort);
      if (message.error) {
        const error = message.error.name === 'AbortError' ? new DOMException(message.error.message, 'AbortError') : new SereneDBError(message.error.message, message.error.code ?? 'WORKER_OPERATION');
        pending.reject(error);
      } else pending.resolve(message.result);
    };
    worker.onerror = event => { event.preventDefault(); this.fail(new SereneDBError(`Database worker failed: ${event.message}. A write interrupted by a crash may have committed; reopen and inspect its IDs.`, 'WORKER_FAILED')); };
    worker.onmessageerror = () => this.fail(new SereneDBError('Database worker message could not be decoded.', 'WORKER_FAILED'));
  }

  static async open(name: string, options: OpenOptions): Promise<WorkerConnection> {
    const worker = new Worker(options.workerUrl ?? new URL('./worker.js', import.meta.url), { type: 'module', name: `SereneDB:${name}` });
    const connection = new WorkerConnection(worker);
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        connection.call('open', [name, { ...options, workerUrl: undefined, worker: false }]),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new SereneDBError('Worker initialization timed out. Check worker URLs and Content Security Policy.', 'WORKER_FAILED')), (options.blockedTimeoutMs ?? 5000) + 15000); }),
      ]);
      return connection;
    } catch (error) { connection.terminate(); throw error; }
    finally { clearTimeout(timer); }
  }

  call<T>(method: string, args: unknown[] = [], options: OperationOptions = {}): Promise<T> {
    if (this.closed) return Promise.reject(new SereneDBError('Database connection is closed.', 'CLOSED'));
    if (this.pending.size >= 128) return Promise.reject(new SereneDBError('Database queue is full.', 'BACKPRESSURE'));
    try { checkAbort(options.signal); } catch (error) { return Promise.reject(error); }
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const pending: Pending = { resolve: resolve as (value: unknown) => void, reject, options };
      if (options.signal) {
        pending.abort = () => { if (!this.closed) this.worker.postMessage({ cancel: id }); };
        options.signal.addEventListener('abort', pending.abort, { once: true });
      }
      this.pending.set(id, pending);
      const buffers = new Set<ArrayBuffer>();
      const collect = (value: unknown, depth = 0) => {
        if (depth > 8 || value === null || typeof value !== 'object') return;
        if (value instanceof Float32Array || value instanceof Float64Array) { if (value.buffer instanceof ArrayBuffer) buffers.add(value.buffer); return; }
        if (Array.isArray(value)) { for (const entry of value) collect(entry, depth + 1); }
        else if ('vector' in value) collect((value as { vector: unknown }).vector, depth + 1);
      };
      // Collection owns its cloned input buffers, so transfer never detaches caller data.
      if (method === 'upsert' || method === 'search') collect(args);
      try { this.worker.postMessage({ id, method, args }, Array.from(buffers)); }
      catch (error) { this.pending.delete(id); if (pending.abort) options.signal?.removeEventListener('abort', pending.abort); reject(error); }
    });
  }

  private fail(error: Error): void {
    this.closed = true;
    for (const pending of this.pending.values()) { if (pending.abort) pending.options.signal?.removeEventListener('abort', pending.abort); pending.reject(error); }
    this.pending.clear(); this.worker.terminate();
  }
  terminate(): void { this.fail(new SereneDBError('Database connection is closed.', 'CLOSED')); }
}
