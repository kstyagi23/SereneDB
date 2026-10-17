import { LocalConnection, WorkerConnection } from './rpc.js';
import type { Connection } from './rpc.js';
import { checkAbort, SereneDBError } from './errors.js';
import type { CollectionOptions, CollectionState, DatabaseExport, DocumentInput, Metadata, OpenOptions, OperationOptions, SearchOptions, SearchResponse, StoredRecord } from './types.js';
export type * from './types.js';
export { SereneDBError } from './errors.js';
export { chunkDocument } from './chunking.js';

export class Collection {
  constructor(private connection: Connection, public readonly name: string, public readonly generation: string) {}
  private call<T>(method: string, args: unknown[] = [], options: OperationOptions = {}): Promise<T> { return this.connection.call<T>(method, [this.name, this.generation, ...args], options); }
  upsert(documents: DocumentInput | DocumentInput[], options: OperationOptions = {}): Promise<{ count: number; upserted: number; revision: number; generation: string }> {
    try {
      checkAbort(options.signal);
      const inputs = Array.isArray(documents) ? documents : [documents];
      if (inputs.length > 100000) throw new SereneDBError('Atomic batch has too many documents. Use ingest() for streaming.', 'BATCH_TOO_LARGE');
      let bytes = 0;
      for (const item of inputs) {
        bytes += (item?.text?.length ?? 0) * 2 + (item?.vector?.length ?? 0) * 4;
        if (bytes > 64 * 1024 * 1024) throw new SereneDBError('Atomic batch exceeds 64 MiB. Use ingest() for streaming.', 'BATCH_TOO_LARGE');
      }
      return this.call('upsert', [structuredClone(inputs)], options);
    } catch (error) { return Promise.reject(error); }
  }
  search(options: SearchOptions, operation: OperationOptions = {}): Promise<SearchResponse> {
    try {
      const { signal, ...query } = options;
      return this.call('search', [structuredClone(query)], { ...operation, signal: signal ?? operation.signal });
    } catch (error) { return Promise.reject(error); }
  }
  get(id: string, options: OperationOptions = {}): Promise<StoredRecord | null> { return this.call('get', [id], options); }
  async has(id: string): Promise<boolean> { return Boolean(await this.get(id)); }
  delete(ids: string | string[], options: OperationOptions = {}): Promise<{ deleted: number; count: number; revision: number }> { return this.call('delete', [Array.isArray(ids) ? [...ids] : [ids]], options); }
  clear(options: OperationOptions = {}): Promise<{ deleted: number; count: number; revision: number }> { return this.call('clear', [], options); }
  state(options: OperationOptions = {}): Promise<CollectionState> { return this.call('state', [], options); }
  async count(): Promise<number> { return (await this.state()).count; }
  scan(options: { offset?: number; limit?: number; includeVectors?: boolean; filter?: SearchOptions['filter'] } = {}, operation: OperationOptions = {}): Promise<Array<Omit<StoredRecord, 'vector'> & { vector?: Float32Array }>> { return this.call('scan', [structuredClone(options)], operation); }
  async *iterate(options: { batchSize?: number; filter?: SearchOptions['filter']; signal?: AbortSignal } = {}): AsyncGenerator<StoredRecord> {
    const size = options.batchSize ?? 256;
    if (!Number.isInteger(size) || size < 1 || size > 10000) throw new RangeError('batchSize must be between 1 and 10000.');
    let offset = 0;
    while (true) {
      const records = await this.scan({ offset, limit: size, includeVectors: true, filter: options.filter }, { signal: options.signal });
      for (const record of records) yield record as StoredRecord;
      if (records.length < size) return;
      offset += size;
    }
  }
  async addTexts(documents: Array<{ id: string; text: string; metadata?: Metadata }>, options: OperationOptions = {}): Promise<unknown> { return this.upsert(documents, options); }
  async ingest(source: AsyncIterable<DocumentInput> | Iterable<DocumentInput>, options: OperationOptions & { batchSize?: number } = {}): Promise<{ upserted: number }> {
    const batchSize = options.batchSize ?? 128;
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 10000) throw new RangeError('batchSize must be between 1 and 10000.');
    let batch: DocumentInput[] = [], upserted = 0;
    for await (const item of source) {
      if (options.signal?.aborted) throw new DOMException('Ingestion aborted; earlier batches remain committed.', 'AbortError');
      batch.push(item);
      if (batch.length === batchSize) { upserted += (await this.upsert(batch, options)).upserted; batch = []; }
    }
    if (batch.length) upserted += (await this.upsert(batch, options)).upserted;
    return { upserted };
  }
  preload(options: OperationOptions = {}): Promise<unknown> { return this.call('preload', [], options); }
  checkpoint(options: OperationOptions = {}): Promise<boolean> { return this.call('checkpoint', [], options); }
  compact(options: OperationOptions = {}): Promise<unknown> { return this.call('compact', [], options); }
  stats(options: OperationOptions = {}): Promise<unknown> { return this.call('stats', [], options); }
  unload(): Promise<void> { return this.call('unload'); }
  drop(): Promise<void> { return this.call('drop'); }
}

export class SereneDB {
  private constructor(private connection: Connection, public readonly name: string) {}
  static async open(name = 'SereneDB', options: OpenOptions = {}): Promise<SereneDB> {
    const worker = options.worker ?? typeof Worker !== 'undefined';
    if (worker && typeof Worker === 'undefined') throw new Error('Workers are unavailable; explicitly pass worker: false for same-thread use.');
    return new SereneDB(worker ? await WorkerConnection.open(name, options) : await LocalConnection.open(name, options), name);
  }
  async collection(name = 'default', options: CollectionOptions = {}, operation: OperationOptions = {}): Promise<Collection> {
    const state = await this.connection.call<CollectionState>('collection', [name, structuredClone(options)], operation);
    return new Collection(this.connection, name, state.generation);
  }
  listCollections(): Promise<CollectionState[]> { return this.connection.call('list'); }
  export(options: OperationOptions = {}): Promise<DatabaseExport> { return this.connection.call('export', [], options); }
  import(data: DatabaseExport, options: OperationOptions & { replace?: boolean } = {}): Promise<{ collections: number }> { return this.connection.call('import', [structuredClone(data), options.replace ?? false], options); }
  async close(): Promise<void> { try { await this.connection.call('close'); } finally { this.connection.terminate(); } }
  async destroy(options: OperationOptions = {}): Promise<void> { try { await this.connection.call('destroy', [], options); } finally { this.connection.terminate(); } }
  static async storageInfo(): Promise<{ usage?: number; quota?: number; persistent: boolean }> {
    if (typeof navigator === 'undefined' || !navigator.storage) return { persistent: false };
    return { ...await navigator.storage.estimate(), persistent: await navigator.storage.persisted() };
  }
  static async requestPersistence(): Promise<boolean> { return typeof navigator !== 'undefined' && Boolean(await navigator.storage?.persist?.()); }
}
