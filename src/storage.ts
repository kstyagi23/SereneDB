import { checkAbort, invalid, SereneDBError } from './errors.js';
import type { CollectionSchema, CollectionState, IndexSnapshot, Mutation, MutationResult, OpenOptions, PersistedIndex, StoredRecord, TransformersConfig } from './types.js';
import { canonical, collectionSchema, embeddingFingerprint, integer, prepareMetadata, validName } from './validation.js';

const VERSION = 2;
const SYSTEM = 'system';
const COLLECTIONS = 'collections';
const RECORDS = 'records';
const INDEXES = 'indexes';
const ROOT = 'root';

interface RootState {
  key: typeof ROOT;
  id: string;
  generation: string;
  status: 'active' | 'destroying';
}

interface IndexRow extends PersistedIndex {
  collection: string;
}

export interface EnsureCollectionOptions {
  explicit?: Partial<Record<'dimension' | 'metric' | 'embedding' | 'index', boolean>>;
  signal?: AbortSignal;
  expectedGeneration?: string | null;
}

interface TransactionContext<T> {
  store(name: string): IDBObjectStore;
  watch<V>(request: IDBRequest<V>, success: (value: V) => void): void;
  done(value: T): void;
}

function identifier(): string {
  if (typeof globalThis.crypto?.randomUUID === 'function') return globalThis.crypto.randomUUID();
  if (typeof globalThis.crypto?.getRandomValues !== 'function') throw new SereneDBError('Secure random identifiers are unavailable in this environment.', 'UNSUPPORTED_ENVIRONMENT');
  return Array.from(globalThis.crypto.getRandomValues(new Uint8Array(16)), byte => byte.toString(16).padStart(2, '0')).join('');
}

function collectionRange(name: string): IDBKeyRange {
  // String IDs sort below array keys; this also covers every Unicode string ID.
  return IDBKeyRange.bound([name], [name, []], false, true);
}

function errorName(error: unknown): string | undefined {
  return error && typeof error === 'object' && 'name' in error ? String(error.name) : undefined;
}

function validEmbedding(config: TransformersConfig): boolean {
  return !!config && config.provider === 'transformers' && typeof config.model === 'string' && !!config.model.trim()
    && typeof config.revision === 'string' && !!config.revision.trim() && typeof config.dtype === 'string' && !!config.dtype.trim()
    && ['auto', 'wasm', 'webgpu'].includes(config.device) && ['mean', 'cls', 'last_token'].includes(config.pooling)
    && typeof config.normalize === 'boolean' && typeof config.queryPrefix === 'string' && typeof config.documentPrefix === 'string'
    && Number.isSafeInteger(config.maxTokens) && config.maxTokens > 0 && Number.isSafeInteger(config.batchSize) && config.batchSize > 0
    && Number.isSafeInteger(config.batchTokens) && config.batchTokens >= config.maxTokens && typeof config.allowRemoteModels === 'boolean'
    && (config.localModelPath === undefined || typeof config.localModelPath === 'string') && (config.wasmPath === undefined || typeof config.wasmPath === 'string');
}

function stateFrom(value: unknown, name?: string): CollectionState {
  if (value === undefined && name !== undefined) throw new SereneDBError(`Collection ${JSON.stringify(name)} does not exist.`, 'COLLECTION_NOT_FOUND');
  const state = value as CollectionState | undefined;
  if (!state || typeof state.name !== 'string' || !state.name.trim() || state.name.length > 512 || (name !== undefined && state.name !== name)
    || typeof state.generation !== 'string' || !state.generation
    || !Number.isSafeInteger(state.count) || state.count < 0 || !Number.isSafeInteger(state.revision) || state.revision < 0
    || !Number.isFinite(state.createdAt) || !Number.isFinite(state.updatedAt) || !state.schema
    || !['cosine', 'dot', 'euclidean'].includes(state.schema.metric)
    || (state.schema.dimension !== null && (!Number.isInteger(state.schema.dimension) || state.schema.dimension < 1 || state.schema.dimension > 16384))
    || (state.schema.dimension === null && state.count !== 0) || !state.schema.index || !['hnsw', 'flat'].includes(state.schema.index.type)
    || !Number.isInteger(state.schema.index.m) || state.schema.index.m < 2 || state.schema.index.m > 64
    || !Number.isInteger(state.schema.index.efConstruction) || state.schema.index.efConstruction < state.schema.index.m || state.schema.index.efConstruction > 4096
    || !Number.isInteger(state.schema.index.efSearch) || state.schema.index.efSearch < 1 || state.schema.index.efSearch > 65536
    || !Number.isInteger(state.schema.index.annThreshold) || state.schema.index.annThreshold < 0 || state.schema.index.annThreshold > 10000000
    || (state.schema.embedding !== null && (!state.schema.embedding || typeof state.schema.embedding.fingerprint !== 'string' || !state.schema.embedding.fingerprint || !validEmbedding(state.schema.embedding.config)))) {
    throw new SereneDBError('Persisted collection state is invalid; no cached or emergency data will be substituted.', 'STORAGE_CORRUPT');
  }
  try {
    if (state.schema.embedding && state.schema.embedding.fingerprint !== embeddingFingerprint(state.schema.embedding.config)) throw new Error('Embedding fingerprint mismatch');
  } catch (error) { throw new SereneDBError('Persisted embedding schema is invalid.', 'STORAGE_CORRUPT', { cause: error }); }
  return state;
}

function recordFrom(value: unknown, state: CollectionState, id?: string): StoredRecord | undefined {
  if (value === undefined) return undefined;
  const record = value as StoredRecord;
  if (!record || record.collection !== state.name || typeof record.id !== 'string' || !record.id.trim() || record.id.length > 512
    || (id !== undefined && record.id !== id) || !(record.vector instanceof Float32Array)
    || state.schema.dimension === null || record.vector.length !== state.schema.dimension
    || !Number.isFinite(record.timestamp) || (record.text !== undefined && typeof record.text !== 'string')
    || !record.metadata || typeof record.metadata !== 'object' || Array.isArray(record.metadata)
    || (Object.getPrototypeOf(record.metadata) !== Object.prototype && Object.getPrototypeOf(record.metadata) !== null)) {
    throw new SereneDBError('Persisted vector record does not match its collection schema.', 'STORAGE_CORRUPT');
  }
  let nonzero = false;
  for (const value of record.vector) {
    if (!Number.isFinite(value)) throw new SereneDBError('Persisted vector contains a nonfinite float32 value.', 'STORAGE_CORRUPT');
    nonzero ||= value !== 0;
  }
  if (state.schema.metric === 'cosine' && !nonzero) throw new SereneDBError('Persisted cosine vector has zero norm.', 'STORAGE_CORRUPT');
  return record;
}

function checkSchema(tx: IDBTransaction): void {
  const records = tx.objectStore(RECORDS);
  if (tx.objectStore(SYSTEM).keyPath !== 'key' || tx.objectStore(COLLECTIONS).keyPath !== 'name'
    || tx.objectStore(INDEXES).keyPath !== 'collection' || canonical(records.keyPath) !== canonical(['collection', 'id'])
    || !records.indexNames.contains('collection') || [SYSTEM, COLLECTIONS, RECORDS, INDEXES].some(name => tx.objectStore(name).autoIncrement)) {
    throw new SereneDBError('The IndexedDB stores do not match SereneDB version 2.', 'STORAGE_CORRUPT');
  }
  const index = records.index('collection');
  if (index.keyPath !== 'collection' || index.unique || index.multiEntry) throw new SereneDBError('The IndexedDB collection index is invalid.', 'STORAGE_CORRUPT');
}

/** IndexedDB is the authority; returned state and records are never a write-back cache. */
export class Storage {
  private db: IDBDatabase | null;
  private rootId = '';
  private rootGeneration = '';
  private lifecycle: 'open' | 'destroying' | 'closed' = 'open';
  private closedError = new SereneDBError('Database is closed. Open a new Storage connection to use it.', 'DATABASE_CLOSED');
  private readonly active = new Map<IDBTransaction, (error: unknown) => boolean>();
  private durabilityOptions = true;
  private readonly durability: 'strict' | 'relaxed';
  private readonly blockedTimeoutMs: number;

  private constructor(private readonly databaseName: string, db: IDBDatabase, options: OpenOptions) {
    this.db = db;
    this.durability = options.durability ?? 'strict';
    this.blockedTimeoutMs = options.blockedTimeoutMs ?? 5000;
    db.onversionchange = () => this.closeWith(new SereneDBError('Database connection closed after an IndexedDB version change or deletion. Reopen the database.', 'DATABASE_CLOSED'));
    db.onclose = () => this.closeWith(new SereneDBError('IndexedDB closed the database connection unexpectedly. Reopen the database before continuing.', 'DATABASE_CLOSED'));
  }

  get name(): string { return this.databaseName; }
  /** Database-root generation. Mutation/checkpoint generation arguments are collection generations. */
  get generation(): string { return this.rootGeneration; }

  static async open(name: string, options: OpenOptions = {}): Promise<Storage> {
    validName(name, 'Database name');
    options = { ...options };
    if (options.durability !== undefined && options.durability !== 'strict' && options.durability !== 'relaxed') invalid('Durability must be strict or relaxed.');
    if (options.blockedTimeoutMs !== undefined) integer(options.blockedTimeoutMs, 'Blocked timeout', 1, 2147483647);
    if (!globalThis.indexedDB) throw new SereneDBError('IndexedDB is unavailable in this environment.', 'UNSUPPORTED_ENVIRONMENT');
    return new Promise<Storage>((resolve, reject) => {
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      let upgradeError: unknown;
      const request = globalThis.indexedDB.open(name, VERSION);
      const clearTimer = () => { if (timer !== undefined) clearTimeout(timer); timer = undefined; };
      const fail = (error: unknown) => {
        clearTimer();
        if (!settled) { settled = true; reject(error); }
      };
      request.onblocked = () => {
        if (!settled && timer === undefined) timer = setTimeout(() => fail(new SereneDBError('Opening the database is blocked by another connection. Close older connections and retry.', 'STORAGE_BLOCKED')), options.blockedTimeoutMs ?? 5000);
      };
      request.onupgradeneeded = event => {
        clearTimer();
        const tx = request.transaction!;
        if (settled) { tx.abort(); return; }
        try {
          if (event.oldVersion !== 0) throw new SereneDBError('Legacy IndexedDB data requires an explicit export and import into a new version 2 database. The original stores and records have not been changed.', 'LEGACY_DATA');
          const db = request.result;
          const system = db.createObjectStore(SYSTEM, { keyPath: 'key' });
          db.createObjectStore(COLLECTIONS, { keyPath: 'name' });
          db.createObjectStore(RECORDS, { keyPath: ['collection', 'id'] }).createIndex('collection', 'collection', { unique: false });
          db.createObjectStore(INDEXES, { keyPath: 'collection' });
          system.put({ key: ROOT, id: identifier(), generation: identifier(), status: 'active' } satisfies RootState);
        } catch (error) {
          upgradeError = error;
          tx.abort();
        }
      };
      request.onerror = () => fail(upgradeError ?? request.error ?? new SereneDBError('IndexedDB could not open the database.', 'STORAGE_ERROR'));
      request.onsuccess = () => {
        clearTimer();
        const db = request.result;
        if (settled) { db.close(); return; }
        for (const store of [SYSTEM, COLLECTIONS, RECORDS, INDEXES]) {
          if (!db.objectStoreNames.contains(store)) {
            db.close();
            fail(new SereneDBError(`IndexedDB store ${JSON.stringify(store)} is missing. Existing data has not been replaced.`, 'STORAGE_CORRUPT'));
            return;
          }
        }
        const storage = new Storage(name, db, options);
        void storage.transaction<void>([COLLECTIONS, RECORDS, INDEXES], 'readonly', context => {
          checkSchema(context.store(SYSTEM).transaction);
          context.done(undefined);
        }).then(() => {
          if (settled) { storage.close(); return; }
          settled = true;
          resolve(storage);
        }, error => { storage.close(); fail(error); });
      };
    });
  }

  private connection(allowDestroying = false): IDBDatabase {
    if (!this.db || this.lifecycle === 'closed') throw this.closedError;
    if (this.lifecycle === 'destroying' && !allowDestroying) throw new SereneDBError('Database destruction is in progress.', 'DATABASE_DESTROYING');
    return this.db;
  }

  private closeWith(error: SereneDBError): void {
    if (this.lifecycle === 'closed') return;
    this.lifecycle = 'closed';
    this.closedError = error;
    const db = this.db;
    this.db = null;
    for (const abort of this.active.values()) abort(error);
    db?.close();
  }

  private transaction<T>(stores: string[], mode: IDBTransactionMode, enqueue: (context: TransactionContext<T>) => void, signal?: AbortSignal, allowDestroying = false): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      let tx: IDBTransaction;
      try {
        checkAbort(signal);
        const db = this.connection(allowDestroying);
        const scope = Array.from(new Set([SYSTEM, ...stores]));
        if (mode === 'readwrite' && this.durabilityOptions) {
          try { tx = db.transaction(scope, mode, { durability: this.durability }); }
          catch (error) {
            if (errorName(error) !== 'TypeError' && errorName(error) !== 'NotSupportedError') throw error;
            tx = db.transaction(scope, mode);
            this.durabilityOptions = false;
          }
        } else tx = db.transaction(scope, mode);
      } catch (error) {
        if (errorName(error) === 'InvalidStateError') {
          const closed = new SereneDBError('IndexedDB connection is no longer open. Reopen the database before continuing.', 'DATABASE_CLOSED', { cause: error });
          this.closeWith(closed);
          reject(closed);
        } else reject(error);
        return;
      }
      let failure: unknown;
      let value: T;
      let ready = false;
      const cancel = (error: unknown): boolean => {
        // Cancellation is too late once IDB starts committing. Preserve its acknowledgement.
        try { tx.abort(); } catch { return false; }
        failure ??= error;
        return true;
      };
      const fail = (error: unknown) => { failure ??= error; cancel(error); };
      const onAbort = () => cancel(new DOMException('Operation aborted.', 'AbortError'));
      const cleanup = () => { this.active.delete(tx); signal?.removeEventListener('abort', onAbort); };
      const context: TransactionContext<T> = {
        store: name => tx.objectStore(name),
        done: result => { value = result; ready = true; },
        watch: (request, success) => {
          request.onsuccess = () => {
            if (failure !== undefined) return;
            try { success(request.result); } catch (error) { fail(error); }
          };
        },
      };
      this.active.set(tx, cancel);
      tx.onerror = event => fail((event.target as IDBRequest).error ?? tx.error ?? new SereneDBError('IndexedDB request failed.', 'STORAGE_ERROR'));
      tx.onabort = () => { cleanup(); reject(failure ?? tx.error ?? new DOMException('IndexedDB transaction aborted.', 'AbortError')); };
      tx.oncomplete = () => {
        cleanup();
        if (failure !== undefined) reject(failure);
        else if (!ready) reject(new SereneDBError('IndexedDB transaction completed without a result.', 'STORAGE_ERROR'));
        else resolve(value);
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      if (signal?.aborted) { onAbort(); return; }
      try {
        context.watch(context.store(SYSTEM).get(ROOT), (root: RootState | undefined) => {
          if (!root || root.key !== ROOT || typeof root.id !== 'string' || !root.id || typeof root.generation !== 'string' || !root.generation || (root.status !== 'active' && root.status !== 'destroying')) {
            const error = new SereneDBError('Persisted database identity is missing or invalid; automatic recovery is not permitted.', 'STORAGE_CORRUPT');
            this.closeWith(error);
            throw error;
          }
          if (root.status !== 'active') {
            const error = new SereneDBError('The persisted database has been marked for destruction. Its pending deletion must finish before it can be reopened.', 'DATABASE_DESTROYING');
            this.closeWith(error);
            throw error;
          }
          if (!this.rootId) { this.rootId = root.id; this.rootGeneration = root.generation; }
          else if (root.id !== this.rootId || root.generation !== this.rootGeneration) {
            const error = new SereneDBError('Database generation changed; this connection is stale. Open a new connection.', 'DATABASE_STALE');
            this.closeWith(error);
            throw error;
          }
          enqueue(context);
        });
      } catch (error) { fail(error); }
    });
  }

  async listCollections(signal?: AbortSignal): Promise<CollectionState[]> {
    return this.transaction<CollectionState[]>([COLLECTIONS], 'readonly', context => {
      context.watch(context.store(COLLECTIONS).getAll(), values => context.done(values.map(value => stateFrom(value))));
    }, signal);
  }

  async ensureCollection(name: string, schema: CollectionSchema, options: EnsureCollectionOptions = {}): Promise<CollectionState> {
    validName(name, 'Collection name');
    const signal = options.signal;
    const fields = options.explicit === undefined ? undefined : { ...options.explicit };
    const expectedGeneration = options.expectedGeneration;
    checkAbort(signal);
    if (!schema || !['cosine', 'dot', 'euclidean'].includes(schema.metric) || schema.dimension === undefined || !schema.index || (schema.embedding !== null && (!schema.embedding || !validEmbedding(schema.embedding.config)))) invalid('A complete, valid collection schema is required.');
    const normalized = collectionSchema({ dimension: schema.dimension ?? undefined, metric: schema.metric, index: schema.index, embedder: schema.embedding?.config });
    if (schema.embedding && schema.embedding.fingerprint !== normalized.embedding?.fingerprint) invalid('Embedding fingerprint does not match its configuration.', 'SCHEMA_MISMATCH');
    const explicit = (key: 'dimension' | 'metric' | 'embedding' | 'index') => fields === undefined || fields[key] === true;
    return this.transaction<CollectionState>([COLLECTIONS, INDEXES], 'readwrite', context => {
      const collections = context.store(COLLECTIONS);
      context.watch(collections.get(name), value => {
        const now = Date.now();
        if (value === undefined) {
          if (typeof expectedGeneration === 'string') throw new SereneDBError('Import destination was dropped after validation.', 'COLLECTION_STALE');
          const state: CollectionState = { name, generation: identifier(), schema: normalized, revision: 0, count: 0, createdAt: now, updatedAt: now };
          collections.add(state);
          context.done(state);
          return;
        }
        const state = stateFrom(value, name);
        if (expectedGeneration === null || typeof expectedGeneration === 'string' && state.generation !== expectedGeneration) throw new SereneDBError('Import destination was replaced after validation.', 'COLLECTION_STALE');
        if ((explicit('metric') && normalized.metric !== state.schema.metric)
          || (explicit('embedding') && (normalized.embedding?.fingerprint ?? null) !== (state.schema.embedding?.fingerprint ?? null))
          || (explicit('index') && canonical(normalized.index) !== canonical(state.schema.index))
          || (explicit('dimension') && normalized.dimension !== null && state.schema.dimension !== null && normalized.dimension !== state.schema.dimension)) {
          throw new SereneDBError(`Explicit schema options do not match collection ${JSON.stringify(name)}.`, 'SCHEMA_MISMATCH');
        }
        if (explicit('dimension') && normalized.dimension !== null && state.schema.dimension === null) {
          state.schema.dimension = normalized.dimension;
          state.revision += 1;
          if (!Number.isSafeInteger(state.revision)) throw new SereneDBError('Collection revision is exhausted.', 'STORAGE_ERROR');
          state.updatedAt = now;
          collections.put(state);
          context.store(INDEXES).delete(name);
        }
        context.done(state);
      });
    }, signal);
  }

  async refreshState(name: string, signal?: AbortSignal): Promise<CollectionState> {
    validName(name, 'Collection name');
    return this.transaction<CollectionState>([COLLECTIONS], 'readonly', context => {
      context.watch(context.store(COLLECTIONS).get(name), value => context.done(stateFrom(value, name)));
    }, signal);
  }

  async readCollection(name: string, signal?: AbortSignal): Promise<{ state: CollectionState; records: StoredRecord[]; snapshot?: PersistedIndex }> {
    validName(name, 'Collection name');
    return this.transaction([COLLECTIONS, RECORDS, INDEXES], 'readonly', context => {
      let state: CollectionState;
      let records: StoredRecord[];
      let index: IndexRow | undefined;
      let pending = 3;
      const finish = () => {
        if (--pending !== 0) return;
        if (records.length !== state.count) throw new SereneDBError('Persisted record count does not match collection state.', 'STORAGE_CORRUPT');
        for (const record of records) recordFrom(record, state);
        const snapshot = index?.generation === state.generation && index.revision === state.revision
          ? { generation: index.generation, revision: index.revision, payload: index.payload } : undefined;
        context.done({ state, records, ...(snapshot ? { snapshot } : {}) });
      };
      context.watch(context.store(COLLECTIONS).get(name), value => { state = stateFrom(value, name); finish(); });
      context.watch(context.store(RECORDS).index('collection').getAll(name), value => { records = value as StoredRecord[]; finish(); });
      context.watch(context.store(INDEXES).get(name), value => { index = value as IndexRow | undefined; finish(); });
    }, signal);
  }

  async get(name: string, id: string, signal?: AbortSignal, expectedGeneration?: string): Promise<StoredRecord | undefined> {
    validName(name, 'Collection name');
    validName(id, 'Record ID');
    return this.transaction<StoredRecord | undefined>([COLLECTIONS, RECORDS], 'readonly', context => {
      context.watch(context.store(COLLECTIONS).get(name), value => {
        const state = stateFrom(value, name);
        this.checkCollectionGeneration(state, expectedGeneration);
        context.watch(context.store(RECORDS).get([name, id]), record => context.done(recordFrom(record, state, id)));
      });
    }, signal);
  }

  /** Deletes happen before upserts; duplicate upsert IDs use their last supplied record. */
  async mutate(name: string, mutation: Mutation, expectedGeneration?: string, signal?: AbortSignal): Promise<MutationResult> {
    validName(name, 'Collection name');
    checkAbort(signal);
    if (!mutation || typeof mutation !== 'object') invalid('Mutation must be an object.');
    if (mutation.upserts !== undefined && !Array.isArray(mutation.upserts)) invalid('Upserts must be an array.');
    if (mutation.deleteIds !== undefined && !Array.isArray(mutation.deleteIds)) invalid('Delete IDs must be an array.');
    if (mutation.clear !== undefined && typeof mutation.clear !== 'boolean') invalid('Clear must be a boolean.');
    const clear = mutation.clear === true;
    const upserts = new Map<string, StoredRecord>();
    const deleteIds = new Set<string>();
    let dimension: number | null = null;
    let zeroVector = false;
    const now = Date.now();
    for (const input of mutation.upserts ?? []) {
      if (!input || typeof input !== 'object') invalid('Each upsert must be a prepared record.');
      const { id, vector: source, text, timestamp, metadata } = input;
      validName(id, 'Record ID');
      if (!(source instanceof Float32Array)) invalid('Prepared vectors must be Float32Array values.');
      integer(source.length, 'Vector dimension', 1, 16384);
      const vector = new Float32Array(source);
      integer(vector.length, 'Vector dimension', 1, 16384);
      if (dimension !== null && dimension !== vector.length) invalid('All vectors in a mutation must have the same dimension.', 'SCHEMA_MISMATCH');
      dimension = vector.length;
      let nonzero = false;
      for (const value of vector) {
        if (!Number.isFinite(value)) invalid('Prepared vectors must contain only finite float32 values.');
        nonzero ||= value !== 0;
      }
      zeroVector ||= !nonzero;
      if (text !== undefined && typeof text !== 'string') invalid('Record text must be a string.');
      if (timestamp !== undefined && (typeof timestamp !== 'number' || !Number.isFinite(timestamp))) invalid('Record timestamp must be finite.');
      upserts.set(id, {
        collection: name, id, vector, metadata: prepareMetadata(metadata),
        timestamp: timestamp ?? now, ...(text !== undefined ? { text } : {}),
      });
    }
    for (const id of mutation.deleteIds ?? []) { validName(id, 'Record ID'); deleteIds.add(id); }
    return this.transaction<MutationResult>([COLLECTIONS, RECORDS, INDEXES], 'readwrite', context => {
      const collections = context.store(COLLECTIONS);
      const records = context.store(RECORDS);
      context.watch(collections.get(name), value => {
        const state = stateFrom(value, name);
        this.checkCollectionGeneration(state, expectedGeneration);
        if (dimension !== null && state.schema.dimension !== null && dimension !== state.schema.dimension) throw new SereneDBError(`Vector dimension mismatch: expected ${state.schema.dimension}, received ${dimension}.`, 'SCHEMA_MISMATCH');
        if (state.schema.metric === 'cosine' && zeroVector) invalid('Cosine vectors must have a nonzero norm.');
        const existing = new Set<string>();
        const apply = () => {
          let count = clear ? 0 : state.count;
          let deleted = clear ? state.count : 0;
          if (clear) records.delete(collectionRange(name));
          else for (const id of deleteIds) {
            if (existing.has(id)) { records.delete([name, id]); count -= 1; deleted += 1; }
          }
          for (const [id, record] of upserts) {
            if (clear || !existing.has(id) || deleteIds.has(id)) count += 1;
            records.put(record);
          }
          if (deleted > 0 || upserts.size > 0) {
            if (!Number.isSafeInteger(count) || count < 0 || !Number.isSafeInteger(state.revision + 1)) throw new SereneDBError('Collection count or revision is invalid or exhausted.', 'STORAGE_CORRUPT');
            if (state.schema.dimension === null && dimension !== null) state.schema.dimension = dimension;
            state.count = count;
            state.revision += 1;
            state.updatedAt = Math.max(state.updatedAt, Date.now());
            collections.put(state);
            context.store(INDEXES).delete(name);
          }
          context.done({ state, upserted: Array.from(upserts.values()), deleted });
        };
        if (clear || state.count === 0) { apply(); return; }
        const ids = new Set([...deleteIds, ...upserts.keys()]);
        let pending = ids.size;
        if (pending === 0) { apply(); return; }
        for (const id of ids) context.watch(records.getKey([name, id]), key => {
          if (key !== undefined) existing.add(id);
          if (--pending === 0) apply();
        });
      });
    }, signal);
  }

  private checkCollectionGeneration(state: CollectionState, expected?: string): void {
    if (expected !== undefined && state.generation !== expected) throw new SereneDBError('Collection generation changed; this handle refers to a dropped or replaced collection.', 'COLLECTION_STALE');
  }

  async checkpoint(name: string, generation: string, revision: number, payload: IndexSnapshot, signal?: AbortSignal): Promise<boolean> {
    validName(name, 'Collection name');
    integer(revision, 'Checkpoint revision', 0, Number.MAX_SAFE_INTEGER);
    checkAbort(signal);
    const copy = structuredClone(payload);
    return this.transaction<boolean>([COLLECTIONS, INDEXES], 'readwrite', context => {
      context.watch(context.store(COLLECTIONS).get(name), value => {
        if (value === undefined) { context.done(false); return; }
        const state = stateFrom(value, name);
        if (state.generation !== generation || state.revision !== revision) { context.done(false); return; }
        if (!copy || copy.dimension !== state.schema.dimension || copy.metric !== state.schema.metric || canonical(copy.options) !== canonical(state.schema.index)) throw new SereneDBError('Index checkpoint does not match the collection schema.', 'SCHEMA_MISMATCH');
        context.store(INDEXES).put({ collection: name, generation, revision, payload: copy } satisfies IndexRow);
        context.done(true);
      });
    }, signal);
  }

  async dropCollection(name: string, expectedGeneration?: string, signal?: AbortSignal): Promise<void> {
    validName(name, 'Collection name');
    return this.transaction<void>([COLLECTIONS, RECORDS, INDEXES], 'readwrite', context => {
      context.watch(context.store(COLLECTIONS).get(name), value => {
        if (value === undefined) { context.done(undefined); return; }
        this.checkCollectionGeneration(stateFrom(value, name), expectedGeneration);
        context.store(RECORDS).delete(collectionRange(name));
        context.store(INDEXES).delete(name);
        context.store(COLLECTIONS).delete(name);
        context.done(undefined);
      });
    }, signal);
  }

  /** Closing aborts this connection's active transactions; it never writes cached state. */
  close(): void { this.closeWith(new SereneDBError('Database is closed. Open a new Storage connection to use it.', 'DATABASE_CLOSED')); }

  /**
   * The terminal root/data reset commits before deletion. IndexedDB deletion cannot
   * be canceled: after that commit, abort/timeout only stops waiting and deletion
   * may finish later. This instance stays closed and stale peers cannot write.
   */
  async destroy(signal?: AbortSignal): Promise<void> {
    checkAbort(signal);
    const db = this.connection();
    this.lifecycle = 'destroying';
    for (const abort of this.active.values()) abort(new SereneDBError('Database destruction interrupted the transaction.', 'DATABASE_DESTROYING'));
    try {
      await this.transaction<void>([COLLECTIONS, RECORDS, INDEXES], 'readwrite', context => {
        context.store(SYSTEM).put({ key: ROOT, id: this.rootId, generation: identifier(), status: 'destroying' } satisfies RootState);
        context.store(COLLECTIONS).clear();
        context.store(RECORDS).clear();
        context.store(INDEXES).clear();
        context.done(undefined);
      }, signal, true);
    } catch (error) {
      if (this.db === db && this.lifecycle === 'destroying') this.lifecycle = 'open';
      throw error;
    }
    this.closeWith(new SereneDBError('Database has been destroyed or is awaiting physical deletion.', 'DATABASE_CLOSED'));
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const request = globalThis.indexedDB.deleteDatabase(this.name);
      const cleanup = () => { if (timer !== undefined) clearTimeout(timer); signal?.removeEventListener('abort', onAbort); };
      const finish = (error?: unknown) => {
        cleanup();
        if (settled) return;
        settled = true;
        if (error !== undefined) reject(error); else resolve();
      };
      const onAbort = () => {
        // A complete request wins even if another completion listener aborts the signal.
        if (request.readyState === 'done') finish(request.error ?? undefined);
        else finish(new DOMException('Waiting for database deletion was aborted; the pending deletion cannot be canceled and may still finish.', 'AbortError'));
      };
      request.onsuccess = () => finish();
      request.onerror = () => finish(request.error ?? new SereneDBError('IndexedDB could not delete the database.', 'STORAGE_ERROR'));
      request.onblocked = () => {
        if (!settled && timer === undefined) timer = setTimeout(() => finish(new SereneDBError('Database deletion is blocked by another connection. Close peers; the pending deletion cannot be canceled and may still finish.', 'STORAGE_BLOCKED')), this.blockedTimeoutMs);
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      if (signal?.aborted) onAbort();
    });
  }
}
