import { checkAbort, invalid, SereneDBError } from './errors.js';
import { compileFilter } from './filter.js';
import { LexicalIndex } from './lexical.js';
import { SIMDScanner } from './distance.js';
import { Storage } from './storage.js';
import { VectorIndex } from './vector-index.js';
import { TransformersEmbedding, transformersEmbedder } from './transformers.js';
import { collectionSchema, integer, prepareMetadata, prepareVector, validName } from './validation.js';
import type { CollectionOptions, CollectionState, DatabaseExport, DocumentInput, IndexMatch, OpenOptions, OperationOptions, PreparedRecord, SearchOptions, SearchResponse, SearchResult, StoredRecord } from './types.js';

type Header = Omit<StoredRecord, 'vector'>;
interface WorkingSet {
  state: CollectionState;
  records: Map<string, Header>;
  index: VectorIndex | null;
  lexical: LexicalIndex;
}

const yieldTask = () => new Promise<void>(resolve => setTimeout(resolve, 0));

export class Engine {
  private entries = new Map<string, WorkingSet>();
  private models = new Map<string, TransformersEmbedding>();
  private queue: Promise<unknown> = Promise.resolve();
  private pending = 0;
  private closed = false;
  private readonly maxResidentCollections = 2;
  private constructor(private storage: Storage, private scanner: SIMDScanner | null) {}

  static async open(name: string, options: OpenOptions = {}): Promise<Engine> {
    const storage = await Storage.open(name, options);
    const scanner = options.wasm === false ? null : await SIMDScanner.create();
    return new Engine(storage, scanner);
  }

  call(method: string, args: unknown[] = [], options: OperationOptions = {}): Promise<unknown> {
    if (this.pending >= 128) return Promise.reject(new SereneDBError('Database queue is full. Await operations before enqueueing more.', 'BACKPRESSURE'));
    this.pending++;
    const task = this.queue.then(async () => {
      checkAbort(options.signal);
      if (this.closed && method !== 'close') throw new SereneDBError('Database is closed.', 'CLOSED');
      const progress = options.onProgress;
      const safe = { ...options, onProgress: progress ? (event: Parameters<NonNullable<OperationOptions['onProgress']>>[0]) => { try { progress(event); } catch { /* Observer failures cannot invalidate a commit. */ } } : undefined };
      return this.dispatch(method, args, safe);
    });
    this.queue = task.catch(() => {});
    return task.finally(() => { this.pending--; });
  }

  private async working(name: string, generation?: string, options: OperationOptions = {}): Promise<WorkingSet> {
    validName(name, 'Collection name');
    const state = await this.storage.refreshState(name, options.signal);
    if (generation && state.generation !== generation) throw new SereneDBError('Collection was dropped or replaced. Open a new collection handle.', 'STALE_HANDLE');
    const cached = this.entries.get(name);
    if (cached && cached.state.generation === state.generation && cached.state.revision === state.revision) { this.entries.delete(name); this.entries.set(name, cached); return cached; }
    checkAbort(options.signal);
    const snapshot = await this.storage.readCollection(name, options.signal);
    if (generation && snapshot.state.generation !== generation) throw new SereneDBError('Collection was replaced.', 'STALE_HANDLE');
    const records = new Map<string, Header>();
    const lexical = new LexicalIndex();
    for (const record of snapshot.records) {
      const { vector: _vector, ...header } = record;
      records.set(record.id, header);
      lexical.add(record);
    }
    if (this.scanner) this.scanner.clear();
    let index: VectorIndex | null = null;
    if (snapshot.state.schema.dimension !== null) {
      if (snapshot.snapshot?.generation === snapshot.state.generation && snapshot.snapshot.revision === snapshot.state.revision) {
        try {
          index = VectorIndex.fromSnapshot(snapshot.snapshot.payload);
          if (index.size !== records.size || index.dimension !== snapshot.state.schema.dimension || snapshot.snapshot.payload.metric !== snapshot.state.schema.metric || snapshot.records.some(record => {
            const indexed = index!.get(record.id);
            return !indexed || indexed.some((value, i) => value !== record.vector[i]);
          })) index = null;
        } catch { index = null; }
      }
      if (!index) {
        index = new VectorIndex(snapshot.state.schema.dimension, snapshot.state.schema.metric, snapshot.state.schema.index);
        let lastYield = performance.now();
        for (let i = 0; i < snapshot.records.length; i++) {
          checkAbort(options.signal);
          const record = snapshot.records[i];
          index.add(record.id, record.vector);
          if (performance.now() - lastYield > 8) {
            options.onProgress?.({ phase: 'indexing', completed: i + 1, total: snapshot.records.length });
            await yieldTask(); lastYield = performance.now();
          }
        }
      }
    }
    checkAbort(options.signal);
    const entry = { state: snapshot.state, records, index, lexical };
    this.entries.delete(name);
    this.entries.set(name, entry);
    while (this.entries.size > this.maxResidentCollections) {
      const [oldName, old] = this.entries.entries().next().value!;
      if (old.index) { try { await this.storage.checkpoint(oldName, old.state.generation, old.state.revision, old.index.exportSnapshot()); } catch { /* Eviction must not depend on a derived checkpoint. */ } }
      this.entries.delete(oldName);
    }
    this.scanner?.clear();
    return entry;
  }

  private model(entry: WorkingSet): TransformersEmbedding {
    const schema = entry.state.schema.embedding;
    if (!schema) invalid('This collection has no embedder. Provide vectors or create a collection with transformersEmbedder().', 'NO_EMBEDDER');
    const key = entry.state.generation;
    let model = this.models.get(key);
    if (!model) { model = new TransformersEmbedding(schema.config); this.models.set(key, model); }
    return model;
  }

  private materialize(entry: WorkingSet, id: string, includeVector = true): StoredRecord | Header | null {
    const record = entry.records.get(id);
    if (!record) return null;
    const clone = structuredClone(record);
    return includeVector ? { ...clone, vector: entry.index!.get(id)! } : clone;
  }

  private async prepare(entry: WorkingSet, inputs: DocumentInput[], options: OperationOptions): Promise<PreparedRecord[]> {
    if (!Array.isArray(inputs)) invalid('Documents must be an array.');
    integer(inputs.length, 'Atomic batch size', 0, 100000);
    const raw: PreparedRecord[] = [];
    const texts: string[] = [], positions: number[] = [];
    let dimension = entry.state.schema.dimension;
    let payloadBytes = 0;
    for (const item of inputs) {
      if (!item || typeof item !== 'object') invalid('Document must be an object.');
      validName(item.id, 'Document ID');
      if (item.text !== undefined && (typeof item.text !== 'string' || item.text.length > 2000000)) invalid('Document text must be a string of at most 2 million characters.');
      const metadata = prepareMetadata(item.metadata);
      payloadBytes += (item.text?.length ?? 0) * 2 + JSON.stringify(metadata).length * 2 + ((item.vector?.length ?? dimension ?? 384) * 4);
      if (payloadBytes > 64 * 1024 * 1024) invalid('Atomic batch exceeds the 64 MiB input budget. Use collection.ingest() for bounded streaming batches.', 'BATCH_TOO_LARGE');
      if (item.vector !== undefined) {
        if (entry.state.schema.embedding && item.space !== entry.state.schema.embedding.fingerprint) invalid('Raw vectors in an embedding collection require its exact space fingerprint.', 'SCHEMA_MISMATCH');
        const vector = prepareVector(item.vector, entry.state.schema.metric, dimension);
        dimension ??= vector.length;
        raw.push({ id: item.id, vector, metadata, ...(item.text !== undefined ? { text: item.text } : {}) });
      } else {
        if (typeof item.text !== 'string' || !item.text.trim()) invalid('Each document requires a vector or nonempty text.');
        this.model(entry);
        positions.push(raw.length); texts.push(item.text);
        raw.push({ id: item.id, vector: new Float32Array(0), text: item.text, metadata });
      }
    }
    checkAbort(options.signal);
    if (texts.length) {
      const vectors = await this.model(entry).encodeDocuments(texts, options);
      for (let i = 0; i < vectors.length; i++) {
        const vector = prepareVector(vectors[i], entry.state.schema.metric, dimension);
        dimension ??= vector.length;
        raw[positions[i]].vector = vector;
      }
    }
    checkAbort(options.signal);
    return raw;
  }

  private async upsert(name: string, generation: string, inputs: DocumentInput[], options: OperationOptions): Promise<unknown> {
    const entry = await this.working(name, generation, options);
    const records = await this.prepare(entry, inputs, options);
    const result = await this.storage.mutate(name, { upserts: records }, generation, options.signal);
    this.scanner?.clear();
    // Storage has committed. Cancellation or a failed derived index cannot undo that acknowledgement.
    if (result.state.revision === entry.state.revision + 1 && result.state.count >= entry.records.size) {
      try {
        entry.index ??= result.state.schema.dimension === null ? null : new VectorIndex(result.state.schema.dimension, result.state.schema.metric, result.state.schema.index);
        let lastYield = performance.now();
        for (const record of result.upserted) {
          entry.index!.add(record.id, record.vector);
          const { vector: _vector, ...header } = record;
          entry.records.set(record.id, header); entry.lexical.add(record);
          if (performance.now() - lastYield > 8) { await yieldTask(); lastYield = performance.now(); }
        }
        entry.state = result.state;
      } catch { this.entries.delete(name); }
    } else if (result.state.revision !== entry.state.revision) this.entries.delete(name);
    return { count: result.state.count, upserted: result.upserted.length, revision: result.state.revision, generation: result.state.generation };
  }

  private async search(name: string, generation: string, input: SearchOptions, options: OperationOptions): Promise<SearchResponse> {
    const start = performance.now();
    if (!input || typeof input !== 'object') invalid('Search options must be an object.');
    const topK = input.topK ?? 10;
    integer(topK, 'topK', 0, 10000);
    if (input.efSearch !== undefined) integer(input.efSearch, 'efSearch', 1, 65536);
    const requested = input.mode ?? 'auto';
    if (!['auto', 'ann', 'exact', 'hybrid'].includes(requested)) invalid('Search mode must be auto, exact, ann, or hybrid.');
    if (input.minScore !== undefined && !Number.isFinite(input.minScore)) invalid('minScore must be finite.');
    if (input.maxDistance !== undefined && (!Number.isFinite(input.maxDistance) || input.maxDistance < 0)) invalid('maxDistance must be finite and nonnegative.');
    if (input.mmr !== undefined && (!Number.isFinite(input.mmr) || input.mmr < 0 || input.mmr > 1)) invalid('mmr must be between zero and one.');
    if (input.vector !== undefined && input.text !== undefined) invalid('Pass either text or a query vector, not both.');
    if (input.vector === undefined && (typeof input.text !== 'string' || !input.text.trim())) invalid('Search requires a vector or nonempty text.');
    const entry = await this.working(name, generation, options);
    if (input.maxDistance !== undefined && entry.state.schema.metric !== 'euclidean') invalid('maxDistance is only valid for Euclidean collections.');
    const predicate = compileFilter(input.filter);
    const filter = (id: string) => predicate(entry.records.get(id)! as StoredRecord);
    let query: Float32Array | null = null;
    if (input.vector !== undefined) {
      if (entry.state.schema.embedding && input.space !== entry.state.schema.embedding.fingerprint) invalid('Query vector embedding-space fingerprint does not match.', 'SCHEMA_MISMATCH');
      query = prepareVector(input.vector, entry.state.schema.metric, entry.state.schema.dimension);
    } else if (entry.state.schema.embedding) query = prepareVector(await this.model(entry).encodeQuery(input.text!, options), entry.state.schema.metric, entry.state.schema.dimension);
    else if (requested !== 'hybrid') invalid('Text search requires a configured embedder. Use hybrid mode for lexical-only search.', 'NO_EMBEDDER');
    checkAbort(options.signal);
    const exact = requested === 'exact' || entry.state.schema.index.type === 'flat' || Boolean(input.filter) || entry.records.size < entry.state.schema.index.annThreshold;
    let mode: SearchResponse['mode'] = requested === 'hybrid' ? 'hybrid' : exact ? 'exact' : 'ann';
    const candidates = input.mmr !== undefined || mode === 'hybrid' ? Math.min(10000, Math.max(topK * 4, 32)) : topK;
    let matches: IndexMatch[] = [];
    if (query && entry.index && topK > 0) {
      let scores: Float32Array | undefined;
      if (exact && this.scanner && entry.records.size >= 256) {
        try {
          const packed = entry.index.getPackedVectors();
          this.scanner.load(entry.index, packed.vectors, packed.count, entry.index.dimension);
          const calculated = await this.scanner.score(query, entry.state.schema.metric, options.signal);
          if (calculated.every(Number.isFinite)) scores = calculated;
        } catch (error) { checkAbort(options.signal); this.scanner.clear(); }
      }
      matches = await entry.index.search(query, { topK: candidates, exact, efSearch: input.efSearch, filter: input.filter ? filter : undefined, minScore: input.minScore, maxDistance: input.maxDistance, signal: options.signal, scores });
    }
    if (mode === 'hybrid' && topK > 0) {
      const lexical = entry.lexical.search(input.text ?? '', candidates, filter);
      const fused = new Map<string, number>();
      for (const ranking of [matches, lexical]) for (let i = 0; i < ranking.length; i++) fused.set(ranking[i].id, (fused.get(ranking[i].id) ?? 0) + 1 / (60 + i + 1));
      if (query && (input.minScore !== undefined || input.maxDistance !== undefined)) {
        const allowed = new Set((await entry.index!.search(query, { topK: entry.records.size, exact: true, filter, minScore: input.minScore, maxDistance: input.maxDistance, signal: options.signal })).map(match => match.id));
        for (const id of fused.keys()) if (!allowed.has(id)) fused.delete(id);
      }
      matches = Array.from(fused, ([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
    }
    if (input.mmr !== undefined && query && matches.length > topK) matches = await this.diversify(entry, matches, topK, input.mmr, options.signal);
    checkAbort(options.signal);
    const results: SearchResult[] = matches.slice(0, topK).map(match => {
      const record = this.materialize(entry, match.id, input.includeVectors === true)! as StoredRecord;
      const { collection: _collection, ...value } = record;
      return { ...value, score: match.score, ...(match.distance === undefined ? {} : { distance: match.distance }) };
    });
    return { results, mode, revision: entry.state.revision, generation: entry.state.generation, elapsedMs: performance.now() - start };
  }

  private async diversify(entry: WorkingSet, matches: IndexMatch[], topK: number, lambda: number, signal?: AbortSignal): Promise<IndexMatch[]> {
    const vectors = matches.map(match => prepareVector(entry.index!.get(match.id)!, 'cosine'));
    const penalties = new Float64Array(matches.length).fill(-1);
    const chosen = new Set<number>(), results: IndexMatch[] = [];
    const maximum = Math.max(...matches.map(match => match.score)), minimum = Math.min(...matches.map(match => match.score));
    while (results.length < topK && chosen.size < matches.length) {
      let best = -1, bestScore = -Infinity;
      for (let i = 0; i < matches.length; i++) if (!chosen.has(i)) {
        const relevance = (matches[i].score - minimum) / (maximum - minimum || 1);
        const score = lambda * relevance - (1 - lambda) * penalties[i];
        if (score > bestScore) { best = i; bestScore = score; }
      }
      chosen.add(best); results.push(matches[best]);
      for (let i = 0; i < vectors.length; i++) if (!chosen.has(i)) {
        let dot = 0;
        for (let j = 0; j < vectors[i].length; j++) dot += vectors[i][j] * vectors[best][j];
        penalties[i] = Math.max(penalties[i], dot);
      }
      if (results.length % 8 === 0) { await yieldTask(); checkAbort(signal); }
    }
    return results;
  }

  private async dispatch(method: string, args: unknown[], options: OperationOptions): Promise<unknown> {
    const [name, generation] = args as [string, string];
    switch (method) {
      case 'collection': {
        const supplied = (args[1] ?? {}) as CollectionOptions;
        const normalized = { ...supplied, ...(supplied.embedder ? { embedder: transformersEmbedder(supplied.embedder) } : {}) };
        return this.storage.ensureCollection(name, collectionSchema(normalized), { explicit: { dimension: supplied.dimension !== undefined, metric: supplied.metric !== undefined, embedding: supplied.embedder !== undefined, index: supplied.index !== undefined }, signal: options.signal });
      }
      case 'list': return this.storage.listCollections();
      case 'state': {
        const state = await this.storage.refreshState(name, options.signal);
        if (state.generation !== generation) throw new SereneDBError('Collection handle is stale.', 'STALE_HANDLE');
        return state;
      }
      case 'upsert': return this.upsert(name, generation, args[2] as DocumentInput[], options);
      case 'get': {
        validName(args[2], 'Document ID');
        return await this.storage.get(name, args[2] as string, options.signal, generation) ?? null;
      }
      case 'scan': {
        const input = (args[2] ?? {}) as { offset?: number; limit?: number; includeVectors?: boolean; filter?: SearchOptions['filter'] };
        const offset = input.offset ?? 0, limit = input.limit ?? 100;
        integer(offset, 'offset', 0, Number.MAX_SAFE_INTEGER); integer(limit, 'limit', 0, 100000);
        const entry = await this.working(name, generation, options), filter = compileFilter(input.filter);
        const ids: string[] = [];
        let passed = 0;
        for (const [id, record] of entry.records) if (filter(record as StoredRecord)) { if (passed++ >= offset && ids.length < limit) ids.push(id); }
        return ids.map(id => this.materialize(entry, id, input.includeVectors === true));
      }
      case 'search': return this.search(name, generation, args[2] as SearchOptions, options);
      case 'delete': case 'clear': {
        const ids = method === 'delete' ? args[2] as string[] : [];
        if (!Array.isArray(ids)) invalid('IDs must be an array.');
        ids.forEach(id => validName(id, 'Document ID'));
        const cached = this.entries.get(name);
        const result = await this.storage.mutate(name, method === 'clear' ? { clear: true } : { deleteIds: ids }, generation, options.signal);
        this.scanner?.clear();
        if (method === 'delete' && cached?.state.generation === generation && result.state.revision === cached.state.revision + 1) {
          for (const id of ids) { cached.records.delete(id); cached.index?.remove(id); cached.lexical.remove(id); }
          cached.state = result.state;
        } else if (method === 'clear' || cached?.state.revision !== result.state.revision) this.entries.delete(name);
        return { deleted: result.deleted, count: result.state.count, revision: result.state.revision };
      }
      case 'drop': {
        await this.storage.dropCollection(name, generation, options.signal);
        this.entries.delete(name); this.scanner?.clear();
        await this.models.get(generation)?.dispose(); this.models.delete(generation);
        return;
      }
      case 'preload': {
        const entry = await this.working(name, generation, options);
        await this.model(entry).preload(options);
        return this.model(entry).runtimeInfo();
      }
      case 'unload': {
        const state = await this.storage.refreshState(name, options.signal);
        if (state.generation !== generation) throw new SereneDBError('Collection handle is stale.', 'STALE_HANDLE');
        const cached = this.entries.get(name);
        if (cached?.index && cached.state.revision === state.revision) { try { await this.storage.checkpoint(name, generation, state.revision, cached.index.exportSnapshot()); } catch { /* Unload never requires a snapshot write. */ } }
        this.entries.delete(name); this.scanner?.clear();
        await this.models.get(generation)?.dispose(); this.models.delete(generation);
        return;
      }
      case 'compact': {
        const entry = await this.working(name, generation, options);
        if (entry.index) { await entry.index.compact({ signal: options.signal }); this.scanner?.clear(); }
        if (entry.index) await this.storage.checkpoint(name, generation, entry.state.revision, entry.index.exportSnapshot());
        return this.statistics(entry);
      }
      case 'stats': return this.statistics(await this.working(name, generation, options));
      case 'checkpoint': {
        const entry = await this.working(name, generation, options);
        return entry.index ? this.storage.checkpoint(name, generation, entry.state.revision, entry.index.exportSnapshot()) : true;
      }
      case 'export': {
        const output: DatabaseExport = { format: 'serenedb', version: 2, collections: [] };
        for (const state of await this.storage.listCollections()) {
          const snapshot = await this.storage.readCollection(state.name);
          output.collections.push({ state: snapshot.state, records: snapshot.records.map(record => ({ ...record, vector: Array.from(record.vector) })) });
        }
        return output;
      }
      case 'import': return this.importData(args[0] as DatabaseExport, Boolean(args[1]), options);
      case 'close': case 'destroy': {
        if (method === 'close') {
          for (const entry of this.entries.values()) if (entry.index) { try { await this.storage.checkpoint(entry.state.name, entry.state.generation, entry.state.revision, entry.index.exportSnapshot()); } catch { /* Derived snapshots are optional. */ } }
          this.storage.close();
        } else await this.storage.destroy(options.signal);
        this.closed = true;
        this.entries.clear(); this.scanner?.clear();
        for (const model of this.models.values()) await model.dispose();
        this.models.clear(); return;
      }
      default: invalid(`Unknown database operation ${method}.`);
    }
  }

  private statistics(entry: WorkingSet): unknown {
    return { ...structuredClone(entry.state), vectorBytes: entry.records.size * (entry.state.schema.dimension ?? 0) * 4, index: entry.index?.stats() ?? null, kernel: this.scanner ? 'wasm-simd' : 'javascript', model: this.models.get(entry.state.generation)?.runtimeInfo() ?? null };
  }

  private async importData(data: DatabaseExport, replace: boolean, options: OperationOptions): Promise<unknown> {
    if (!data || data.format !== 'serenedb' || data.version !== 2 || !Array.isArray(data.collections)) invalid('Unsupported export. Expected SereneDB format version 2.');
    const prepared = data.collections.map(collection => {
      validName(collection.state?.name, 'Collection name');
      const schema = collection.state.schema;
      const validated = collectionSchema({ dimension: schema.dimension ?? undefined, metric: schema.metric, index: schema.index, ...(schema.embedding ? { embedder: transformersEmbedder(schema.embedding.config) } : {}) });
      if (!Array.isArray(collection.records)) invalid('Import records must be an array.');
      const records = collection.records.map(record => {
        validName(record.id, 'Document ID');
        if (record.text !== undefined && typeof record.text !== 'string') invalid('Imported text must be a string.');
        return { id: record.id, vector: prepareVector(record.vector, validated.metric, validated.dimension), metadata: prepareMetadata(record.metadata), text: record.text, timestamp: record.timestamp };
      });
      const dimensions = new Set(records.map(record => record.vector.length));
      if (dimensions.size > 1) invalid('Imported collection contains mixed dimensions.');
      return { name: collection.state.name, schema: validated, records };
    });
    if (new Set(prepared.map(collection => collection.name)).size !== prepared.length) invalid('Import collection names must be unique.');
    let committed = 0;
    for (const collection of prepared) {
      checkAbort(options.signal);
      const state = await this.storage.ensureCollection(collection.name, collection.schema);
      await this.storage.mutate(collection.name, { upserts: collection.records, clear: replace }, state.generation, options.signal);
      this.entries.delete(collection.name); committed++;
      options.onProgress?.({ phase: 'importing', completed: committed, total: prepared.length });
    }
    this.scanner?.clear();
    return { collections: committed };
  }
}
