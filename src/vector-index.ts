import { checkAbort, invalid } from './errors.js';
import type { IndexConfig, IndexMatch, IndexSnapshot, Metric, VectorSearchOptions } from './types.js';
import { integer, validName } from './validation.js';

const FORMAT = 1;
const SEED = 0x5eed1234;
const MAX_LEVEL = 32;
const MAX_ROWS = 1_000_000;
const MAX_ELEMENTS = 134_217_728;
const MAX_LINKS = 16_777_216;
const MAX_ID_CHARACTERS = 16_777_216;
const TASK_BUDGET_MS = 8;

function clock(): number {
  return globalThis.performance?.now() ?? Date.now();
}

async function yieldTask(signal?: AbortSignal): Promise<void> {
  checkAbort(signal);
  const scheduler = (globalThis as typeof globalThis & { scheduler?: { yield?: () => Promise<void> } }).scheduler;
  if (scheduler?.yield) await scheduler.yield();
  else await new Promise<void>(resolve => setTimeout(resolve, 0));
  checkAbort(signal);
}

function compare(a: number, da: number, b: number, db: number, ids: string[]): number {
  if (da !== db) return da < db ? -1 : 1;
  if (ids[a] !== ids[b]) return ids[a] < ids[b] ? -1 : 1;
  return a - b;
}

// Parallel numeric arrays avoid allocating a candidate object for every distance.
class NodeHeap {
  private nodes: number[] = [];
  private distances: number[] = [];
  private length = 0;

  constructor(private readonly ids: string[], private readonly worstFirst: boolean) {}

  get size(): number { return this.length; }
  get node(): number { return this.nodes[0]; }
  get distance(): number { return this.distances[0]; }
  clear(): void { this.length = 0; }

  private precedes(a: number, da: number, b: number, db: number): boolean {
    const order = compare(a, da, b, db, this.ids);
    return this.worstFirst ? order > 0 : order < 0;
  }

  push(node: number, distance: number): void {
    let position = this.length++;
    while (position > 0) {
      const parent = (position - 1) >>> 1;
      if (!this.precedes(node, distance, this.nodes[parent], this.distances[parent])) break;
      this.nodes[position] = this.nodes[parent];
      this.distances[position] = this.distances[parent];
      position = parent;
    }
    this.nodes[position] = node;
    this.distances[position] = distance;
  }

  replace(node: number, distance: number): void {
    let position = 0;
    const half = this.length >>> 1;
    while (position < half) {
      let child = position * 2 + 1;
      if (child + 1 < this.length && this.precedes(this.nodes[child + 1], this.distances[child + 1], this.nodes[child], this.distances[child])) child++;
      if (!this.precedes(this.nodes[child], this.distances[child], node, distance)) break;
      this.nodes[position] = this.nodes[child];
      this.distances[position] = this.distances[child];
      position = child;
    }
    this.nodes[position] = node;
    this.distances[position] = distance;
  }

  pop(): number {
    const node = this.nodes[0];
    if (--this.length > 0) this.replace(this.nodes[this.length], this.distances[this.length]);
    return node;
  }
}

interface Workspace {
  ids: string[];
  visited: Uint32Array;
  stamp: number;
  candidates: NodeHeap;
  found: NodeHeap;
  pruning: NodeHeap;
}

interface Work {
  steps: number;
  quantum: number;
}

interface CooperativeState {
  signal?: AbortSignal;
  version: number;
  deadline: number;
}

export interface VectorIndexStats {
  size: number;
  nodeCount: number;
  capacity: number;
  tombstones: number;
  tombstoneRatio: number;
  links: number;
  needsRebuild: boolean;
}

export class VectorIndex {
  readonly metric: Metric;
  readonly options: IndexConfig;
  private readonly dimensionValue: number;
  private rows = new Float32Array(0);
  private active = new Uint8Array(0);
  private ids: string[] = [];
  private levels: number[] = [];
  private neighbors: number[][][] = [];
  private lookup = new Map<string, number>();
  private count = 0;
  private capacityValue = 0;
  private entryPoint = -1;
  private maxLevel = -1;
  private randomState = SEED;
  private linkCount = 0;
  private idCharacters = 0;
  private version = 0;
  private workspaces: Workspace[] = [];

  constructor(dimension: number, metric: Metric, options: IndexConfig) {
    integer(dimension, 'Dimension', 1, 16384);
    if (metric !== 'cosine' && metric !== 'dot' && metric !== 'euclidean') invalid('Metric must be cosine, dot, or euclidean.');
    if (!options || (options.type !== 'hnsw' && options.type !== 'flat')) invalid('Index type must be hnsw or flat.');
    integer(options.m, 'HNSW m', 2, 64);
    integer(options.efConstruction, 'efConstruction', options.m, 4096);
    integer(options.efSearch, 'efSearch', 1, 65536);
    integer(options.annThreshold, 'annThreshold', 0, 10000000);
    this.dimensionValue = dimension;
    this.metric = metric;
    this.options = Object.freeze({ ...options });
  }

  get dimension(): number { return this.dimensionValue; }
  get size(): number { return this.lookup.size; }
  get physicalCount(): number { return this.count; }
  get tombstoneRatio(): number { return this.count ? (this.count - this.size) / this.count : 0; }
  get stats(): VectorIndexStats {
    const tombstones = this.count - this.size;
    return {
      size: this.size,
      nodeCount: this.count,
      capacity: this.capacityValue,
      tombstones,
      tombstoneRatio: this.tombstoneRatio,
      links: this.linkCount,
      needsRebuild: tombstones > 0 && (this.size === 0 || (tombstones >= 64 && this.tombstoneRatio >= 0.2)),
    };
  }

  has(id: string): boolean { return this.lookup.has(id); }

  get(id: string): Float32Array | undefined {
    const node = this.lookup.get(id);
    return node === undefined ? undefined : this.rows.slice(node * this.dimension, (node + 1) * this.dimension);
  }

  // Borrowed read-only views for bulk distance kernels; invalidated by mutations.
  getPackedVectors(): { vectors: Float32Array; active: Uint8Array; ids: string[]; count: number } {
    return { vectors: this.rows.subarray(0, this.count * this.dimension), active: this.active.subarray(0, this.count), ids: this.ids, count: this.count };
  }

  private validateVector(vector: Float32Array): void {
    if (!(vector instanceof Float32Array)) invalid('Index vectors must be prepared Float32Array values.');
    if (vector.length !== this.dimension) invalid(`Vector dimension mismatch: expected ${this.dimension}, received ${vector.length}.`, 'SCHEMA_MISMATCH');
    let nonzero = false;
    for (let i = 0; i < vector.length; i++) {
      if (!Number.isFinite(vector[i])) invalid(`Vector element ${i} must be finite.`);
      nonzero ||= vector[i] !== 0;
    }
    if (this.metric === 'cosine' && !nonzero) invalid('Cosine vectors must have a nonzero norm.');
  }

  private ensureCapacity(required: number): void {
    const limit = Math.min(MAX_ROWS, Math.floor(MAX_ELEMENTS / this.dimension));
    if (required > limit) invalid('Vector index exceeds its safe in-memory capacity; compact tombstones or split the collection.', 'INDEX_CAPACITY');
    if (required <= this.capacityValue) return;
    const capacity = Math.min(limit, Math.max(16, required, this.capacityValue * 2));
    const rows = new Float32Array(capacity * this.dimension);
    const active = new Uint8Array(capacity);
    rows.set(this.rows.subarray(0, this.count * this.dimension));
    active.set(this.active.subarray(0, this.count));
    this.rows = rows;
    this.active = active;
    this.capacityValue = capacity;
  }

  private acquire(): Workspace {
    const workspace = this.workspaces.pop();
    if (workspace) {
      if (workspace.visited.length < this.capacityValue) workspace.visited = new Uint32Array(this.capacityValue);
      return workspace;
    }
    return { ids: this.ids, visited: new Uint32Array(this.capacityValue), stamp: 0, candidates: new NodeHeap(this.ids, false), found: new NodeHeap(this.ids, true), pruning: new NodeHeap(this.ids, true) };
  }

  private release(workspace: Workspace): void {
    workspace.candidates.clear();
    workspace.found.clear();
    workspace.pruning.clear();
    if (workspace.ids === this.ids && this.workspaces.length < 2) this.workspaces.push(workspace);
  }

  private work(): Work {
    return { steps: 0, quantum: Math.max(1, Math.min(256, Math.floor(131072 / this.dimension))) };
  }

  private assertVersion(version: number): void {
    if (this.version !== version) invalid('Vector index changed during the operation; retry against the current revision.', 'INDEX_CHANGED');
  }

  private async cooperate<T>(iterator: Generator<void, T>, state: CooperativeState): Promise<T> {
    for (;;) {
      checkAbort(state.signal);
      this.assertVersion(state.version);
      const step = iterator.next();
      checkAbort(state.signal);
      this.assertVersion(state.version);
      if (clock() >= state.deadline) {
        await yieldTask(state.signal);
        this.assertVersion(state.version);
        state.deadline = clock() + TASK_BUDGET_MS;
      }
      if (step.done) return step.value;
    }
  }

  private sync<T>(iterator: Generator<void, T>): T {
    for (;;) {
      const step = iterator.next();
      if (step.done) return step.value;
    }
  }

  private distance(query: Float32Array, node: number): number {
    const rows = this.rows;
    const offset = node * this.dimension;
    const end = this.dimension - (this.dimension % 4);
    let sum = 0;
    let i = 0;
    if (this.metric === 'euclidean') {
      for (; i < end; i += 4) {
        const a = query[i] - rows[offset + i];
        const b = query[i + 1] - rows[offset + i + 1];
        const c = query[i + 2] - rows[offset + i + 2];
        const d = query[i + 3] - rows[offset + i + 3];
        sum += a * a + b * b + c * c + d * d;
      }
      for (; i < this.dimension; i++) { const d = query[i] - rows[offset + i]; sum += d * d; }
      return sum;
    }
    for (; i < end; i += 4) sum += query[i] * rows[offset + i] + query[i + 1] * rows[offset + i + 1] + query[i + 2] * rows[offset + i + 2] + query[i + 3] * rows[offset + i + 3];
    for (; i < this.dimension; i++) sum += query[i] * rows[offset + i];
    return this.metric === 'cosine' ? -Math.max(-1, Math.min(1, sum)) : -sum;
  }

  private between(a: number, b: number): number {
    const rows = this.rows;
    const oa = a * this.dimension;
    const ob = b * this.dimension;
    const end = this.dimension - (this.dimension % 4);
    let sum = 0;
    let i = 0;
    if (this.metric === 'euclidean') {
      for (; i < end; i += 4) {
        const x = rows[oa + i] - rows[ob + i];
        const y = rows[oa + i + 1] - rows[ob + i + 1];
        const z = rows[oa + i + 2] - rows[ob + i + 2];
        const w = rows[oa + i + 3] - rows[ob + i + 3];
        sum += x * x + y * y + z * z + w * w;
      }
      for (; i < this.dimension; i++) { const d = rows[oa + i] - rows[ob + i]; sum += d * d; }
      return sum;
    }
    for (; i < end; i += 4) sum += rows[oa + i] * rows[ob + i] + rows[oa + i + 1] * rows[ob + i + 1] + rows[oa + i + 2] * rows[ob + i + 2] + rows[oa + i + 3] * rows[ob + i + 3];
    for (; i < this.dimension; i++) sum += rows[oa + i] * rows[ob + i];
    return this.metric === 'cosine' ? -Math.max(-1, Math.min(1, sum)) : -sum;
  }

  private randomLevel(): number {
    let state = this.randomState;
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    this.randomState = state >>> 0;
    return Math.min(MAX_LEVEL, Math.floor(-Math.log((this.randomState + 1) / 4294967297) / Math.log(this.options.m)));
  }

  private *greedy(query: Float32Array, entry: number, fromLevel: number, toLevel: number, work: Work): Generator<void, number> {
    let best = this.distance(query, entry);
    for (let level = fromLevel; level >= toLevel; level--) {
      let changed = true;
      while (changed) {
        changed = false;
        const neighbors = this.neighbors[entry][level];
        for (let i = 0; i < neighbors.length; i++) {
          const node = neighbors[i];
          const distance = this.distance(query, node);
          if (compare(node, distance, entry, best, this.ids) < 0) { best = distance; entry = node; changed = true; }
          if (++work.steps >= work.quantum) { work.steps = 0; yield; }
        }
      }
    }
    return entry;
  }

  private *layer(query: Float32Array, entries: number[], ef: number, level: number, workspace: Workspace, work: Work, activeOnly: boolean): Generator<void, NodeHeap> {
    const { candidates, found, visited } = workspace;
    candidates.clear();
    found.clear();
    workspace.stamp = (workspace.stamp + 1) >>> 0;
    if (workspace.stamp === 0) { visited.fill(0); workspace.stamp = 1; }
    const stamp = workspace.stamp;
    for (const node of entries) {
      if (visited[node] === stamp) continue;
      visited[node] = stamp;
      const distance = this.distance(query, node);
      candidates.push(node, distance);
      if (!activeOnly || this.active[node]) {
        if (found.size < ef) found.push(node, distance);
        else if (compare(node, distance, found.node, found.distance, this.ids) < 0) found.replace(node, distance);
      }
      if (++work.steps >= work.quantum) { work.steps = 0; yield; }
    }
    while (candidates.size) {
      const current = candidates.node;
      if (found.size >= ef && compare(current, candidates.distance, found.node, found.distance, this.ids) > 0) break;
      candidates.pop();
      const neighbors = this.neighbors[current][level];
      for (let i = 0; i < neighbors.length; i++) {
        const node = neighbors[i];
        if (visited[node] === stamp) continue;
        visited[node] = stamp;
        const distance = this.distance(query, node);
        if (found.size < ef || compare(node, distance, found.node, found.distance, this.ids) < 0) {
          candidates.push(node, distance);
          // Deleted rows remain navigable but never occupy the result beam.
          if (!activeOnly || this.active[node]) {
            if (found.size < ef) found.push(node, distance);
            else found.replace(node, distance);
          }
        }
        if (++work.steps >= work.quantum) { work.steps = 0; yield; }
      }
    }
    return found;
  }

  private sorted(heap: NodeHeap): { nodes: number[]; distances: number[] } {
    const nodes = new Array<number>(heap.size);
    const distances = new Array<number>(heap.size);
    for (let i = heap.size - 1; i >= 0; i--) { distances[i] = heap.distance; nodes[i] = heap.pop(); }
    return { nodes, distances };
  }

  private *select(nodes: number[], distances: number[], limit: number, work: Work): Generator<void, number[]> {
    const selected: number[] = [];
    const pruned: number[] = [];
    // Prefer live connections; use dead ones only when a layer lacks live nodes.
    for (let pass = 0; pass < 2; pass++) {
      for (let i = 0; i < nodes.length; i++) {
        const node = nodes[i];
        if (Boolean(this.active[node]) !== (pass === 0)) continue;
        let diverse = true;
        for (let j = 0; j < selected.length; j++) {
          const distance = this.between(node, selected[j]);
          if (++work.steps >= work.quantum) { work.steps = 0; yield; }
          if (distance < distances[i]) { diverse = false; break; }
        }
        if (diverse) selected.push(node);
        else pruned.push(node);
        if (selected.length === limit) return selected;
      }
      for (const node of pruned) { selected.push(node); if (selected.length === limit) return selected; }
      pruned.length = 0;
    }
    return selected;
  }

  private *insert(id: string, vector: Float32Array, work: Work): Generator<void, void> {
    validName(id, 'Id');
    this.validateVector(vector);
    if (this.size === 0 && this.count) {
      this.rows = new Float32Array(0);
      this.active = new Uint8Array(0);
      this.ids = [];
      this.levels = [];
      this.neighbors = [];
      this.workspaces = [];
      this.count = this.capacityValue = this.linkCount = this.idCharacters = 0;
      this.entryPoint = this.maxLevel = -1;
      this.randomState = SEED;
    }
    if (this.idCharacters + id.length > MAX_ID_CHARACTERS || (this.options.type === 'hnsw' && this.linkCount + 2 * this.options.m * (MAX_LEVEL + 1) > MAX_LINKS)) invalid('Vector index graph exceeds its safe in-memory capacity; compact or split the collection.', 'INDEX_CAPACITY');
    this.ensureCapacity(this.count + 1);
    const previous = this.lookup.get(id);
    const node = this.count++;
    const level = this.options.type === 'hnsw' ? this.randomLevel() : 0;
    this.rows.set(vector, node * this.dimension);
    this.ids.push(id);
    this.levels.push(level);
    this.neighbors.push(Array.from({ length: level + 1 }, () => []));
    this.idCharacters += id.length;
    this.active[node] = 1;
    if (previous !== undefined) this.active[previous] = 0;
    this.lookup.set(id, node);
    if (this.entryPoint === -1) { this.entryPoint = node; this.maxLevel = level; this.version++; return; }
    if (this.options.type === 'flat') { this.version++; return; }
    const query = this.rows.subarray(node * this.dimension, (node + 1) * this.dimension);
    const workspace = this.acquire();
    try {
      let entry = this.entryPoint;
      if (this.maxLevel > level) entry = yield* this.greedy(query, entry, this.maxLevel, level + 1, work);
      let entries = [entry];
      for (let layer = Math.min(level, this.maxLevel); layer >= 0; layer--) {
        const found = yield* this.layer(query, entries, this.options.efConstruction, layer, workspace, work, false);
        const candidates = this.sorted(found);
        const selected = yield* this.select(candidates.nodes, candidates.distances, this.options.m, work);
        this.neighbors[node][layer] = selected;
        this.linkCount += selected.length;
        const cap = layer === 0 ? this.options.m * 2 : this.options.m;
        for (const neighbor of selected) {
          const links = this.neighbors[neighbor][layer];
          links.push(node);
          this.linkCount++;
          if (links.length > cap) {
            workspace.pruning.clear();
            for (const candidate of links) {
              workspace.pruning.push(candidate, this.between(neighbor, candidate));
              if (++work.steps >= work.quantum) { work.steps = 0; yield; }
            }
            const ordered = this.sorted(workspace.pruning);
            const retained = yield* this.select(ordered.nodes, ordered.distances, cap, work);
            this.neighbors[neighbor][layer] = retained;
            this.linkCount += retained.length - links.length;
          }
        }
        entries = candidates.nodes;
      }
      if (level > this.maxLevel) { this.entryPoint = node; this.maxLevel = level; }
      this.version++;
    } finally {
      this.release(workspace);
    }
  }

  add(id: string, preparedVector: Float32Array): void { this.sync(this.insert(id, preparedVector, this.work())); }

  remove(id: string): boolean {
    const node = this.lookup.get(id);
    if (node === undefined) return false;
    this.active[node] = 0;
    this.lookup.delete(id);
    this.version++;
    return true;
  }

  private rankLimit(options: VectorSearchOptions): number {
    if (this.metric !== 'euclidean') return options.minScore === undefined ? Infinity : -options.minScore;
    if (options.minScore !== undefined && options.minScore > 0) return -1;
    const scoreLimit = options.minScore === undefined ? Infinity : options.minScore * options.minScore;
    const distanceLimit = options.maxDistance === undefined ? Infinity : options.maxDistance * options.maxDistance;
    return Math.min(scoreLimit, distanceLimit);
  }

  private matches(heap: NodeHeap, topK: number, limit: number): IndexMatch[] {
    while (heap.size && (heap.size > topK || heap.distance > limit)) heap.pop();
    const results = new Array<IndexMatch>(heap.size);
    for (let i = heap.size - 1; i >= 0; i--) {
      const rank = heap.distance;
      const id = this.ids[heap.pop()];
      if (this.metric === 'euclidean') {
        const distance = Math.sqrt(rank);
        results[i] = { id, score: distance === 0 ? 0 : -distance, distance };
      } else results[i] = { id, score: rank === 0 ? 0 : -rank };
    }
    return results;
  }

  private *exact(query: Float32Array, options: VectorSearchOptions, workspace: Workspace, work: Work): Generator<void, IndexMatch[]> {
    const heap = workspace.found;
    heap.clear();
    const k = Math.min(options.topK, this.size);
    const limit = this.rankLimit(options);
    const scores = options.scores;
    for (let node = 0; node < this.count; node++) {
      if (scores && (!Number.isFinite(scores[node]) || (this.metric === 'euclidean' && scores[node] < 0))) invalid('Precomputed distances must be finite and squared Euclidean distances must be nonnegative.');
      if (this.active[node] && (!options.filter || options.filter(this.ids[node]))) {
        let distance = scores ? scores[node] : this.distance(query, node);
        if (scores && this.metric === 'cosine') distance = Math.max(-1, Math.min(1, distance));
        if (distance <= limit) {
          if (heap.size < k) heap.push(node, distance);
          else if (compare(node, distance, heap.node, heap.distance, this.ids) < 0) heap.replace(node, distance);
        }
      }
      if (++work.steps >= work.quantum) { work.steps = 0; yield; }
    }
    return this.matches(heap, k, limit);
  }

  private *ann(query: Float32Array, options: VectorSearchOptions, workspace: Workspace, work: Work): Generator<void, IndexMatch[]> {
    const entry = yield* this.greedy(query, this.entryPoint, this.maxLevel, 1, work);
    const ef = Math.min(this.size, Math.max(options.topK, options.efSearch ?? this.options.efSearch));
    const found = yield* this.layer(query, [entry], ef, 0, workspace, work, true);
    // Severe churn or degenerate disconnected ties must not silently underfill.
    if (found.size < Math.min(options.topK, this.size)) return yield* this.exact(query, options, workspace, work);
    return this.matches(found, options.topK, this.rankLimit(options));
  }

  async search(query: Float32Array, options: VectorSearchOptions): Promise<IndexMatch[]> {
    checkAbort(options?.signal);
    this.validateVector(query);
    if (!options) invalid('Vector search options are required.');
    integer(options.topK, 'topK', 0, MAX_ROWS);
    if (options.efSearch !== undefined) integer(options.efSearch, 'efSearch', 1, 65536);
    if (options.exact !== undefined && typeof options.exact !== 'boolean') invalid('exact must be a boolean.');
    if (options.filter !== undefined && typeof options.filter !== 'function') invalid('filter must be a predicate.');
    if (options.minScore !== undefined && !Number.isFinite(options.minScore)) invalid('minScore must be finite.');
    if (options.maxDistance !== undefined && (this.metric !== 'euclidean' || !Number.isFinite(options.maxDistance) || options.maxDistance < 0)) invalid('maxDistance must be nonnegative and finite, and is only valid with the Euclidean metric.');
    if (options.scores !== undefined && (!(options.scores instanceof Float32Array) || options.scores.length !== this.count)) invalid('Precomputed distances must have one float32 value per physical row.');
    if (!this.size || options.topK === 0) return [];
    const settings = { ...options };
    const preparedQuery = query.slice();
    const work = this.work();
    const state: CooperativeState = { signal: settings.signal, version: this.version, deadline: clock() + TASK_BUDGET_MS };
    const workspace = this.acquire();
    try {
      if (settings.signal && this.count >= work.quantum) { await yieldTask(settings.signal); this.assertVersion(state.version); }
      const exact = settings.exact || settings.filter !== undefined || settings.scores !== undefined || this.options.type === 'flat' || this.size < this.options.annThreshold;
      return await this.cooperate(exact ? this.exact(preparedQuery, settings, workspace, work) : this.ann(preparedQuery, settings, workspace, work), state);
    } finally {
      this.release(workspace);
    }
  }

  async rebuild(records?: Iterable<{ id: string; vector: Float32Array }>, options: { signal?: AbortSignal } = {}): Promise<void> {
    checkAbort(options.signal);
    const state: CooperativeState = { signal: options.signal, version: this.version, deadline: clock() + TASK_BUDGET_MS };
    const next = new VectorIndex(this.dimension, this.metric, this.options);
    const work = this.work();
    const current = this;
    const source = records ?? (function* () {
      for (let node = 0; node < current.count; node++) if (current.active[node]) yield { id: current.ids[node], vector: current.rows.subarray(node * current.dimension, (node + 1) * current.dimension) };
    })();
    if (options.signal) { await yieldTask(options.signal); this.assertVersion(state.version); }
    for (const record of source) await this.cooperate(next.insert(record.id, record.vector, work), state);
    checkAbort(options.signal);
    this.assertVersion(state.version);
    this.rows = next.rows.slice(0, next.count * this.dimension);
    this.active = next.active.slice(0, next.count);
    this.capacityValue = next.count;
    this.ids = next.ids;
    this.levels = next.levels;
    this.neighbors = next.neighbors;
    this.lookup = next.lookup;
    this.count = next.count;
    this.entryPoint = next.entryPoint;
    this.maxLevel = next.maxLevel;
    this.randomState = next.randomState;
    this.linkCount = next.linkCount;
    this.idCharacters = next.idCharacters;
    this.workspaces = [];
    this.version++;
  }

  compact(options: { signal?: AbortSignal } = {}): Promise<void> { return this.rebuild(undefined, options); }

  exportSnapshot(): IndexSnapshot {
    return {
      format: FORMAT,
      dimension: this.dimension,
      metric: this.metric,
      options: { ...this.options },
      ids: this.ids.slice(),
      vectors: this.rows.slice(0, this.count * this.dimension),
      active: this.active.slice(0, this.count),
      levels: this.levels.slice(),
      neighbors: this.neighbors.map(layers => layers.map(neighbors => neighbors.slice())),
      entryPoint: this.entryPoint,
      maxLevel: this.maxLevel,
      randomState: this.randomState,
    };
  }

  static fromSnapshot(snapshot: IndexSnapshot): VectorIndex {
    const bad = (message: string): never => invalid(`Invalid vector index snapshot: ${message}`, 'INVALID_INDEX_SNAPSHOT');
    if (!snapshot || typeof snapshot !== 'object' || snapshot.format !== FORMAT) bad('unsupported format.');
    const index = new VectorIndex(snapshot.dimension, snapshot.metric, snapshot.options);
    if (!Array.isArray(snapshot.ids)) bad('ids must be an array.');
    const count = snapshot.ids.length;
    if (count > MAX_ROWS || count * index.dimension > MAX_ELEMENTS) bad('unsafe row count or matrix size.');
    if (!(snapshot.vectors instanceof Float32Array) || snapshot.vectors.length !== count * index.dimension) bad('matrix length or type.');
    if (!(snapshot.active instanceof Uint8Array) || snapshot.active.length !== count) bad('active flags length or type.');
    if (!Array.isArray(snapshot.levels) || snapshot.levels.length !== count || !Array.isArray(snapshot.neighbors) || snapshot.neighbors.length !== count) bad('graph lengths.');
    if (!Number.isInteger(snapshot.randomState) || snapshot.randomState < 1 || snapshot.randomState > 0xffffffff) bad('random state.');
    if (!Number.isInteger(snapshot.entryPoint) || !Number.isInteger(snapshot.maxLevel)) bad('entry point or maximum level.');
    let maximum = -1;
    let links = 0;
    let idCharacters = 0;
    const lookup = new Map<string, number>();
    for (let node = 0; node < count; node++) {
      const id = snapshot.ids[node];
      if (typeof id !== 'string' || !id.trim() || id.length > 512) bad('invalid id.');
      idCharacters += id.length;
      if (idCharacters > MAX_ID_CHARACTERS) bad('unsafe id size.');
      if (snapshot.active[node] !== 0 && snapshot.active[node] !== 1) bad('active flag must be zero or one.');
      if (snapshot.active[node]) { if (lookup.has(id)) bad('duplicate active id.'); lookup.set(id, node); }
      const level = snapshot.levels[node];
      if (!Number.isInteger(level) || level < 0 || level > MAX_LEVEL || (index.options.type === 'flat' && level !== 0)) bad('invalid node level.');
      maximum = Math.max(maximum, level);
      const layers = snapshot.neighbors[node];
      if (!Array.isArray(layers) || layers.length !== level + 1) bad('invalid layer count.');
      for (let layer = 0; layer <= level; layer++) {
        const neighbors = layers[layer];
        const cap = index.options.type === 'flat' ? 0 : index.options.m * (layer === 0 ? 2 : 1);
        if (!Array.isArray(neighbors) || neighbors.length > cap) bad('degree limit exceeded.');
        links += neighbors.length;
        if (links > MAX_LINKS) bad('unsafe graph size.');
        const unique = new Set<number>();
        for (const neighbor of neighbors) {
          if (!Number.isInteger(neighbor) || neighbor < 0 || neighbor >= count || neighbor === node || snapshot.levels[neighbor] < layer || unique.has(neighbor)) bad('invalid or duplicate neighbor.');
          unique.add(neighbor);
        }
      }
      let nonzero = false;
      const offset = node * index.dimension;
      for (let i = 0; i < index.dimension; i++) {
        const value = snapshot.vectors[offset + i];
        if (!Number.isFinite(value)) bad('nonfinite vector value.');
        nonzero ||= value !== 0;
      }
      if (index.metric === 'cosine' && !nonzero) bad('zero cosine vector.');
    }
    if (snapshot.maxLevel !== maximum || (count === 0 ? snapshot.entryPoint !== -1 : snapshot.entryPoint < 0 || snapshot.entryPoint >= count || snapshot.levels[snapshot.entryPoint] !== maximum)) bad('entry point does not match maximum level.');
    index.rows = snapshot.vectors.slice();
    index.active = snapshot.active.slice();
    index.ids = snapshot.ids.slice();
    index.levels = snapshot.levels.slice();
    index.neighbors = snapshot.neighbors.map(layers => layers.map(neighbors => neighbors.slice()));
    index.lookup = lookup;
    index.capacityValue = index.count = count;
    index.entryPoint = snapshot.entryPoint;
    index.maxLevel = maximum;
    index.randomState = snapshot.randomState;
    index.linkCount = links;
    index.idCharacters = idCharacters;
    return index;
  }
}
