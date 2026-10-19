import { checkAbort } from './errors.js';
import type { Metric } from './types.js';

interface NativeExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  allocate(bytes: number): number;
  release(pointer: number, bytes: number): void;
  scan(rows: number, query: number, output: number, count: number, dimension: number, l2: number): void;
  row_distance(rows: number, query: number, row: number, dimension: number, l2: number): number;
  row_between(rows: number, a: number, b: number, dimension: number, l2: number): number;
}

export interface NativeKernel {
  append(row: number, vector: Float32Array): void;
  distance(query: Float32Array, row: number): number;
  between(a: number, b: number): number;
  score(query: Float32Array, signal?: AbortSignal): Promise<Float32Array>;
  dispose(): void;
}

export type NativeKernelFactory = (dimension: number, metric: Metric) => NativeKernel;

export async function loadNativeKernel(): Promise<NativeKernelFactory | null> {
  try {
    const response = await fetch(new URL('./distance.wasm', import.meta.url));
    if (!response.ok) return null;
    const module = await WebAssembly.compile(await response.arrayBuffer());
    return (dimension, metric) => {
      const instance = new WebAssembly.Instance(module);
      const wasm = instance.exports as NativeExports;
      if (typeof wasm.row_distance !== 'function') throw new Error('SIMD kernel version mismatch. Rebuild the WASM asset.');
      return new NativeStore(wasm, dimension, metric);
    };
  } catch { return null; }
}

class NativeStore implements NativeKernel {
  private matrix = 0;
  private query = 0;
  private output = 0;
  private capacity = 0;
  private count = 0;
  private queryIdentity: Float32Array | null = null;
  private disposed = false;
  private view: Float32Array | null = null;
  private readonly l2: number;

  constructor(private wasm: NativeExports, private dimension: number, private metric: Metric) {
    this.l2 = metric === 'euclidean' ? 1 : 0;
    this.query = wasm.allocate(dimension * 4);
    if (!this.query) throw new Error('SIMD query allocation failed.');
  }

  private memory(): Float32Array {
    if (!this.view || this.view.buffer !== this.wasm.memory.buffer) this.view = new Float32Array(this.wasm.memory.buffer);
    return this.view;
  }

  append(row: number, vector: Float32Array): void {
    if (this.disposed || row < 0 || row > this.count || vector.length !== this.dimension) throw new Error('Invalid SIMD vector append.');
    if (row >= this.capacity) {
      const next = Math.max(16, row + 1, this.capacity * 2);
      const matrix = this.wasm.allocate(next * this.dimension * 4), output = this.wasm.allocate(next * 4);
      if (!matrix || !output) { if (matrix) this.wasm.release(matrix, next * this.dimension * 4); if (output) this.wasm.release(output, next * 4); throw new Error('SIMD matrix allocation failed.'); }
      if (this.matrix) {
        const memory = this.memory();
        memory.copyWithin(matrix / 4, this.matrix / 4, this.matrix / 4 + this.count * this.dimension);
        this.wasm.release(this.matrix, this.capacity * this.dimension * 4);
        this.wasm.release(this.output, this.capacity * 4);
      }
      this.matrix = matrix; this.output = output; this.capacity = next;
    }
    this.memory().set(vector, this.matrix / 4 + row * this.dimension);
    this.count = Math.max(this.count, row + 1);
    this.queryIdentity = null;
  }

  private setQuery(query: Float32Array): void {
    if (this.disposed || query.length !== this.dimension) throw new Error('Invalid SIMD query.');
    if (query !== this.queryIdentity) { this.memory().set(query, this.query / 4); this.queryIdentity = query; }
  }

  private sanitize(value: number): number {
    return this.metric === 'cosine' ? Math.max(-1, Math.min(1, value)) : value;
  }

  distance(query: Float32Array, row: number): number {
    this.setQuery(query);
    return this.sanitize(this.wasm.row_distance(this.matrix, this.query, row, this.dimension, this.l2));
  }

  between(a: number, b: number): number { return this.sanitize(this.wasm.row_between(this.matrix, a, b, this.dimension, this.l2)); }

  async score(query: Float32Array, signal?: AbortSignal): Promise<Float32Array> {
    this.setQuery(query); checkAbort(signal);
    let lastYield = performance.now();
    const chunk = Math.max(64, Math.floor(524288 / this.dimension));
    for (let start = 0; start < this.count; start += chunk) {
      this.wasm.scan(this.matrix + start * this.dimension * 4, this.query, this.output + start * 4, Math.min(chunk, this.count - start), this.dimension, this.l2);
      if (start + chunk < this.count && performance.now() - lastYield > 8) {
        const scheduler = (globalThis as unknown as { scheduler?: { yield?: () => Promise<void> } }).scheduler;
        if (scheduler?.yield) await scheduler.yield(); else await new Promise<void>(resolve => setTimeout(resolve, 0));
        checkAbort(signal); lastYield = performance.now();
      }
    }
    return this.memory().slice(this.output / 4, this.output / 4 + this.count);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    if (this.matrix) this.wasm.release(this.matrix, this.capacity * this.dimension * 4);
    if (this.output) this.wasm.release(this.output, this.capacity * 4);
    if (this.query) this.wasm.release(this.query, this.dimension * 4);
    this.matrix = this.output = this.query = this.capacity = this.count = 0;
    this.view = this.queryIdentity = null;
  }
}
