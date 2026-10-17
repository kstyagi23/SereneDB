import { checkAbort } from './errors.js';
import type { Metric } from './types.js';

interface KernelExports extends WebAssembly.Exports {
  memory: WebAssembly.Memory;
  allocate: (bytes: number) => number;
  release: (pointer: number, bytes: number) => void;
  scan: (rows: number, query: number, output: number, count: number, dimension: number, euclidean: number) => void;
}

export class SIMDScanner {
  private matrix = 0;
  private query = 0;
  private output = 0;
  private count = 0;
  private dimension = 0;
  private bytes = 0;
  private owner: unknown;

  private constructor(private kernel: KernelExports) {}

  static async create(): Promise<SIMDScanner | null> {
    try {
      const response = await fetch(new URL('./distance.wasm', import.meta.url));
      if (!response.ok) return null;
      const { instance } = await WebAssembly.instantiate(await response.arrayBuffer());
      return new SIMDScanner(instance.exports as KernelExports);
    } catch {
      return null;
    }
  }

  load(owner: unknown, vectors: Float32Array, count: number, dimension: number): void {
    if (this.owner === owner) return;
    this.clear();
    this.bytes = count * dimension * 4;
    this.count = count;
    this.dimension = dimension;
    this.matrix = this.kernel.allocate(this.bytes);
    this.query = this.kernel.allocate(dimension * 4);
    this.output = this.kernel.allocate(count * 4);
    if (!this.matrix || !this.query || !this.output) { this.clear(); throw new Error('WebAssembly memory allocation failed.'); }
    new Float32Array(this.kernel.memory.buffer, this.matrix, count * dimension).set(vectors.subarray(0, count * dimension));
    this.owner = owner;
  }

  async score(vector: Float32Array, metric: Metric, signal?: AbortSignal): Promise<Float32Array> {
    checkAbort(signal);
    if (vector.length !== this.dimension) throw new Error('SIMD query dimension mismatch.');
    new Float32Array(this.kernel.memory.buffer, this.query, this.dimension).set(vector);
    const chunk = Math.max(64, Math.floor(262144 / this.dimension));
    let lastYield = performance.now();
    for (let start = 0; start < this.count; start += chunk) {
      this.kernel.scan(this.matrix + start * this.dimension * 4, this.query, this.output + start * 4, Math.min(chunk, this.count - start), this.dimension, metric === 'euclidean' ? 1 : 0);
      if (start + chunk < this.count && performance.now() - lastYield > 8) { await new Promise<void>(resolve => setTimeout(resolve, 0)); checkAbort(signal); lastYield = performance.now(); }
    }
    return new Float32Array(this.kernel.memory.buffer, this.output, this.count).slice();
  }

  clear(): void {
    if (this.matrix) this.kernel.release(this.matrix, this.bytes);
    if (this.query) this.kernel.release(this.query, this.dimension * 4);
    if (this.output) this.kernel.release(this.output, this.count * 4);
    this.matrix = this.query = this.output = this.count = this.dimension = this.bytes = 0;
    this.owner = null;
  }
}
