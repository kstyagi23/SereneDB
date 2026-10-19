import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

test('WASM SIMD distances agree with independent float32 reference including tail lanes', async t => {
  let bytes: Buffer;
  try { bytes = await readFile(new URL('../dist/distance.wasm', import.meta.url)); }
  catch { t.skip('Optional WASM kernel is not built. Run npm run build:wasm.'); return; }
  const { instance } = await WebAssembly.instantiate(bytes);
  const wasm = instance.exports as unknown as { memory: WebAssembly.Memory; allocate(n: number): number; release(p: number, n: number): void; scan(r: number, q: number, o: number, n: number, d: number, l2: number): void };
  for (const dimension of [1, 3, 4, 7, 384]) {
    const count = 13;
    const rows = Float32Array.from({ length: count * dimension }, (_, i) => Math.sin(i) / 10);
    const query = Float32Array.from({ length: dimension }, (_, i) => Math.cos(i) / 10);
    const matrix = wasm.allocate(rows.byteLength), pointer = wasm.allocate(query.byteLength), output = wasm.allocate(count * 4);
    new Float32Array(wasm.memory.buffer, matrix, rows.length).set(rows);
    new Float32Array(wasm.memory.buffer, pointer, query.length).set(query);
    for (const l2 of [0, 1]) {
      wasm.scan(matrix, pointer, output, count, dimension, l2);
      const scores = new Float32Array(wasm.memory.buffer, output, count);
      for (let row = 0; row < count; row++) {
        let sum = 0;
        for (let i = 0; i < dimension; i++) sum += l2 ? (rows[row * dimension + i] - query[i]) ** 2 : rows[row * dimension + i] * query[i];
        assert.ok(Math.abs(scores[row] - (l2 ? sum : -sum)) < 1e-5);
      }
    }
    wasm.release(matrix, rows.byteLength); wasm.release(pointer, query.byteLength); wasm.release(output, count * 4);
  }
});
