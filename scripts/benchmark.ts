import { VectorIndex } from '../src/vector-index.js';
import { collectionSchema, prepareVector } from '../src/validation.js';

const count = Number(process.env.N ?? 10000), dimension = Number(process.env.D ?? 384), queries = Number(process.env.Q ?? 30);
const options = collectionSchema({ index: { m: 16, efConstruction: 160, efSearch: 128 } }).index;
const index = new VectorIndex(dimension, 'cosine', options);
let seed = 42;
const random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296) - 0.5;
const start = performance.now();
for (let i = 0; i < count; i++) index.add(`v${i}`, prepareVector(Array.from({ length: dimension }, random), 'cosine'));
const buildMs = performance.now() - start;
const times: number[] = [], exactTimes: number[] = [];
let recall = 0;
for (let i = 0; i < queries; i++) {
  const query = prepareVector(Array.from({ length: dimension }, random), 'cosine');
  let timer = performance.now();
  const exact = await index.search(query, { topK: 10, exact: true }); exactTimes.push(performance.now() - timer);
  timer = performance.now();
  const ann = await index.search(query, { topK: 10, efSearch: 128 }); times.push(performance.now() - timer);
  const ids = new Set(exact.map(item => item.id)); recall += ann.filter(item => ids.has(item.id)).length / 10;
}
const percentile = (values: number[], p: number) => { values.sort((a, b) => a - b); return Number(values[Math.floor((values.length - 1) * p)].toFixed(2)); };
console.log(JSON.stringify({ count, dimension, queries, buildMs: Math.round(buildMs), annP50Ms: percentile(times, .5), annP95Ms: percentile(times, .95), exactP50Ms: percentile(exactTimes, .5), exactP95Ms: percentile(exactTimes, .95), recallAt10: Number((recall / queries).toFixed(4)), stats: index.stats(), environment: 'Node JavaScript kernel (not browser SIMD, no database/RPC overhead)' }, null, 2));
