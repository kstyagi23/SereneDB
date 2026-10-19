import { test, expect } from '@playwright/test';

test('dense 384D browser performance and recall curve', async ({ page }) => {
  test.skip(!process.env.SERENEDB_BENCHMARK, 'Set SERENEDB_BENCHMARK=1 for the larger measured workload.');
  test.setTimeout(600000);
  await page.goto('/test.html');
  const result = await page.evaluate(async () => {
    const count = 10000, dimension = 384;
    const db = await window.SereneDB.open(`dense-${crypto.randomUUID()}`);
    const docs = await db.collection('docs', { dimension, index: { m: 16, efConstruction: 128, efSearch: 128 } });
    let seed = 77;
    const random = () => ((seed = (1664525 * seed + 1013904223) >>> 0) / 4294967296) - .5;
    const records = Array.from({ length: count }, (_, i) => ({ id: `v-${i}`, vector: Float32Array.from({ length: dimension }, random) }));
    let time = performance.now(); await docs.ingest(records, { batchSize: 1000 });
    const ingestionMs = performance.now() - time;
    time = performance.now();
    await docs.search({ vector: Float32Array.from({ length: dimension }, random), mode: 'auto', topK: 10 });
    const coldExactMs = performance.now() - time;
    time = performance.now(); await docs.checkpoint(); const indexBuildMs = performance.now() - time;
    const curves = [64, 128, 256, 512, 1024].map(efSearch => ({ efSearch, recall: 0, latency: [] as number[] }));
    const exactTimes: number[] = [];
    for (let i = 0; i < 20; i++) {
      const vector = Float32Array.from({ length: dimension }, random);
      time = performance.now(); const exact = await docs.search({ vector, mode: 'exact', topK: 10 }); exactTimes.push(performance.now() - time);
      const ids = new Set(exact.results.map(item => item.id));
      for (const curve of curves) {
        time = performance.now(); const response = await docs.search({ vector, mode: 'ann', topK: 10, efSearch: curve.efSearch }); curve.latency.push(performance.now() - time);
        curve.recall += response.results.filter(item => ids.has(item.id)).length / 10;
      }
    }
    const percentiles = (times: number[]) => { times.sort((a, b) => a - b); return { p50Ms: Number(times[Math.floor(times.length * .5)].toFixed(2)), p95Ms: Number(times[Math.floor(times.length * .95)].toFixed(2)) }; };
    const stats = await docs.stats(); await db.destroy();
    return { count, dimension, ingestionMs: Math.round(ingestionMs), coldExactMs: Math.round(coldExactMs), indexBuildMs: Math.round(indexBuildMs), exact: percentiles(exactTimes), curves: curves.map(curve => ({ efSearch: curve.efSearch, recallAt10: Number((curve.recall / 20).toFixed(4)), ...percentiles(curve.latency) })), stats };
  });
  console.log(`Dense browser benchmark ${test.info().project.name}: ${JSON.stringify(result)}`);
  expect(result.curves.at(-1)!.recallAt10).toBeGreaterThanOrEqual(.95);
});
