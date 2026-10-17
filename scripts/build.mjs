import { build } from 'esbuild';
import { mkdir, copyFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';

await mkdir('dist', { recursive: true });
const types = spawnSync('node', ['node_modules/typescript/bin/tsc', '--emitDeclarationOnly'], { stdio: 'inherit' });
if (types.status !== 0) process.exit(types.status ?? 1);
await build({
  entryPoints: ['src/index.ts', 'src/local.ts', 'src/worker.ts', 'src/transformers.ts'],
  outdir: 'dist', bundle: true, format: 'esm', splitting: true, sourcemap: true,
  target: ['es2022'], platform: 'browser', external: ['@huggingface/transformers'],
  plugins: [{ name: 'optional-browser-model-runtime', setup(build) {
    build.onResolve({ filter: /^@huggingface\/transformers$/ }, () => ({ path: './vendor/transformers.js', external: true }));
  } }],
});
await mkdir('dist/vendor', { recursive: true });
await copyFile('node_modules/@huggingface/transformers/dist/transformers.js', 'dist/vendor/transformers.js');
for (const asset of ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm']) await copyFile(`node_modules/onnxruntime-web/dist/${asset}`, `dist/vendor/${asset}`);
try { await copyFile('native/target/wasm32-unknown-unknown/release/serenedb_distance.wasm', 'dist/distance.wasm'); } catch { console.warn('Optional SIMD kernel not built. Run npm run build:wasm; JavaScript search remains available.'); }
