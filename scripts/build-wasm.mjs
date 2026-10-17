import { spawnSync } from 'node:child_process';
import { copyFile, mkdir } from 'node:fs/promises';

const result = spawnSync('cargo', ['build', '--release', '--target', 'wasm32-unknown-unknown', '--manifest-path', 'native/Cargo.toml'], {
  stdio: 'inherit', env: { ...process.env, RUSTFLAGS: '-C target-feature=+simd128' }, shell: false,
});
if (result.status !== 0) process.exit(result.status ?? 1);
await mkdir('dist', { recursive: true });
await copyFile('native/target/wasm32-unknown-unknown/release/serenedb_distance.wasm', 'dist/distance.wasm');
