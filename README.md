# SereneDB

A worker-native, transactional vector retrieval engine for the browser. Version 2 replaces the original localStorage-cache prototype with authoritative IndexedDB storage, packed float32 vectors, a real HNSW graph, optional WebAssembly SIMD, and an optional Transformers.js embedding adapter.

## Build And Run

```sh
npm install
npm run build
npm run dev
```

Open `http://127.0.0.1:4173`. The default demo is lexical and downloads no model. Semantic mode runs a pinned MiniLM ONNX model locally; document text is not sent to an embedding service.

For the optional SIMD kernel, install Rust's `wasm32-unknown-unknown` target and run:

```sh
rustup target add wasm32-unknown-unknown
npm run build:wasm
npm run build
```

JavaScript search remains available when SIMD is unavailable or its asset cannot load. Building the JavaScript package does not require Rust.

## Quick Start

```js
import { SereneDB } from 'serenedb';
import { transformersEmbedder } from 'serenedb/transformers';

const db = await SereneDB.open('knowledge');
const documents = await db.collection('documents', {
  embedder: transformersEmbedder({
    model: 'Xenova/all-MiniLM-L6-v2',
    revision: '751bff37182d3f1213fa05d7196b954e230abad9',
    dtype: 'q8',
    device: 'wasm',
    pooling: 'mean',
    normalize: true,
  }),
  index: { m: 16, efConstruction: 160, efSearch: 128 },
});

await documents.upsert([
  {
    id: 'browser-storage',
    text: 'IndexedDB stores structured data persistently in the browser.',
    metadata: { category: 'engineering' },
  },
]);

const response = await documents.search({
  text: 'How can my web app store knowledge offline?',
  topK: 10,
  mode: 'hybrid',
  filter: { category: { $eq: 'engineering' } },
});

console.log(response.results, response.revision, response.mode);
await documents.checkpoint();
await db.close();
```

For static browser hosting, import `./dist/index.js` and `./dist/transformers.js`. Deploy the complete `dist` directory, including worker/chunk modules, optional `distance.wasm`, and model-runtime vendor assets. Do not open the demo through `file://`.

## Raw Vectors

No model runtime is imported until text encoding is requested.

```js
const db = await SereneDB.open('vectors');
const vectors = await db.collection('items', {
  dimension: 384,
  metric: 'cosine',
  index: 'hnsw',
});

await vectors.upsert({
  id: 'item-1',
  vector: new Float32Array(384).fill(0.1),
  metadata: { tags: ['local'] },
});

const { results } = await vectors.search({
  vector: new Float32Array(384).fill(0.1),
  mode: 'exact',
  topK: 5,
});
```

Vectors must be finite, dense rank-one numeric arrays or `Float32Array`/`Float64Array` values with dimension 1..16384. Cosine vectors are normalized on ingestion/query and zero-norm vectors are rejected. Dot/L2 vectors retain their float32 magnitudes. Metadata is owned, finite JSON-compatible plain data; functions, cycles, dates, and BigInt are rejected.

Raw vectors in a model-backed collection must explicitly include `space`, equal to `(await collection.state()).schema.embedding.fingerprint`. Query vectors use the same field. Matching dimensions alone does not establish an embedding space. This assertion remains the external encoder's responsibility.

## Search

| Option | Behavior |
| --- | --- |
| `mode: 'auto'` | Exact for small/flat collections and dense (256+ dimensional) collections up to 20000 records; HNSW otherwise. |
| `mode: 'exact'` | Exhaustive reference using bounded heap top-k; SIMD when available. |
| `mode: 'ann'` | HNSW where eligible; small/flat/filtered queries explicitly report exact fallback. |
| `mode: 'hybrid'` | BM25 plus vector rank fusion (RRF); lexical-only when no embedder/query vector. |
| `efSearch` | ANN recall/latency tradeoff, at least the requested top-k internally. |
| `filter` | Applied before selecting nearest neighbors; currently forces exact vector search. |
| `minScore` | Optional higher-is-better score cutoff; no implicit zero cutoff. |
| `maxDistance` | Optional nonnegative Euclidean distance cutoff. |
| `includeVectors` | Return independently owned float32 buffers; off by default. |
| `mmr` | Optional 0..1 relevance/diversity weight over an expanded shortlist. |
| `signal` | Cancellation before commit or while searching/encoding. |

Cosine/dot scores are higher-is-better; cosine is clamped to [-1,1]. Euclidean results contain raw `distance` and `score = -distance`. Hybrid scores are RRF scores, not cosine or relevance probabilities. Default `topK` is 10; zero returns no results. Exact ties are ordered by ID. SIMD float32 accumulation has ordinary floating-point tolerance relative to JavaScript's double accumulation.

Supported metadata filters: equality, `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$exists`, `$contains`, `$and`, `$or`, `$not`, and dotted nested fields. `id` and `text` are also filterable. Unknown operators fail explicitly.

## Collections And Lifecycle

- `db.collection(name, options)` creates or opens a persisted collection. Explicit incompatible metric, dimension, embedding space, or index settings reject.
- `collection.upsert(recordOrArray, {signal, onProgress})` commits one atomic batch. Duplicate IDs use their last supplied value.
- `collection.ingest(iterable, {batchSize, signal, onProgress})` bounds streaming input; each batch is independently committed.
- `collection.get(id)`, `has(id)`, `count()`, and `state()` read authoritative state without loading a graph.
- `collection.delete(idOrIds)` and `clear()` maintain durable counts/revisions transactionally.
- `collection.scan({offset, limit, filter, includeVectors})` and `iterate()` traverse a revision-refreshed working set. Iteration is not a multi-page transaction snapshot during concurrent writes.
- `collection.checkpoint()` stores a generation/revision-tagged derived index. A changed revision rejects stale checkpoint publication.
- `collection.compact()` rebuilds the graph and removes tombstones.
- `collection.unload()` releases its working set/model; committed records remain intact.
- `collection.preload()` loads and warms its embedding model/WASM assets.
- `collection.drop()` invalidates existing handles via a new collection generation on recreation.
- `db.export()` returns a versioned JSON-compatible backup from committed records; each collection is internally consistent, not a database-wide MVCC snapshot.
- `db.import(export, {replace})` validates all input first, gates destination generations/schemas, then atomically replaces/upserts each collection's records. Creating a new collection precedes its record transaction; a database-wide import is not one transaction.
- `db.close()` drains queued operations, optionally checkpoints indexes, and releases resources.
- `db.destroy()` commits terminal root invalidation and clears records before physical IndexedDB deletion. Peers close on version change.

Every operation checks durable generation/revision rather than treating notifications as proof of freshness. Searches report the revision of the consistent snapshot they searched. A concurrent later write does not invalidate an earlier search result's snapshot.

## Durability And Recovery

Records, schema, count, revision, and index invalidation commit in one IndexedDB transaction. A mutation resolves only on transaction completion. Strict durability is requested by default and falls back only where the browser does not support that transaction option. Choose relaxed durability explicitly with `SereneDB.open(name, {durability: 'relaxed'})`.

Memory, graph checkpoints, and SIMD matrices are derived state. Invalid/stale checkpoints are rebuilt from records. Exact retrieval loads packed vectors without constructing an unused HNSW graph; ANN or an explicit checkpoint builds the graph lazily. There is no localStorage vector cache, unload save dependency, or stale emergency-snapshot replay.

If the worker/process dies before an acknowledgement reaches the caller, a transaction may already have committed. Reopen and inspect the operation's IDs; retrying an upsert is idempotent by ID. Cancellation arriving after commit does not undo the write.

IndexedDB deletion cannot be canceled. After the terminal destructive transaction commits, a blocked timeout or cancellation only stops waiting; deletion may finish later. The connection remains closed and stale peers cannot repopulate it.

Browser origin eviction, quota exhaustion, user clearing, hardware failure, and malicious code sharing the origin remain outside database guarantees. `SereneDB.storageInfo()` and `requestPersistence()` expose browser storage controls. Keep external backups for irreplaceable data.

## Embedding Contract

The optional adapter loads AutoTokenizer/AutoModel and a Transformers.js feature-extraction pipeline with the configured revision. It validates pooled `[batch, dimension]` tensors and clones finite CPU rows rather than flattening arbitrary model tensors. Document/query prefixes, pooling, normalization, dtype, truncation, and model revision form the collection's embedding-space fingerprint.

`device: 'auto'` probes WebGPU and falls back to WASM without changing dtype. A model repository must actually contain that dtype's ONNX artifacts. Model-weight quantization does not quantize stored embeddings: vectors remain float32.

Batch size and padded token-slot budgets bound inference. `maxTokens` is injected into tokenization, and right-padded last-token pooling avoids mixed-length batches. Inputs are limited to 131072 UTF-16 characters including their prefix. Prefix/pooling choices must follow the selected model card. Pin immutable revision SHAs or versioned local model directories; mutable `main` cannot guarantee unchanged weights.

Native ONNX inference is not interruptible. An aborted caller receives an abort error; late output is discarded and the native inference stays serialized until completion. Shared runtime environment conflicts reject rather than silently changing another model's settings.

For offline use, deploy the application/runtime assets, preload the model, and retain browser Cache Storage. `allowRemoteModels: false` supports cache/local-only model loading; `localModelPath` supports self-hosted model directories. Preloading is not a guarantee against subsequent browser eviction. A fully offline page needs application-shell caching as well as model caching.

## Performance And Limits

HNSW is implemented, not a configuration-only placeholder. Packed rows use four bytes per component, a bounded heap avoids sorting every qualifying record, and worker execution keeps retrieval/inference away from the UI thread. Optional SIMD accelerates both bulk exact scoring and random-access HNSW distances, with a JavaScript fallback.

Search needs its active collection in memory. This is not an out-of-core Arrow/Lance engine or a claim of measured superiority to LanceDB. Filters use exact fallback, and highly selective indexed ANN filters remain future work.

Input batches are bounded to 100000 records and an estimated 64 MiB payload; prefer streaming smaller batches. Queues reject after 128 pending operations. At most two collection working sets and one model adapter are resident by default (`maxResidentCollections` and `maxResidentModels` are configurable). Worker operations time out after five minutes by default (`operationTimeoutMs` is configurable); a timeout closes the worker and may leave an unknown commit status. Model assets and graph/index snapshot copies require additional memory.

The graph limits packed payload to 512 MiB, one million physical nodes, and bounded links/IDs; actual browser-safe capacities can be much smaller. Tombstones are removed through compaction. 100000 x 384 float32 components alone consume about 146.5 MiB, excluding metadata, models, graph links, snapshots, and the optional SIMD copy.

Benchmarks are seeded and executable; compare recall and total memory, not only latency:

```sh
npm run benchmark
```

Set `N`, `D`, and `Q` environment variables to change corpus size, dimension, and query count. The command measures the Node JavaScript index kernel, not browser SIMD, IndexedDB, or worker RPC. Browser tests separately measure end-to-end recall and responsiveness.

The larger browser benchmark is opt-in: set `SERENEDB_BENCHMARK=1`, then run `npx playwright test --project=edge tests/browser/performance.spec.ts`. It reports an exact/ANN recall-latency curve for 10000 x 384 random vectors. High-dimensional unstructured data may require a large beam, and exact SIMD can outperform ANN at high recall. Do not assume a default HNSW setting meets a universal recall guarantee.

Measured on the development machine, headless Edge, 10000 x 384 seeded independent random vectors, 20 queries, M16/efConstruction128:

| Search | Recall@10 | P50 | P95 |
| --- | --- | --- | --- |
| SIMD exact | 1.000 | 4.41 ms | 18.79 ms |
| HNSW ef64 | 0.445 | 3.37 ms | 19.75 ms |
| HNSW ef128 | 0.665 | 4.35 ms | 24.83 ms |
| HNSW ef256 | 0.840 | 5.64 ms | 29.94 ms |
| HNSW ef512 | 0.955 | 6.91 ms | 39.50 ms |
| HNSW ef1024 | 0.990 | 9.22 ms | 17.69 ms |

Durable ingestion took 1.31 seconds, the first exact query including cold vector loading 444 ms, and graph construction plus checkpoint 23.1 seconds. Tail timings varied noticeably across runs; these include worker/RPC overhead and are not comparisons against LanceDB. A separate 1500 x 64 workload achieved 0.9917 recall at ef128 across the four browser engines. Mobile devices, larger corpora, and realistic embedding distributions require their own measurements.

## Tests

```sh
npm run typecheck
npm test
npx playwright install chromium firefox webkit
npm run build
npm run test:browser
npm run test:model
```

The Edge project requires Microsoft Edge installed. Real-model tests fetch a pinned public model on their first run, then block external networking to verify cached restart inference. Unit tests use fake-indexeddb and injected embedding runtimes; browser tests exercise real IndexedDB, workers, SIMD, transaction fault injection, multiple pages, graph recovery, filters, and ANN recall.

The GitHub Actions workflow builds the SIMD kernel and runs typechecks, units, Chromium, Firefox, and WebKit. Real model downloads and the larger dense benchmark remain opt-in/local checks rather than network-dependent CI gates.

## Version 1 Migration

The old constructor/synchronous-search API is intentionally replaced by async collections. Opening an existing version-1 database aborts the upgrade without changing its records because the old format lacked trustworthy metric/embedding provenance.

```js
import { migrateLegacy } from 'serenedb/legacy';
await migrateLegacy('old-db', 'new-db', 'cosine');
```

Migration requires an explicit metric and different destination name. It reads the legacy `vectors` store without modifying it, validates through the new API, and never restores stale localStorage emergency data. Malformed legacy rows fail rather than being silently discarded.

## License

MIT. Transformers.js, ONNX Runtime, and downloaded models retain their own licenses; verify model licensing for your application.
