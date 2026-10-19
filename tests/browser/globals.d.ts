import type { SereneDB, Collection } from '../../src/index.js';
import type { transformersEmbedder } from '../../src/transformers.js';
declare global {
  interface Window {
    SereneDB: typeof SereneDB;
    transformersEmbedder: typeof transformersEmbedder;
    db: SereneDB;
    docs: Collection;
  }
}
export {};
