import test from 'node:test';
import assert from 'node:assert/strict';
import { chunkDocument } from '../src/chunking.js';

test('chunks preserve source offsets and overlapping ranges with bounded progress', () => {
  const text = Array.from({ length: 100 }, (_, i) => `word${i}`).join(' ');
  const chunks = chunkDocument({ id: 'document', text, metadata: { category: 'db' } }, { maxChars: 100, overlapChars: 16 });
  assert.ok(chunks.length > 1);
  for (const chunk of chunks) {
    assert.equal(chunk.text, text.slice(chunk.metadata!.sourceStart as number, chunk.metadata!.sourceEnd as number));
    assert.equal(chunk.metadata!.sourceId, 'document'); assert.equal(chunk.metadata!.category, 'db');
    assert.ok(chunk.text!.length <= 100);
  }
  assert.equal(chunks.at(-1)!.metadata!.sourceEnd, text.length);
  assert.throws(() => chunkDocument({ id: 'a', text }, { maxChars: 100, overlapChars: 100 }));
});
