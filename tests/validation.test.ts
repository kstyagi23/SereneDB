import test from 'node:test';
import assert from 'node:assert/strict';
import { prepareVector, prepareMetadata, collectionSchema } from '../src/validation.js';
import { compileFilter } from '../src/filter.js';
import { LexicalIndex } from '../src/lexical.js';
import type { StoredRecord } from '../src/types.js';

test('finite rank-one positive-dimensional vectors and stable normalization', () => {
  for (const input of [[], [[1, 2]], [NaN], [Infinity], ['1'], [null], Array(2)]) assert.throws(() => prepareVector(input, 'cosine'));
  assert.throws(() => prepareVector([0, 0], 'cosine'));
  assert.throws(() => prepareVector([1, 2], 'cosine', 3));
  assert.deepEqual(prepareVector([1e200], 'cosine'), new Float32Array([1]));
  assert.deepEqual(prepareVector([1e-200], 'cosine'), new Float32Array([1]));
  const input = new Float32Array([1, 0]);
  const output = prepareVector(input, 'cosine'); input[0] = 0;
  assert.equal(output[0], 1);
  assert.deepEqual(prepareVector([0, 0], 'euclidean'), new Float32Array([0, 0]));
  assert.throws(() => prepareVector([1e200], 'dot'));
});

test('metadata is plain finite JSON and owns its values', () => {
  const input = { tags: ['one'], score: 1, nested: { value: null } };
  const cloned = prepareMetadata(input); input.tags.push('two');
  assert.deepEqual(cloned.tags, ['one']);
  const cycle: Record<string, unknown> = {}; cycle.self = cycle;
  for (const metadata of [cycle, { date: new Date() }, { value: 1n }, { value: undefined }, { value: NaN }, { fn: () => 1 }]) assert.throws(() => prepareMetadata(metadata));
});

test('collection configuration does not accept inactive optimization switches', () => {
  assert.throws(() => collectionSchema({ metric: 'bad' as never }));
  assert.throws(() => collectionSchema({ index: { m: 0 } }));
  assert.throws(() => collectionSchema({ dimension: 0 }));
});

const record: StoredRecord = { collection: 'docs', id: 'one', vector: new Float32Array([1]), timestamp: 1, text: 'vector storage', metadata: { category: 'db', number: 4, nested: { enabled: true }, tags: ['local', 'fast'] } };

test('filters enforce boolean, membership, range, nested and missing semantics', () => {
  for (const filter of [
    { category: 'db' }, { 'nested.enabled': true }, { number: { $gte: 4, $lt: 5 } },
    { tags: { $contains: 'local' } }, { category: { $in: ['db', 'other'] } },
    { absent: { $exists: false } }, { $and: [{ category: 'db' }, { $not: { number: 3 } }] },
  ]) assert.equal(compileFilter(filter)(record), true);
  assert.equal(compileFilter({ category: 'other' })(record), false);
  assert.equal(compileFilter({ absent: null })(record), false);
  assert.throws(() => compileFilter({ category: { $wat: 1 } }));
  assert.throws(() => compileFilter({ '__proto__.x': 1 }));
});

test('BM25 supports updates deletes and prefilters', () => {
  const index = new LexicalIndex();
  index.add(record);
  index.add({ ...record, id: 'two', text: 'baking bread' });
  assert.equal(index.search('vector', 10, () => true)[0].id, 'one');
  assert.equal(index.search('vector', 10, id => id !== 'one').length, 0);
  index.add({ ...record, text: 'unrelated food' });
  assert.equal(index.search('vector', 10, () => true).length, 0);
  index.remove('two');
  assert.equal(index.search('bread', 10, () => true).length, 0);
});
