import { invalid } from './errors.js';
import type { Filter, StoredRecord } from './types.js';

function equal(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === undefined || b === undefined) return false;
  if (Array.isArray(a) !== Array.isArray(b)) return false;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((value, i) => equal(value, b[i]));
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const left = Object.keys(a), right = Object.keys(b);
    return left.length === right.length && left.every(key => Object.hasOwn(b, key) && equal((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]));
  }
  return false;
}

export function compileFilter(filter?: Filter): (record: StoredRecord) => boolean {
  if (filter === undefined) return () => true;
  let nodes = 0;
  const compile = (input: unknown, depth: number): ((record: StoredRecord) => boolean) => {
    if (++nodes > 256 || depth > 16 || !input || typeof input !== 'object' || Array.isArray(input)) invalid('Invalid filter or filter complexity limit exceeded.');
    const clauses = Object.entries(input).map(([field, condition]) => {
      if (field === '$and' || field === '$or' || field === '$not') {
        if (field === '$not') { const match = compile(condition, depth + 1); return (record: StoredRecord) => !match(record); }
        if (!Array.isArray(condition) || !condition.length) invalid(`${field} requires a nonempty filter array.`);
        const children = condition.map(child => compile(child, depth + 1));
        return field === '$and' ? (record: StoredRecord) => children.every(child => child(record)) : (record: StoredRecord) => children.some(child => child(record));
      }
      if (field.startsWith('$')) invalid(`Unknown filter operator ${field}.`);
      const segments = field.replace(/^metadata\./, '').split('.');
      if (segments.some(segment => !segment || ['__proto__', 'constructor', 'prototype'].includes(segment))) invalid('Invalid filter field.');
      const get = (record: StoredRecord): unknown => {
        if (field === 'id') return record.id;
        if (field === 'text') return record.text;
        let value: unknown = record.metadata;
        for (const segment of segments) {
          if (!value || typeof value !== 'object' || !Object.hasOwn(value, segment)) return undefined;
          value = (value as Record<string, unknown>)[segment];
        }
        return value;
      };
      if (!condition || typeof condition !== 'object' || Array.isArray(condition) || !Object.keys(condition).some(key => key.startsWith('$'))) return (record: StoredRecord) => equal(get(record), condition);
      const tests = Object.entries(condition).map(([operator, expected]) => {
        switch (operator) {
          case '$eq': return (actual: unknown) => equal(actual, expected);
          case '$ne': return (actual: unknown) => !equal(actual, expected);
          case '$in':
          case '$nin':
            if (!Array.isArray(expected)) invalid(`${operator} requires an array.`);
            return (actual: unknown) => expected.some(value => equal(value, actual)) === (operator === '$in');
          case '$exists':
            if (typeof expected !== 'boolean') invalid('$exists requires a boolean.');
            return (actual: unknown) => (actual !== undefined) === expected;
          case '$contains': return (actual: unknown) => Array.isArray(actual) ? actual.some(value => equal(value, expected)) : typeof actual === 'string' && typeof expected === 'string' && actual.includes(expected);
          case '$gt': case '$gte': case '$lt': case '$lte':
            if (!(typeof expected === 'number' && Number.isFinite(expected)) && typeof expected !== 'string') invalid(`${operator} requires a finite number or string.`);
            return (actual: unknown) => {
              if (typeof actual !== typeof expected || (typeof actual !== 'number' && typeof actual !== 'string')) return false;
              const comparison = typeof actual === 'number' ? actual - (expected as number) : actual < (expected as string) ? -1 : actual > (expected as string) ? 1 : 0;
              return operator === '$gt' ? comparison > 0 : operator === '$gte' ? comparison >= 0 : operator === '$lt' ? comparison < 0 : comparison <= 0;
            };
          default: invalid(`Unknown filter operator ${operator}.`);
        }
      });
      return (record: StoredRecord) => tests.every(test => test(get(record)));
    });
    return record => clauses.every(clause => clause(record));
  };
  return compile(filter, 0);
}
