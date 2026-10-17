import { integer } from './validation.js';
import { invalid } from './errors.js';
import type { DocumentInput, Metadata } from './types.js';

export interface ChunkOptions {
  maxChars?: number;
  overlapChars?: number;
}

/** Character-boundary chunking retains source offsets; embedding truncation remains model-controlled. */
export function chunkDocument(document: { id: string; text: string; metadata?: Metadata }, options: ChunkOptions = {}): DocumentInput[] {
  const maxChars = options.maxChars ?? 1200, overlap = options.overlapChars ?? 160;
  integer(maxChars, 'maxChars', 64, 100000);
  integer(overlap, 'overlapChars', 0, maxChars - 1);
  if (typeof document.text !== 'string') invalid('Chunk source must contain text.');
  const output: DocumentInput[] = [];
  let start = 0;
  while (start < document.text.length) {
    let end = Math.min(start + maxChars, document.text.length);
    if (end < document.text.length) {
      const boundary = document.text.lastIndexOf(' ', end);
      if (boundary > start + maxChars * 0.6) end = boundary;
      if (end > start && /[\uD800-\uDBFF]/.test(document.text[end - 1])) end--;
    }
    const text = document.text.slice(start, end);
    if (text.trim()) output.push({ id: `${document.id}#${output.length}`, text, metadata: { ...document.metadata, sourceId: document.id, sourceStart: start, sourceEnd: end } });
    if (end === document.text.length) break;
    const next = Math.max(start + 1, end - overlap);
    start = next;
    if (/[\uDC00-\uDFFF]/.test(document.text[start])) start++;
  }
  return output;
}
