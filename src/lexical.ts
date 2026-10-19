import type { StoredRecord } from './types.js';

function tokens(text: string): string[] {
  return text.normalize('NFKC').toLowerCase().match(/[\p{L}\p{N}_]+/gu) ?? [];
}

export class LexicalIndex {
  private postings = new Map<string, Map<string, number>>();
  private documents = new Map<string, Map<string, number>>();
  private lengths = new Map<string, number>();
  private totalLength = 0;

  add(record: StoredRecord): void {
    this.remove(record.id);
    if (!record.text) return;
    const words = tokens(record.text);
    const frequencies = new Map<string, number>();
    for (const word of words) frequencies.set(word, (frequencies.get(word) ?? 0) + 1);
    for (const [word, frequency] of frequencies) {
      let posting = this.postings.get(word);
      if (!posting) this.postings.set(word, posting = new Map());
      posting.set(record.id, frequency);
    }
    this.documents.set(record.id, frequencies);
    this.lengths.set(record.id, words.length);
    this.totalLength += words.length;
  }

  remove(id: string): void {
    const frequencies = this.documents.get(id);
    if (!frequencies) return;
    for (const word of frequencies.keys()) {
      const posting = this.postings.get(word)!;
      posting.delete(id);
      if (!posting.size) this.postings.delete(word);
    }
    this.totalLength -= this.lengths.get(id) ?? 0;
    this.documents.delete(id);
    this.lengths.delete(id);
  }

  search(text: string, topK: number, filter: (id: string) => boolean): Array<{ id: string; score: number }> {
    const scores = new Map<string, number>();
    const n = this.documents.size;
    const average = this.totalLength / Math.max(n, 1);
    for (const word of new Set(tokens(text))) {
      const posting = this.postings.get(word);
      if (!posting) continue;
      const idf = Math.log(1 + (n - posting.size + 0.5) / (posting.size + 0.5));
      for (const [id, frequency] of posting) {
        if (!filter(id)) continue;
        const denominator = frequency + 1.2 * (0.25 + 0.75 * (this.lengths.get(id)! / Math.max(average, 1)));
        scores.set(id, (scores.get(id) ?? 0) + idf * frequency * 2.2 / denominator);
      }
    }
    return Array.from(scores, ([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score || a.id.localeCompare(b.id)).slice(0, topK);
  }
}
