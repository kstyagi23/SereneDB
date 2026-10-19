export class SereneDBError extends Error {
  constructor(message: string, public readonly code = 'INVALID_OPERATION', options?: ErrorOptions) {
    super(message, options);
    this.name = 'SereneDBError';
  }
}

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException('Operation aborted.', 'AbortError');
}

export function invalid(message: string, code = 'INVALID_INPUT'): never {
  throw new SereneDBError(message, code);
}
