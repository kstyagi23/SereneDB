import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.map': 'application/json' };
const port = Number(process.env.PORT ?? 4173);
http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    const relative = decodeURIComponent(url.pathname === '/' ? '/test.html' : url.pathname);
    const filename = path.resolve(root, `.${relative}`);
    if (!filename.startsWith(`${root}${path.sep}`)) throw new Error('Invalid path');
    const data = await readFile(filename);
    response.writeHead(200, {
      'Content-Type': mime[path.extname(filename)] ?? 'application/octet-stream',
      'Cache-Control': 'no-cache',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    });
    response.end(data);
  } catch { response.writeHead(404); response.end('Not found'); }
}).listen(port, '127.0.0.1', () => console.log(`SereneDB: http://127.0.0.1:${port}`));
