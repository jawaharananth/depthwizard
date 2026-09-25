// Localhost-only static server for the viewer.
//   node scripts/serve.mjs [port]
import { createServer } from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const ROOT = resolve("viewer");
const SCENES = resolve("scenes");
const PORT = +(process.argv[2] || 8000);
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".png": "image/png", ".jpg": "image/jpeg",
  ".glb": "model/gltf-binary", ".geojson": "application/geo+json", ".css": "text/css",
  ".wasm": "application/wasm", ".bin": "application/octet-stream",
};

createServer((req, res) => {
  let p = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (p.endsWith("/")) p += "index.html";
  // /scenes/<name>/... are pre-built scenes in the repo's scenes/ folder
  // (the same place server.py serves them from).
  const base = p.startsWith("/scenes/") ? SCENES : ROOT;
  const file = normalize(join(base, p.startsWith("/scenes/") ? p.slice("/scenes/".length) : p));
  if (!file.startsWith(base)) { res.writeHead(403).end(); return; }
  let st;
  try { st = statSync(file); } catch { res.writeHead(404).end(); return; }
  if (!st.isFile()) { res.writeHead(404).end(); return; }
  res.writeHead(200, {
    "Content-Type": TYPES[extname(file).toLowerCase()] || "application/octet-stream",
    "Content-Length": st.size, "Cache-Control": "no-cache",
  });
  createReadStream(file).pipe(res);
}).listen(PORT, "127.0.0.1", () => console.log(`serving ${ROOT} on http://127.0.0.1:${PORT}`));
