// Reliable viewer screenshot: drive headless Edge over the DevTools protocol
// and capture only after the scene has actually loaded, instead of relying on
// --screenshot (which fires on the load event, long before a large GLB is
// fetched and parsed).
//
//   node scripts/shot.mjs <out.png> "<query e.g. az=35&el=28&dist=0.5>" [width] [height]
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [out, query = "", W = "1600", H = "1000"] = process.argv.slice(2);
const EDGE = process.env.BROWSER || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const PORT = 9333;
const profile = mkdtempSync(join(tmpdir(), "edgeshot-"));
const edge = spawn(EDGE, [
  "--headless=new", "--disable-gpu-sandbox", "--enable-unsafe-webgpu",
  "--use-angle=swiftshader", "--no-first-run", "--disable-extensions",
  `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  `--window-size=${W},${H}`, "about:blank",
], { stdio: "ignore" });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function targets() {
  for (let i = 0; i < 60; i++) {
    try { return await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); }
    catch { await sleep(500); }
  }
  throw new Error("DevTools endpoint never came up");
}

const page = (await targets()).find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  else if (m.method === "Runtime.consoleAPICalled") {
    const txt = m.params.args.map((a) => a.value ?? a.description ?? "").join(" ");
    if (m.params.type === "error" || m.params.type === "warning") console.log(`[${m.params.type}]`, txt.slice(0, 300));
  } else if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    console.log("[exception]", (d.exception?.description || d.text || "").slice(0, 400));
  }
});
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params }));
});

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride",
  { width: +W, height: +H, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: `http://127.0.0.1:8000/?${query}` });

const t0 = Date.now();
let ready = false;
while (Date.now() - t0 < 15 * 60 * 1000) {
  const r = await send("Runtime.evaluate", { expression:
    `(() => { const e = document.getElementById("loading");
              return e ? [getComputedStyle(e).display, e.textContent] : ["missing", ""]; })()`,
    returnByValue: true });
  const [disp, txt] = r.result?.result?.value || ["", ""];
  if (((Date.now() - t0) / 1000 | 0) % 30 < 2) console.log("t", ((Date.now() - t0) / 1000 | 0), disp, txt.slice(0, 60));
  if (disp === "none") { ready = true; break; }
  if (/fail/i.test(txt)) { console.error("viewer error:", txt); break; }
  await sleep(2000);
}
console.log(ready ? `loaded in ${((Date.now() - t0) / 1000).toFixed(0)} s` : "not loaded");
// Give the post chain and shadow map time to settle over several frames.
await sleep(ready ? 20000 : 0);
const dbg = await send("Runtime.evaluate", {
  expression: "window.__dwDebug ? JSON.stringify(window.__dwDebug()) : 'no probe'",
  returnByValue: true });
console.log("state", dbg.result?.result?.value);
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.result.data, "base64"));
console.log("wrote", out);
ws.close();
// Kill the whole browser process tree (GPU/renderer children outlive edge.kill()).
spawn("taskkill", ["/F", "/T", "/PID", String(edge.pid)], { stdio: "ignore" });
await sleep(1500);
process.exit(0);
