// DepthWizard viewer interface.
//
// Loaded by index.html's module script after the scene and render loop exist.
// It owns everything the user sees around the 3D view:
//   left  -- the place sheet: facts about the scene that is open, every one
//            read from scene.json (built from the same rasters the DSM uses);
//   right -- places (radio list) and the four actions: RGB, Confidence,
//            Fly-through, LiDAR;
//   cursor -- a reticle and two numbers: the estimated height above ground
//            at that spot and its 1-sigma uncertainty, read from exact
//            rasters (hover_agl.u16 / hover_sigma.u16), never from the mesh.
// The LiDAR action splits the view with a movable horizontal line: the
// estimate above it, airborne LiDAR below it, both standing on the same
// ground so only the heights differ.

export function installDWX(ctx) {
  const { THREE, scene, camera, renderer, BASE, getMeta, getGround, getRGB,
          setConfidenceTexture, setTextureMode, tour } = ctx;

  // ---------------------------------------------------------------- styles
  const css = document.createElement("style");
  css.textContent = `
  body.dwx #ui, body.dwx .dw-chrome, body.dwx #hover-readout, body.dwx #compare-panel,
  body.dwx #compare-toggle, body.dwx #help, body.dwx #btn-help { display: none !important; }
  .dwx-card { position: fixed; z-index: 30; color: #e8eef2;
    background: linear-gradient(180deg, rgba(14,20,30,.86), rgba(10,14,22,.86));
    border: 1px solid rgba(242,181,68,.18); border-radius: 14px;
    box-shadow: 0 18px 50px rgba(0,0,0,.35); backdrop-filter: blur(10px);
    font: 13px/1.5 "Segoe UI Variable", "Segoe UI", system-ui, sans-serif; }
  .dwx-num { font-family: "Cascadia Mono", "JetBrains Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums; }
  #dwx-sheet { left: 16px; top: 16px; width: 286px; max-height: calc(100vh - 32px);
    overflow-y: auto; padding: 16px 16px 12px; }
  #dwx-sheet .place { font-size: 19px; font-weight: 650; letter-spacing: -.01em; margin: 0; }
  #dwx-sheet .where { color: #9fb0bf; font-size: 12px; margin: 2px 0 12px; }
  .dwx-chip { display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 99px;
    border: 1px solid currentColor; margin-right: 6px; }
  .dwx-group { border-top: 1px solid rgba(255,255,255,.07); padding: 10px 0 4px; }
  .dwx-group h4 { margin: 0 0 6px; font-size: 12px; font-weight: 600; color: #f2b544; }
  .dwx-row { display: flex; justify-content: space-between; gap: 12px; padding: 2px 0; }
  .dwx-row span:first-child { color: #a9b8c6; }
  .dwx-row span:last-child { text-align: right; }
  .dwx-foot { color: #7f91a1; font-size: 11.5px; margin: 4px 0 2px; }
  #dwx-dock { right: 16px; top: 16px; width: 248px; padding: 14px; }
  #dwx-dock h4 { margin: 0 0 8px; font-size: 12px; font-weight: 600; color: #f2b544; }
  .dwx-land { color: #7f91a1; font-size: 11px; margin: 8px 0 2px; text-transform: lowercase; }
  .dwx-radio { display: flex; align-items: center; gap: 9px; padding: 5px 8px; border-radius: 9px;
    cursor: pointer; }
  .dwx-radio:hover { background: rgba(255,255,255,.05); }
  .dwx-radio input { accent-color: #f2b544; margin: 0; }
  .dwx-radio .lid { margin-left: auto; font-size: 10px; color: #5ec8b8; }
  .dwx-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px;
    border-top: 1px solid rgba(255,255,255,.07); padding-top: 12px; }
  .dwx-btn { appearance: none; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04);
    color: #e8eef2; border-radius: 10px; padding: 9px 8px; font: 600 12.5px system-ui, sans-serif;
    cursor: pointer; transition: background .15s, border-color .15s; }
  .dwx-btn:hover:not(:disabled) { border-color: rgba(242,181,68,.6); }
  .dwx-btn.on { background: #f2b544; color: #111820; border-color: #f2b544; }
  .dwx-btn:disabled { opacity: .38; cursor: not-allowed; }
  .dwx-btn:focus-visible, .dwx-radio input:focus-visible { outline: 2px solid #5ec8b8; outline-offset: 2px; }
  #dwx-legend { margin-top: 10px; display: none; }
  #dwx-legend .bar { height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #1b8a6b, #7fbf4f, #f2d046, #f28b30, #d7303f); }
  #dwx-legend .ticks { display: flex; justify-content: space-between; color: #9fb0bf; font-size: 11px; }
  #dwx-reticle { position: fixed; z-index: 45; pointer-events: none; display: none;
    width: 22px; height: 22px; margin: -11px 0 0 -11px; border-radius: 50%;
    border: 1.5px solid #f2b544; box-shadow: 0 0 0 1px rgba(0,0,0,.45); }
  #dwx-reticle::before, #dwx-reticle::after { content: ""; position: absolute; background: #f2b544; }
  #dwx-reticle::before { left: 10px; top: -6px; width: 1.5px; height: 6px; }
  #dwx-reticle::after { top: 10px; left: 22px; width: 6px; height: 1.5px; }
  #dwx-tag { position: fixed; z-index: 46; pointer-events: none; display: none;
    padding: 6px 11px 7px; border-radius: 10px; background: rgba(10,14,22,.92);
    border: 1px solid rgba(242,181,68,.35); white-space: nowrap; }
  #dwx-tag .h { font-size: 18px; font-weight: 650; color: #fff; }
  #dwx-tag .u { font-size: 12.5px; margin-left: 8px; }
  #dwx-tag .k { display: block; font-size: 10.5px; color: #8fa2b3; margin-top: 1px; }
  #dwx-wipe { position: fixed; left: 0; right: 0; z-index: 25; height: 0; display: none;
    border-top: 2px solid #5ec8b8; box-shadow: 0 0 12px rgba(94,200,184,.6); pointer-events: none; }
  #dwx-wipe .lbl { position: absolute; left: 50%; transform: translateX(-50%); font: 600 11px system-ui;
    padding: 2px 9px; border-radius: 99px; background: rgba(10,14,22,.85); color: #e8eef2; }
  #dwx-wipe .up { bottom: 6px; } #dwx-wipe .dn { top: 6px; color: #5ec8b8; }
  #dwx-rail { position: fixed; right: 278px; top: 16px; bottom: 16px; width: 26px; z-index: 31;
    display: none; cursor: ns-resize; }
  #dwx-rail .track { position: absolute; left: 12px; top: 0; bottom: 0; width: 2px;
    background: rgba(255,255,255,.18); border-radius: 1px; }
  #dwx-rail .knob { position: absolute; left: 1px; width: 24px; height: 24px; margin-top: -12px;
    border-radius: 50%; background: #5ec8b8; border: 2px solid #0e141e; box-shadow: 0 2px 10px rgba(0,0,0,.4); }
  #dwx-rail:focus-visible .knob { outline: 2px solid #f2b544; }
  #dwx-note { position: fixed; left: 50%; bottom: 18px; transform: translateX(-50%); z-index: 32;
    display: none; padding: 7px 14px; }
  @media (prefers-reduced-motion: reduce) { .dwx-btn { transition: none; } }
  `;
  document.head.appendChild(css);
  document.body.classList.add("dwx");

  const el = (tag, attrs = {}, html = "") => {
    const e = document.createElement(tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (html) e.innerHTML = html;
    return e;
  };
  const fmt = (x, d = 1) => (typeof x === "number" && isFinite(x)) ? x.toFixed(d) : "—";
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const sheet = el("div", { id: "dwx-sheet", class: "dwx-card", role: "region", "aria-label": "About this place" });
  const dock = el("div", { id: "dwx-dock", class: "dwx-card", role: "region", "aria-label": "Places and views" });
  const reticle = el("div", { id: "dwx-reticle" });
  const tag = el("div", { id: "dwx-tag", class: "dwx-num" });
  const wipe = el("div", { id: "dwx-wipe" }, '<span class="lbl up">estimate</span><span class="lbl dn">LiDAR</span>');
  const rail = el("div", { id: "dwx-rail", tabindex: "0", role: "slider", "aria-label": "LiDAR split position",
                            "aria-valuemin": "0", "aria-valuemax": "100" },
                  '<div class="track"></div><div class="knob"></div>');
  const note = el("div", { id: "dwx-note", class: "dwx-card" });
  [sheet, dock, reticle, tag, wipe, rail, note].forEach((x) => document.body.appendChild(x));
  const say = (text, ms = 3200) => {
    note.textContent = text; note.style.display = "block";
    clearTimeout(say._t); say._t = setTimeout(() => { note.style.display = "none"; }, ms);
  };

  // ---------------------------------------------------------------- places
  const LAND_COLOURS = { urban: "#f2b544", hilly: "#c89bf2", forested: "#7fbf4f", sparse: "#e8a37a" };
  const currentId = new URLSearchParams(location.search).get("scene") || window.__DW_SCENE || null;
  let places = [];

  async function loadPlaces() {
    try {
      const r = await fetch("/viewer/places.json", { cache: "no-store" });
      places = r.ok ? await r.json() : [];
    } catch (_) { places = []; }
    const avail = await Promise.all(places.map(async (p) => {
      if (p.id === currentId) return true;
      try { return (await fetch(`/scenes/${encodeURIComponent(p.id)}/scene.json`, { cache: "no-store" })).ok; }
      catch (_) { return false; }
    }));
    places = places.filter((_, i) => avail[i]);
  }

  function renderDock(meta) {
    const hasLidar = !!(meta && meta.hover && meta.hover.lidar);
    const hasSigma = !!(meta && meta.hover && meta.hover.sigma);
    const groups = {};
    places.forEach((p) => (groups[p.landscape] = groups[p.landscape] || []).push(p));
    let html = "<h4>Places</h4>";
    if (!places.length) html += '<div class="dwx-foot">This copy holds one place.</div>';
    for (const land of ["urban", "hilly", "forested", "sparse"]) {
      if (!groups[land]) continue;
      html += `<div class="dwx-land">${land}</div>`;
      for (const p of groups[land]) {
        html += `<label class="dwx-radio"><input type="radio" name="dwx-place" value="${esc(p.id)}"`
             + `${p.id === currentId ? " checked" : ""}> <span>${esc(p.name)}</span>`
             + `${p.lidar ? '<span class="lid">LiDAR</span>' : ""}</label>`;
      }
    }
    html += '<div class="dwx-actions">'
         + '<button class="dwx-btn on" id="dwx-rgb" title="The original image on the 3D model">RGB</button>'
         + `<button class="dwx-btn" id="dwx-conf" ${hasSigma ? "" : "disabled"} title="${hasSigma
              ? "Colour every spot by its uncertainty (1 sigma, metres)" : "No uncertainty was computed for this place"}">Confidence</button>`
         + '<button class="dwx-btn" id="dwx-fly" title="Automatic flight around the place">Fly-through</button>'
         + `<button class="dwx-btn" id="dwx-lidar" ${hasLidar ? "" : "disabled"} title="${hasLidar
              ? "Split the view: estimate above the line, airborne LiDAR below. Drag the handle." :
              "No airborne LiDAR exists for this place"}">LiDAR</button>`
         + '<button class="dwx-btn" id="dwx-measured" style="grid-column: span 2" title="Show only what was '
         + 'measured from the image: hides drawn storeys and windows and the modelled 3D tree shapes">'
         + "Measured only</button>"
         + "</div>"
         + '<div id="dwx-legend"><div class="bar"></div><div class="ticks dwx-num">'
         + "<span>±0 m</span><span>±2</span><span>±4</span><span>±6</span><span>±8+</span></div></div>";
    dock.innerHTML = html;
    dock.querySelectorAll('input[name="dwx-place"]').forEach((r) => r.addEventListener("change", () => {
      const q = new URLSearchParams(location.search);
      q.set("scene", r.value);
      location.search = q.toString();
    }));
  }

  function renderSheet(meta) {
    const f = (meta && meta.facts) || {};
    const p = places.find((x) => x.id === currentId) || {};
    const land = p.landscape || "";
    const km = (m) => m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
    let html = `<p class="place">${esc(p.name || meta?.name || "This place")}</p>`
      + `<div class="where">${land ? `<span class="dwx-chip" style="color:${LAND_COLOURS[land] || "#9fb0bf"}">${land}</span>` : ""}`
      + `${esc(p.region || "")}</div>`;
    html += '<div class="dwx-group"><h4>Ground in view</h4>';
    if (f.ground_m) html += `<div class="dwx-row"><span>Footprint</span><span class="dwx-num">${km(f.ground_m[0])} × ${km(f.ground_m[1])}</span></div>`;
    if (f.pixel_m) html += `<div class="dwx-row"><span>Image pixel</span><span class="dwx-num">${fmt(f.pixel_m, 2)} m${f.pixel_assumed ? " (assumed)" : ""}</span></div>`;
    if (f.height_p99_m !== undefined) html += `<div class="dwx-row"><span>Tall end (99th pct)</span><span class="dwx-num">${fmt(f.height_p99_m)} m</span></div>`;
    if (f.above_2p5m_share !== undefined) html += `<div class="dwx-row"><span>Standing above 2.5 m</span><span class="dwx-num">${Math.round(f.above_2p5m_share * 100)} %</span></div>`;
    const td = meta && meta.texture_display;
    if (td && td.default) {
      const orig = new URLSearchParams(location.search).get("tex") === "orig";
      const q = new URLSearchParams(location.search); q.set("tex", orig ? "hires" : "orig");
      html += `<div class="dwx-foot">Texture: ${orig ? "the image as uploaded" : "sharpened ×2 from your image (display only; every height reads the original pixels)"} · <a href="?${q}" style="color:inherit">${orig ? "show sharpened" : "show original"}</a></div>`;
    }
    html += "</div>";
    html += '<div class="dwx-group"><h4>Uncertainty</h4>';
    html += `<div class="dwx-row"><span>Typical ± on objects (1σ)</span><span class="dwx-num">${fmt(f.sigma_median_m, 2)} m</span></div>`;
    html += '<div class="dwx-foot">Every height carries its own ±, calibrated on held-out airborne LiDAR. Hover anywhere to read both.</div></div>';
    if (f.lidar) {
      const L = f.lidar;
      html += '<div class="dwx-group"><h4>Measured against LiDAR</h4>'
        + `<div class="dwx-row"><span>RMSE</span><span class="dwx-num">${fmt(L.rmse_m, 2)} m</span></div>`
        + `<div class="dwx-row"><span>Mean abs. error</span><span class="dwx-num">${fmt(L.mae_m, 2)} m</span></div>`
        + `<div class="dwx-row"><span>Bias</span><span class="dwx-num">${L.bias_m >= 0 ? "+" : ""}${fmt(L.bias_m, 2)} m</span></div>`
        + `<div class="dwx-row"><span>Agreement (r)</span><span class="dwx-num">${fmt(L.r, 3)}</span></div>`
        + `<div class="dwx-row"><span>Tallest: ours / LiDAR</span><span class="dwx-num">${fmt(L.tallest_estimate_m)} / ${fmt(L.tallest_lidar_m)} m</span></div>`
        + `<div class="dwx-foot">${esc(L.source)}; every pixel, nothing fitted to the LiDAR.</div></div>`;
    }
    html += '<div class="dwx-group"><h4>Position</h4>';
    if (f.centre_lonlat) html += `<div class="dwx-row"><span>Centre</span><span class="dwx-num">${fmt(f.centre_lonlat[1], 4)}°, ${fmt(f.centre_lonlat[0], 4)}°</span></div>`;
    if (f.crs) html += `<div class="dwx-row"><span>Map grid</span><span class="dwx-num">${esc(f.crs)}</span></div>`;
    if (!f.centre_lonlat && !f.crs && Object.keys(f).length && !(meta && meta.affine_transform))
      html += '<div class="dwx-foot">The image carried no coordinates: heights are relative (rDSM).</div>';
    html += "</div>";
    // A still (&dwxshot=) is captured by a software renderer whose frame rate
    // says nothing about the viewer on real hardware -- leave it out there.
    if (!new URLSearchParams(location.search).has("dwxshot"))
      html += '<div class="dwx-group"><h4>On this screen</h4>'
        + '<div class="dwx-row"><span>Frame rate</span><span class="dwx-num" id="dwx-fps">—</span></div></div>';
    sheet.innerHTML = html;
  }

  // ------------------------------------------------------ hover rasters
  let R = null;   // { w, h, agl: Uint16Array, sigma, lidar, ground: Float32Array }
  async function loadRasters(meta) {
    const hv = meta && meta.hover;
    if (!hv) return null;
    const get = async (name, Type) => {
      if (!name) return null;
      try {
        const r = await fetch(`${BASE}/${name}`);
        return r.ok ? new Type(await r.arrayBuffer()) : null;
      } catch (_) { return null; }
    };
    const [agl, sigma, lidar, ground] = await Promise.all([
      get(hv.agl, Uint16Array), get(hv.sigma, Uint16Array), get(hv.lidar, Uint16Array), get(hv.ground, Float32Array)]);
    if (!agl || agl.length !== hv.width * hv.height) return null;
    return { w: hv.width, h: hv.height, agl, sigma, lidar, ground };
  }
  const val = (arr, i) => (!arr || arr[i] === 65535) ? null : arr[i] / 100;

  // Ground planar map: world (x,z) in the ground mesh's local frame -> texture (u,v).
  let uvOf = null, xzOf = null;
  function fitPlanar(ground) {
    const pos = ground.geometry.attributes.position, uv = ground.geometry.attributes.uv;
    if (!uv) return;
    const n = pos.count, step = Math.max(1, Math.floor(n / 6000));
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bu = [0, 0, 0], bv = [0, 0, 0];
    for (let i = 0; i < n; i += step) {
      const r = [pos.getX(i), pos.getZ(i), 1];
      for (let a = 0; a < 3; a++) { for (let b = 0; b < 3; b++) A[a][b] += r[a] * r[b]; bu[a] += r[a] * uv.getX(i); bv[a] += r[a] * uv.getY(i); }
    }
    const solve = (M, y) => {
      const m = M.map((row, i) => [...row, y[i]]);
      for (let c = 0; c < 3; c++) {
        let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
        [m[c], m[p]] = [m[p], m[c]];
        for (let r = 0; r < 3; r++) if (r !== c) { const f = m[r][c] / m[c][c]; for (let k = c; k < 4; k++) m[r][k] -= f * m[c][k]; }
      }
      return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
    };
    const U = solve(A, bu), V = solve(A, bv);
    uvOf = (x, z) => [U[0] * x + U[1] * z + U[2], V[0] * x + V[1] * z + V[2]];
    const det = U[0] * V[1] - U[1] * V[0];
    xzOf = (u, v) => [((u - U[2]) * V[1] - U[1] * (v - V[2])) / det, (U[0] * (v - V[2]) - (u - U[2]) * V[0]) / det];
  }
  const cellOf = (u, v) => {
    if (!R || u < 0 || u > 1 || v < 0 || v > 1) return -1;
    const c = Math.min(R.w - 1, Math.round(u * (R.w - 1)));
    const r = Math.min(R.h - 1, Math.round((1 - v) * (R.h - 1)));
    return r * R.w + c;
  };

  // --------------------------------------------------------------- cursor
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let lastEvt = null, pending = false;
  const sigmaColour = (s) => s < 1.5 ? "#7fdc9a" : s < 3 ? "#d8e46a" : s < 5 ? "#f2b544" : "#f27a6a";

  function readCursor() {
    pending = false;
    const e = lastEvt; const ground = getGround();
    if (!e || !ground || !R || !uvOf) return;
    ndc.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const targets = [];
    scene.traverse((o) => { if (o.isMesh && o.visible && o !== lidarMesh && !o.userData.dwxSkip) targets.push(o); });
    if (lidarMesh && lidarOn) targets.push(lidarMesh);
    const hits = ray.intersectObjects(targets, false);
    if (!hits.length) { reticle.style.display = tag.style.display = "none"; return; }
    const local = ground.worldToLocal(hits[0].point.clone());
    const [u, v] = uvOf(local.x, local.z);
    const i = cellOf(u, v);
    if (i < 0) { reticle.style.display = tag.style.display = "none"; return; }
    const belowLine = lidarOn && e.clientY > splitY();
    let html;
    if (belowLine) {
      const h = val(R.lidar, i);
      if (h === null) { reticle.style.display = tag.style.display = "none"; return; }
      html = `<span class="h">${h.toFixed(1)} m</span><span class="u" style="color:#5ec8b8">LiDAR</span>`
           + '<span class="k">measured height above ground</span>';
    } else {
      const h = val(R.agl, i), s = val(R.sigma, i);
      if (h === null) { reticle.style.display = tag.style.display = "none"; return; }
      html = `<span class="h">${h.toFixed(1)} m</span>`
           + (s !== null ? `<span class="u" style="color:${sigmaColour(s)}">± ${s.toFixed(1)} m</span>` : "")
           + '<span class="k">height above ground · uncertainty</span>';
    }
    tag.innerHTML = html;
    reticle.style.left = e.clientX + "px"; reticle.style.top = e.clientY + "px";
    const tx = Math.min(e.clientX + 20, innerWidth - 220), ty = Math.max(8, e.clientY - 58);
    tag.style.left = tx + "px"; tag.style.top = ty + "px";
    reticle.style.display = tag.style.display = "block";
  }
  // Estimated height under a screen point (no display), for choosing stills.
  function heightAt(x, y) {
    const ground = getGround();
    if (!ground || !R || !uvOf) return null;
    ndc.set((x / innerWidth) * 2 - 1, -(y / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const targets = [];
    scene.traverse((o) => { if (o.isMesh && o.visible && o !== lidarMesh && !o.userData.dwxSkip) targets.push(o); });
    const hits = ray.intersectObjects(targets, false);
    if (!hits.length) return null;
    const local = ground.worldToLocal(hits[0].point.clone());
    const [u, v] = uvOf(local.x, local.z);
    const i = cellOf(u, v);
    return i < 0 ? null : val(R.agl, i);
  }

  renderer.domElement.addEventListener("pointermove", (e) => {
    lastEvt = e;
    if (!pending) { pending = true; requestAnimationFrame(readCursor); }
  });
  renderer.domElement.addEventListener("pointerleave", () => { reticle.style.display = tag.style.display = "none"; });

  // ------------------------------------------------------------ confidence
  function buildSigmaTexture() {
    if (!R || !R.sigma) return null;
    const c = document.createElement("canvas"); c.width = R.w; c.height = R.h;
    const g = c.getContext("2d"); const img = g.createImageData(R.w, R.h);
    const stops = [[0, [27, 138, 107]], [2, [127, 191, 79]], [4, [242, 208, 70]], [6, [242, 139, 48]], [8, [215, 48, 63]]];
    const ramp = (s) => {
      if (s >= 8) return stops[4][1];
      for (let k = 0; k < 4; k++) if (s <= stops[k + 1][0]) {
        const t = (s - stops[k][0]) / (stops[k + 1][0] - stops[k][0]);
        return stops[k][1].map((a, j) => a + t * (stops[k + 1][1][j] - a));
      }
      return stops[4][1];
    };
    for (let i = 0; i < R.w * R.h; i++) {
      const s = R.sigma[i];
      const col = s === 65535 ? [40, 44, 52] : ramp(s / 100);
      img.data[i * 4] = col[0]; img.data[i * 4 + 1] = col[1]; img.data[i * 4 + 2] = col[2]; img.data[i * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8;
    return t;
  }

  // ---------------------------------------------------------------- LiDAR
  let lidarMesh = null, lidarOn = false, split = 0.5;
  const planeTop = new THREE.Plane(), planeBot = new THREE.Plane();
  const splitY = () => 16 + split * (innerHeight - 32);

  function buildLidarMesh(ground) {
    if (!R || !R.lidar || !R.ground || !xzOf) return null;
    const step = Math.max(1, Math.ceil(Math.max(R.w, R.h) / 512));
    const gw = Math.floor((R.w - 1) / step) + 1, gh = Math.floor((R.h - 1) / step) + 1;
    const pos = new Float32Array(gw * gh * 3), uvs = new Float32Array(gw * gh * 2);
    for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) {
      const rr = Math.min(R.h - 1, r * step), cc = Math.min(R.w - 1, c * step);
      const i = rr * R.w + cc, k = r * gw + c;
      const u = cc / (R.w - 1), v = 1 - rr / (R.h - 1);
      const [x, z] = xzOf(u, v);
      const l = val(R.lidar, i);
      pos[k * 3] = x; pos[k * 3 + 1] = R.ground[i] + (l === null ? 0 : l); pos[k * 3 + 2] = z;
      uvs[k * 2] = u; uvs[k * 2 + 1] = v;
    }
    const idx = [];
    for (let r = 0; r < gh - 1; r++) for (let c = 0; c < gw - 1; c++) {
      const a = r * gw + c, b = a + 1, d = a + gw, e2 = d + 1;
      idx.push(a, d, b, b, d, e2);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(idx); geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({ map: getRGB(), roughness: 0.95, metalness: 0,
                                                 side: THREE.DoubleSide, clippingPlanes: [planeBot] });
    const m = new THREE.Mesh(geo, mat);
    m.userData.dwxSkip = true; m.visible = false;
    ground.add(m);
    return m;
  }

  function updateSplitPlanes() {
    const y = 1 - 2 * (splitY() / innerHeight);
    const c = camera.position.clone();
    const a = new THREE.Vector3(-1, y, 0.5).unproject(camera);
    const b = new THREE.Vector3(1, y, 0.5).unproject(camera);
    const n = new THREE.Vector3().subVectors(a, c).cross(new THREE.Vector3().subVectors(b, c)).normalize();
    planeTop.setFromNormalAndCoplanarPoint(n, c);
    const probe = new THREE.Vector3(0, Math.min(1, y + 0.2), 0.5).unproject(camera);
    if (planeTop.distanceToPoint(probe) < 0) planeTop.negate();
    planeBot.copy(planeTop).negate();
  }

  const clipped = new Set();
  function setLidar(on) {
    lidarOn = on;
    const ground = getGround();
    if (on && !lidarMesh) lidarMesh = buildLidarMesh(ground);
    if (on && !lidarMesh) { lidarOn = false; say("LiDAR could not be placed on this scene."); return false; }
    scene.traverse((o) => {
      if (!o.isMesh || o === lidarMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      mats.forEach((m) => { if (!m) return; m.clippingPlanes = on ? [planeTop] : null; m.needsUpdate = true; clipped.add(m); });
    });
    if (lidarMesh) lidarMesh.visible = on;
    renderer.localClippingEnabled = on;
    wipe.style.display = rail.style.display = on ? "block" : "none";
    placeKnob();
    return true;
  }
  function placeKnob() {
    wipe.style.top = splitY() + "px";
    rail.querySelector(".knob").style.top = (split * 100) + "%";
    rail.setAttribute("aria-valuenow", String(Math.round(split * 100)));
  }
  const dragTo = (clientY) => {
    const r = rail.getBoundingClientRect();
    split = Math.min(1, Math.max(0, (clientY - r.top) / r.height));
    placeKnob();
  };
  rail.addEventListener("pointerdown", (e) => {
    rail.setPointerCapture?.(e.pointerId); dragTo(e.clientY);
    const move = (ev) => dragTo(ev.clientY);
    const up = () => { rail.removeEventListener("pointermove", move); rail.removeEventListener("pointerup", up); };
    rail.addEventListener("pointermove", move); rail.addEventListener("pointerup", up);
  });
  rail.addEventListener("keydown", (e) => {
    if (e.key === "ArrowUp") { split = Math.max(0, split - 0.03); placeKnob(); e.preventDefault(); }
    if (e.key === "ArrowDown") { split = Math.min(1, split + 0.03); placeKnob(); e.preventDefault(); }
  });

  // ------------------------------------------------------------- actions
  let sigmaTex = null;
  function wireActions() {
    const rgbB = document.getElementById("dwx-rgb"), confB = document.getElementById("dwx-conf");
    const flyB = document.getElementById("dwx-fly"), lidB = document.getElementById("dwx-lidar");
    const legend = document.getElementById("dwx-legend");
    const view = (mode) => {
      if (mode === "confidence") {
        sigmaTex = sigmaTex || buildSigmaTexture();
        if (!sigmaTex) { say("No uncertainty raster for this place."); return; }
        if (lidarOn) { setLidar(false); lidB.classList.remove("on"); }
        setConfidenceTexture(sigmaTex);
      }
      setTextureMode(mode);
      rgbB.classList.toggle("on", mode === "rgb");
      confB.classList.toggle("on", mode === "confidence");
      legend.style.display = mode === "confidence" ? "block" : "none";
    };
    rgbB.addEventListener("click", () => view("rgb"));
    confB.addEventListener("click", () => view(confB.classList.contains("on") ? "rgb" : "confidence"));
    flyB.addEventListener("click", () => {
      const on = tour.toggle();
      flyB.classList.toggle("on", on);
    });
    const measB = document.getElementById("dwx-measured");
    measB.addEventListener("click", () => {
      const on = !measB.classList.contains("on");
      if (window.__dwSetMeasuredOnly) window.__dwSetMeasuredOnly(on);
      measB.classList.toggle("on", on);
      if (on) say("Measured only: heights, outlines, roofs and the photo. Drawn windows and tree shapes hidden.", 4200);
    });
    lidB.addEventListener("click", () => {
      const want = !lidB.classList.contains("on");
      if (want && confB.classList.contains("on")) view("rgb");
      if (setLidar(want) || !want) lidB.classList.toggle("on", want && lidarOn);
      if (want && lidarOn) say("Drag the handle on the right up or down: estimate above the line, LiDAR below.", 4200);
    });
  }

  // --------------------------------------------------------------- frame
  let fpsAcc = 0, fpsN = 0, lastT = performance.now();
  window.__dwxFrame = () => {
    const t = performance.now(); fpsAcc += (t - lastT) / 1000; fpsN++; lastT = t;
    if (fpsAcc >= 1) {
      const f = document.getElementById("dwx-fps");
      const v = Math.round(fpsN / fpsAcc);
      if (f && v > 0) f.textContent = `${v} fps`;       // "—" until a real measurement
      fpsAcc = 0; fpsN = 0;
    }
    if (lidarOn) updateSplitPlanes();
  };
  addEventListener("resize", placeKnob);

  // ---------------------------------------------------------------- start
  (async () => {
    await loadPlaces();
    let meta = getMeta();
    for (let i = 0; !meta && i < 100; i++) { await new Promise((r) => setTimeout(r, 100)); meta = getMeta(); }
    renderSheet(meta); renderDock(meta); wireActions();
    R = await loadRasters(meta);
    let ground = getGround();
    for (let i = 0; !ground && i < 600; i++) { await new Promise((r) => setTimeout(r, 100)); ground = getGround(); }
    if (ground) fitPlanar(ground);
    window.__dwxReady = { rasters: !!R, sigma: !!(R && R.sigma), lidar: !!(R && R.lidar), planar: !!uvOf };
    // Stills for documents: &dwxpanels=0 hides the side panels, and
    // &dwxshot=fx,fy pins the cursor readout at that fraction of the screen.
    const q = new URLSearchParams(location.search);
    if (q.get("dwxpanels") === "0") { sheet.style.display = "none"; dock.style.display = "none"; }
    if (q.get("dwxshot")) {
      await new Promise((r) => setTimeout(r, 1500));
      let fx, fy;
      if (q.get("dwxshot") === "auto") {
        // Pin the readout on the tallest structure in the middle of the view.
        let best = -1;
        for (let y = 0.48; y <= 0.82; y += 0.02) for (let x = 0.30; x <= 0.72; x += 0.02) {
          const v = heightAt(x * innerWidth, y * innerHeight);
          if (v !== null && v > best && v < 150) { best = v; fx = x; fy = y; }
        }
      } else {
        [fx, fy] = q.get("dwxshot").split(",").map(Number);
      }
      if (fx !== undefined) window.__dwxProbe(fx * innerWidth, fy * innerHeight);
      window.__dwxShotDone = tag.textContent;
    }
    // &dwxtest=1: sample the readout on a grid and write it into the page, so
    // a headless browser can check the numbers without a mouse.
    if (new URLSearchParams(location.search).get("dwxtest") === "1") {
      await new Promise((r) => setTimeout(r, 1500));
      const out = [JSON.stringify(window.__dwxReady)];
      for (const fy of [0.35, 0.5, 0.65, 0.8]) for (const fx of [0.3, 0.5, 0.7]) {
        out.push(`${fx},${fy}: ${window.__dwxProbe(fx * innerWidth, fy * innerHeight)}`);
      }
      const pre = document.createElement("pre"); pre.id = "dwx-test"; pre.textContent = out.join("\n");
      pre.style.display = "none"; document.body.appendChild(pre);
    }
  })();

  // Test hooks: read the value under a screen point, and drive the controls.
  window.__dwxProbe = (x, y) => { lastEvt = { clientX: x, clientY: y }; readCursor(); return tag.textContent; };
  window.__dwxSetSplit = (s) => { split = s; placeKnob(); };
}
