// Photo minimap + inspector.
//
// Bottom right: the source photograph with the camera's position and view
// cone, so the 3D view can always be related back to the image it came from.
// Click it (or press M) for the inspector: the photo, the height field, the
// per-pixel uncertainty, or a swipe between photo and height -- zoom, pan, a
// scale bar, and a cursor readout of height and its +/- read from the same
// rasters the 3D cursor uses (hover_agl.u16 / hover_sigma.u16).

const RAMP = [[19, 50, 79], [42, 111, 176], [94, 200, 184], [242, 181, 68], [232, 134, 42], [192, 57, 43]];
function ramp(t) {
  t = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i, a = RAMP[i], b = RAMP[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

export async function installMinimap({ BASE, meta, getGround }) {
  if (!meta || !meta.hover) return null;
  const hv = meta.hover;
  const W = hv.width, H = hv.height;
  const get = async (f) => {
    const r = await fetch(`${BASE}/${f}`, { cache: "force-cache" });
    if (!r.ok) throw new Error(f);
    return new Uint16Array(await r.arrayBuffer());
  };
  let agl = null, sig = null;
  try { agl = await get(hv.agl); } catch (e) { return null; }
  try { if (hv.sigma) sig = await get(hv.sigma); } catch (e) { sig = null; }
  const val = (a, i) => (a && a[i] !== 65535 ? a[i] / 100 : null);

  // ground extent in world units, for camera -> image mapping
  let box = null;
  const ext = () => {
    const g = getGround();
    if (!g) return null;
    if (!box) {
      g.geometry.computeBoundingBox();
      box = g.geometry.boundingBox.clone().applyMatrix4(g.matrixWorld);
    }
    return box;
  };
  const toUV = (x, z) => {
    const b = ext(); if (!b) return null;
    return [(x - b.min.x) / (b.max.x - b.min.x), (b.max.z - z) / (b.max.z - b.min.z)];
  };

  // --- coloured rasters (height, +/-) -------------------------------------
  const p99 = (a) => {
    const v = []; for (let i = 0; i < a.length; i += 13) if (a[i] !== 65535) v.push(a[i]);
    v.sort((x, y) => x - y); return (v[Math.floor(v.length * 0.99)] || 100) / 100;
  };
  const hiH = Math.max(5, p99(agl)), hiS = sig ? Math.max(2, p99(sig)) : 1;
  function rasterCanvas(a, hi) {
    const c = document.createElement("canvas"); c.width = W; c.height = H;
    const ctx = c.getContext("2d"), im = ctx.createImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const o = i * 4;
      if (a[i] === 65535) { im.data[o + 3] = 0; continue; }
      const [r, g, b] = ramp(a[i] / 100 / hi);
      im.data[o] = r; im.data[o + 1] = g; im.data[o + 2] = b; im.data[o + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
    return c;
  }
  let heightCanvas = null, sigmaCanvas = null;

  // --- styles ----------------------------------------------------------------
  const css = document.createElement("style");
  css.textContent = `
  #dwm { position: fixed; right: 16px; bottom: 14px; z-index: 31; width: 150px; height: 150px; border-radius: 14px; overflow: hidden;
         border: 1px solid rgba(255,255,255,.14); box-shadow: 0 14px 30px -14px rgba(0,0,0,.6); cursor: pointer; background: #11181f; }
  #dwm img, #dwm canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  #dwm .lb { position: absolute; left: 8px; bottom: 6px; font: 700 10.5px "Cascadia Mono", ui-monospace, monospace; letter-spacing: .08em;
             color: #fff; text-shadow: 0 1px 3px #000; }
  #dwm:focus-visible { outline: 2px solid #5ec8b8; outline-offset: 2px; }
  #dwi { position: fixed; inset: 24px; z-index: 60; display: none; flex-direction: column; border-radius: 16px; overflow: hidden;
         background: #0d1318; border: 1px solid rgba(255,255,255,.1); box-shadow: 0 30px 80px rgba(0,0,0,.6); color: #e8eef2;
         font: 13px "Segoe UI", system-ui, sans-serif; }
  #dwi header { display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,.08); }
  #dwi header b { font-size: 15px; } #dwi header .dim { color: #8fa1b0; font: 12px "Cascadia Mono", ui-monospace, monospace; }
  #dwi .seg { margin-left: auto; display: flex; gap: 2px; background: rgba(255,255,255,.05); border-radius: 10px; padding: 3px; }
  #dwi .seg button, #dwi .tb { appearance: none; border: 0; background: transparent; color: #c9d5de; font: 600 12.5px inherit; padding: 6px 12px; border-radius: 8px; cursor: pointer; }
  #dwi .seg button.on { background: #f2b544; color: #1b1300; }
  #dwi .tb { border: 1px solid rgba(255,255,255,.12); }
  #dwi .stage { position: relative; flex: 1; overflow: hidden; cursor: grab; background:
         repeating-conic-gradient(#121a21 0% 25%, #0f161c 0% 50%) 50% / 24px 24px; }
  #dwi .stage.drag { cursor: grabbing; }
  #dwi .layer { position: absolute; left: 0; top: 0; transform-origin: 0 0; image-rendering: auto; }
  #dwi .layer img, #dwi .layer canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  #dwi .swipe { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; cursor: ew-resize; display: none; }
  #dwi .swipe::after { content: "\\2194"; position: absolute; top: 50%; left: -15px; width: 32px; height: 32px; margin-top: -16px;
         border-radius: 50%; background: #fff; color: #11181f; display: grid; place-items: center; font-size: 15px; }
  #dwi footer { display: flex; align-items: center; gap: 18px; padding: 10px 16px; border-top: 1px solid rgba(255,255,255,.08);
         font: 12.5px "Cascadia Mono", ui-monospace, monospace; color: #c9d5de; }
  #dwi footer .legend { margin-left: auto; display: flex; align-items: center; gap: 8px; }
  #dwi footer .bar { width: 160px; height: 8px; border-radius: 4px; }
  #dwi .scale { position: absolute; left: 16px; bottom: 14px; font: 11px "Cascadia Mono", monospace; color: #fff; text-shadow: 0 1px 2px #000; }
  #dwi .scale i { display: block; height: 4px; border: 2px solid #fff; border-top: 0; margin-bottom: 3px; }`;
  document.head.appendChild(css);

  // --- minimap ---------------------------------------------------------------
  const mini = document.createElement("div");
  mini.id = "dwm"; mini.tabIndex = 0; mini.setAttribute("role", "button");
  mini.title = "Source photograph -- click or press M to compare photo, height and uncertainty";
  const thumbSrc = `${BASE}/thumb.jpg`;
  mini.innerHTML = `<img src="${thumbSrc}" alt=""><canvas width="150" height="150"></canvas><span class="lb">SOURCE</span>`;
  document.body.appendChild(mini);
  const mc = mini.querySelector("canvas"), mx = mc.getContext("2d");
  const dir = { x: 0, z: 0 };
  function drawCone() {
    const cam = window.__dwCamera, ctl = window.__dwControls;
    if (!cam || !ctl) return;
    const p = toUV(cam.position.x, cam.position.z), t = toUV(ctl.target.x, ctl.target.z);
    mx.clearRect(0, 0, 150, 150);
    if (!p || !t) return;
    const S = 150, px = p[0] * S, py = p[1] * S;
    let a = Math.atan2(t[1] - p[1], t[0] - p[0]);
    const half = (cam.fov || 50) * Math.PI / 360 * cam.aspect * 0.8;
    const len = 62;
    mx.fillStyle = "rgba(242,181,68,.28)"; mx.strokeStyle = "rgba(242,181,68,.9)"; mx.lineWidth = 1.5;
    mx.beginPath(); mx.moveTo(px, py);
    mx.lineTo(px + Math.cos(a - half) * len, py + Math.sin(a - half) * len);
    mx.lineTo(px + Math.cos(a + half) * len, py + Math.sin(a + half) * len);
    mx.closePath(); mx.fill(); mx.stroke();
    mx.fillStyle = "#f2b544"; mx.beginPath(); mx.arc(Math.min(S - 4, Math.max(4, px)), Math.min(S - 4, Math.max(4, py)), 4.5, 0, 7); mx.fill();
    mx.strokeStyle = "#11181f"; mx.lineWidth = 1.5; mx.stroke();
  }
  (function loop() { drawCone(); setTimeout(() => requestAnimationFrame(loop), 120); })();

  // --- inspector -------------------------------------------------------------
  const pixM = (meta.facts && meta.facts.pixel_m) || null;
  const groundW = (meta.facts && meta.facts.ground_m && meta.facts.ground_m[0]) || null;
  const dlg = document.createElement("div");
  dlg.id = "dwi"; dlg.setAttribute("role", "dialog"); dlg.setAttribute("aria-label", "Photo, height and uncertainty inspector");
  const name = (meta._saved_description || "This place").split(",")[0];
  dlg.innerHTML = `<header><b>${name}</b><span class="dim">${groundW ? groundW + " m wide" : ""}</span>
     <div class="seg" role="tablist"><button data-m="photo" class="on">Photo</button><button data-m="height">Height</button>
     ${sig ? '<button data-m="sigma">&plusmn; Uncertainty</button>' : ""}<button data-m="swipe">Swipe photo | height</button></div>
     <button class="tb" data-z="fit" title="Fit">Fit</button><button class="tb" data-z="close" title="Close (Esc)">&times;</button></header>
     <div class="stage"><div class="layer"><img alt="source photograph"><canvas class="hl"></canvas></div>
     <div class="swipe"></div><div class="scale"><i></i><span></span></div></div>
     <footer><span class="rd">Point at the image to read height</span><span class="legend"><span class="lo">0 m</span><canvas class="bar" width="160" height="8"></canvas><span class="hi"></span></span></footer>`;
  document.body.appendChild(dlg);
  const stage = dlg.querySelector(".stage"), layer = dlg.querySelector(".layer");
  const img = layer.querySelector("img"), hl = layer.querySelector("canvas.hl");
  const swipe = dlg.querySelector(".swipe"), rd = dlg.querySelector(".rd");
  const bar = dlg.querySelector("canvas.bar"), bctx = bar.getContext("2d");
  for (let x = 0; x < 160; x++) { const [r, g, b] = ramp(x / 159); bctx.fillStyle = `rgb(${r|0},${g|0},${b|0})`; bctx.fillRect(x, 0, 1, 8); }
  let mode = "photo", zoom = 1, ox = 0, oy = 0, split = 0.5, natW = W, natH = H, imgLoaded = false;

  function applyView() {
    layer.style.width = natW + "px"; layer.style.height = natH + "px";
    layer.style.transform = `translate(${ox}px, ${oy}px) scale(${zoom})`;
    const sw = stage.clientWidth;
    if (mode === "swipe") {
      const cut = split * sw;
      hl.style.clipPath = `inset(0 0 0 ${Math.max(0, (cut - ox) / zoom)}px)`;
      swipe.style.left = cut + "px";
    } else hl.style.clipPath = "none";
    // scale bar: a round number of metres about 120 px long
    if (pixM) {
      const mPerScreenPx = (W * pixM / natW) / zoom;
      const target = 120 * mPerScreenPx, p10 = Math.pow(10, Math.floor(Math.log10(target)));
      const nice = [1, 2, 5, 10].map((k) => k * p10).reduce((a, b) => Math.abs(b - target) < Math.abs(a - target) ? b : a);
      dlg.querySelector(".scale i").style.width = (nice / mPerScreenPx) + "px";
      dlg.querySelector(".scale span").textContent = nice >= 1000 ? nice / 1000 + " km" : nice + " m";
    }
  }
  function fit() {
    const sw = stage.clientWidth, sh = stage.clientHeight;
    zoom = Math.min(sw / natW, sh / natH) * 0.96;
    ox = (sw - natW * zoom) / 2; oy = (sh - natH * zoom) / 2;
    applyView();
  }
  function setMode(m) {
    mode = m;
    dlg.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("on", b.dataset.m === m));
    if ((m === "height" || m === "swipe") && !heightCanvas) heightCanvas = rasterCanvas(agl, hiH);
    if (m === "sigma" && !sigmaCanvas && sig) sigmaCanvas = rasterCanvas(sig, hiS);
    const src = m === "sigma" ? sigmaCanvas : (m === "photo" ? null : heightCanvas);
    hl.style.display = src ? "block" : "none";
    if (src) { hl.width = W; hl.height = H; hl.getContext("2d").drawImage(src, 0, 0); }
    swipe.style.display = m === "swipe" ? "block" : "none";
    dlg.querySelector(".hi").textContent = (m === "sigma" ? "±" + hiS.toFixed(1) : hiH.toFixed(0)) + " m";
    dlg.querySelector(".lo").textContent = m === "sigma" ? "±0 m" : "0 m";
    applyView();
  }
  function open() {
    dlg.style.display = "flex";
    if (!imgLoaded) {
      imgLoaded = true;
      const tf = meta.texture_display && meta.texture_display.default ? "terrain_texture_display.jpg" : (meta.texture_file || "terrain_texture.png");
      img.onload = () => { natW = img.naturalWidth; natH = img.naturalHeight; fit(); };
      img.onerror = () => { img.src = thumbSrc; };
      img.src = `${BASE}/${tf}`;
    }
    setMode(mode); fit();
  }
  const close = () => { dlg.style.display = "none"; };
  mini.addEventListener("click", open);
  mini.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
  dlg.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.m)));
  dlg.querySelector('[data-z="fit"]').addEventListener("click", fit);
  dlg.querySelector('[data-z="close"]').addEventListener("click", close);
  window.addEventListener("keydown", (e) => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === "Escape" && dlg.style.display === "flex") close();
    if ((e.key === "m" || e.key === "M") && !e.ctrlKey && !e.metaKey) (dlg.style.display === "flex" ? close : open)();
  });
  stage.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = stage.getBoundingClientRect(), mxp = e.clientX - r.left, myp = e.clientY - r.top;
    const k = Math.exp(-e.deltaY * 0.0015), nz = Math.min(40, Math.max(0.05, zoom * k));
    ox = mxp - (mxp - ox) * (nz / zoom); oy = myp - (myp - oy) * (nz / zoom); zoom = nz; applyView();
  }, { passive: false });
  let drag = null;
  stage.addEventListener("pointerdown", (e) => {
    if (e.target === swipe) { drag = { swipe: true }; } else drag = { x: e.clientX - ox, y: e.clientY - oy };
    stage.classList.add("drag"); stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointerup", () => { drag = null; stage.classList.remove("drag"); });
  stage.addEventListener("pointermove", (e) => {
    const r = stage.getBoundingClientRect();
    if (drag && drag.swipe) { split = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)); applyView(); }
    else if (drag) { ox = e.clientX - drag.x; oy = e.clientY - drag.y; applyView(); }
    // readout: screen -> image pixel -> raster cell
    const ix = (e.clientX - r.left - ox) / zoom, iy = (e.clientY - r.top - oy) / zoom;
    const u = ix / natW, v = iy / natH;
    if (u < 0 || v < 0 || u >= 1 || v >= 1) { rd.textContent = "Point at the image to read height"; return; }
    const i = Math.floor(v * H) * W + Math.floor(u * W);
    const h = val(agl, i), s = val(sig, i);
    const xm = pixM ? (u * W * pixM).toFixed(1) : (u * 100).toFixed(1) + "%";
    const zm = pixM ? (v * H * pixM).toFixed(1) : (v * 100).toFixed(1) + "%";
    rd.textContent = h === null ? "no data here" :
      `x ${xm} m   y ${zm} m   height ${h.toFixed(1)} m` + (s !== null ? `  ± ${s.toFixed(1)} m` : "");
  });
  new ResizeObserver(() => { if (dlg.style.display === "flex") fit(); }).observe(stage);
  return { open, close };
}
