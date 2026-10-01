// One-click Demo mode: a scripted, captioned camera tour that adapts to the
// loaded scene -- overview, the tallest building with its height readout,
// trees, water, the slope view, and back out. Press D or the Demo button;
// Esc or any click stops it. &demo=1 starts it on load (for recording).

import * as THREE from "three";

const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

export function installDemo({ camera, controls, scene, groundMesh, renderer, sceneName = "" }) {
  if (!groundMesh) return null;
  const box = new THREE.Box3().setFromObject(groundMesh);
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const R = Math.max(size.x, size.z);

  // --- scene landmarks -------------------------------------------------------
  function tallest() {
    let best = null, by = -Infinity;
    const v = new THREE.Vector3();
    scene.traverse((o) => {
      if (!o.isMesh || !/^building|^roof|prism/i.test(o.name || o.parent?.name || "")) return;
      const p = o.geometry && o.geometry.attributes.position;
      if (!p) return;
      const step = Math.max(1, Math.floor(p.count / 4000));
      for (let i = 0; i < p.count; i += step) {
        v.fromBufferAttribute(p, i); o.localToWorld(v);
        // stay well inside the tile so the shot is not at the edge
        if (Math.abs(v.x - c.x) > size.x * 0.38 || Math.abs(v.z - c.z) > size.z * 0.38) continue;
        if (v.y > by) { by = v.y; best = v.clone(); }
      }
    });
    return best;
  }
  function treeSpot() {
    const T = window.__dwTrees;
    if (!T || !T.group) return null;
    const m = T.group.children.find((x) => x.isInstancedMesh && x.count > 50);
    if (!m) return null;
    const a = m.instanceMatrix.array, v = new THREE.Vector3();
    let best = null, bd = Infinity;
    for (let i = 0; i < m.count; i += 7) {
      v.set(a[i * 16 + 12], a[i * 16 + 13], a[i * 16 + 14]); T.group.localToWorld(v);
      const d = Math.hypot(v.x - c.x - size.x * 0.15, v.z - c.z - size.z * 0.15);
      if (d < bd) { bd = d; best = v.clone(); }
    }
    return best;
  }
  function waterSpot() {
    let w = null;
    scene.traverse((o) => { if (!w && o.isMesh && /water/i.test(o.name) && o.visible) w = o; });
    if (!w) return null;
    return new THREE.Box3().setFromObject(w).getCenter(new THREE.Vector3());
  }

  // --- caption --------------------------------------------------------------
  const cap = document.createElement("div");
  cap.id = "dw-demo-caption";
  cap.style.cssText = "position:fixed;left:50%;bottom:7vh;transform:translateX(-50%);z-index:60;" +
    "max-width:min(80vw,900px);padding:12px 22px;border-radius:12px;background:rgba(12,18,24,.72);" +
    "color:#f3f6f8;font:600 clamp(16px,2.1vw,26px)/1.35 system-ui,'Segoe UI',sans-serif;text-align:center;" +
    "letter-spacing:.01em;opacity:0;transition:opacity .6s ease;pointer-events:none;backdrop-filter:blur(6px);";
  document.body.appendChild(cap);
  const btn = document.createElement("button");
  btn.id = "dw-demo-btn";
  btn.textContent = "▶ Demo";
  btn.title = "Play a guided tour of this place (D)";
  btn.style.cssText = "position:fixed;left:16px;bottom:16px;z-index:55;padding:9px 16px;border-radius:9px;" +
    "border:1px solid rgba(242,181,68,.7);background:rgba(20,28,34,.85);color:#f2b544;" +
    "font:600 14px system-ui,sans-serif;cursor:pointer;";
  document.body.appendChild(btn);
  function say(t) { cap.style.opacity = t ? "1" : "0"; if (t) cap.textContent = t; }

  // --- shots ----------------------------------------------------------------
  function shots() {
    const list = [];
    const up = size.y + R * 0.05;
    list.push({ pos: new THREE.Vector3(c.x - R * 0.15, c.y + up + R * 0.55, c.z + R * 0.78), tgt: c.clone(), dur: 6,
                say: `${sceneName ? sceneName + ": " : ""}one satellite image, rebuilt as a measured 3D model` });
    list.push({ pos: new THREE.Vector3(c.x + R * 0.72, c.y + up + R * 0.35, c.z + R * 0.25), tgt: c.clone(), dur: 7,
                say: "Every building from real footprints; heights measured from the image" });
    const t = tallest();
    if (t) {
      const g = new THREE.Vector3(t.x, Math.max(c.y, t.y - (t.y - box.min.y) * 0.5), t.z);
      list.push({ pos: new THREE.Vector3(t.x + 70, t.y + 35, t.z + 85), tgt: g, dur: 7,
                  say: "Swooping in: walls, storeys and roofs at real height" });
      list.push({ pos: new THREE.Vector3(t.x + 58, t.y + 22, t.z + 66), tgt: t.clone(), dur: 5, hover: true,
                  say: "Point anywhere: height above ground, with its own ± uncertainty" });
    }
    const tr = treeSpot();
    if (tr) list.push({ pos: new THREE.Vector3(tr.x - 55, tr.y + 28, tr.z + 40), tgt: tr.clone(), dur: 7,
                        say: "Trees: measured crowns, heights and colours" });
    const w = waterSpot();
    if (w) list.push({ pos: new THREE.Vector3(w.x + 90, w.y + 45, w.z + 110), tgt: w.clone(), dur: 6,
                       say: "Water bodies from the image itself" });
    list.push({ pos: new THREE.Vector3(c.x + R * 0.35, c.y + up + R * 0.5, c.z - R * 0.55), tgt: c.clone(), dur: 6, slope: true,
                say: "Slope view: steep, landslide-prone ground stands out" });
    list.push({ pos: new THREE.Vector3(c.x, c.y + up + R * 0.85, c.z + R * 0.9), tgt: c.clone(), dur: 6, end: true,
                say: "Download the DSM GeoTIFF: absolute heights, matched to the scoring DEM" });
    return list;
  }

  // --- player ---------------------------------------------------------------
  let playing = false, raf = 0;
  function click(id) { const b = document.getElementById(id); if (b) b.click(); }
  function hoverCentre() {
    const el = renderer.domElement, r = el.getBoundingClientRect();
    const ev = { clientX: r.left + r.width / 2, clientY: r.top + r.height / 2, bubbles: true };
    try { el.dispatchEvent(new PointerEvent("pointermove", ev)); } catch (e) { /* older browsers */ }
    el.dispatchEvent(new MouseEvent("mousemove", ev));
  }
  function stop() {
    if (!playing) return;
    playing = false; cancelAnimationFrame(raf);
    controls.enabled = true; say("");
    btn.textContent = "▶ Demo";
    click("btn-rgb"); document.getElementById("dwx-rgb")?.click();
  }
  function play() {
    if (playing) { stop(); return; }
    const S = shots();
    if (!S.length) return;
    playing = true; controls.enabled = false; btn.textContent = "■ Stop";
    let k = -1, t0 = 0, fromP = new THREE.Vector3(), fromT = new THREE.Vector3();
    const next = (now) => {
      k++;
      if (k >= S.length) { setTimeout(stop, 2500); return false; }
      fromP.copy(camera.position); fromT.copy(controls.target); t0 = now;
      say(S[k].say);
      if (S[k].slope) { click("btn-slope"); }
      if (k > 0 && S[k - 1].slope && !S[k].slope) { click("btn-rgb"); document.getElementById("dwx-rgb")?.click(); }
      return true;
    };
    const step = (now) => {
      if (!playing) return;
      if (k < 0 && !next(now)) return;
      const s = S[k], u = Math.min(1, (now - t0) / (s.dur * 1000)), e = ease(u);
      // arc: rise a little mid-move so the camera does not skim through roofs
      const lift = Math.sin(Math.PI * e) * Math.min(60, fromP.distanceTo(s.pos) * 0.15);
      camera.position.lerpVectors(fromP, s.pos, e); camera.position.y += lift;
      controls.target.lerpVectors(fromT, s.tgt, e);
      camera.lookAt(controls.target);
      controls.update();
      if (s.hover && u > 0.5) hoverCentre();
      if (u >= 1 && !next(now)) return;
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }
  btn.addEventListener("click", (e) => { e.stopPropagation(); play(); });
  window.addEventListener("keydown", (e) => {
    if (e.target && /input|textarea|select/i.test(e.target.tagName)) return;
    if (e.key === "d" || e.key === "D") play();
    if (e.key === "Escape") stop();
  });
  renderer.domElement.addEventListener("pointerdown", () => { if (playing) stop(); });
  if (new URLSearchParams(location.search).get("demo") === "1") setTimeout(play, 2500);
  return { play, stop };
}
