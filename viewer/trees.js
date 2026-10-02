// Real-crown 3D trees.
//
// MEASURED: every tree's position, height, crown size, orientation and colour
// come from tree_crowns.json (tools/tree_crowns.py: individual crowns
// segmented from the Meta/WRI canopy-height map, coloured from the scene's
// own photograph).
// INFERRED: branch structure and foliage are procedural (EZ-Tree, MIT,
// vendor/ez-tree). A crown is a measured footprint and height; what its
// branches look like is not something a satellite image records.
//
// The canopy bump the ground mesh carried under each planted crown is lowered
// to the ground, so a tree stands on the street instead of inside a mound.
// Two levels of detail per variant: detailed trees near the camera, light
// ones beyond, reassigned twice a second.

import * as THREE from "three";

const NEAR_MAX = 350;          // detailed (EZ-Tree) trees: at most this many ...
const DETAIL_M = 90;           // ... and only within this distance: farther away their
                               // leaf cards alias into speckle, the light tree reads better
const UPDATE_MS = 500;
const CONIFER_ABOVE_M = 1200;  // hill stations: Shimla ~2,200 m, Gangtok ~1,600 m

const SPECIES = {
  // Aspen's white birch bark is wrong for Indian plains and tropics.
  broadleaf: [["Oak Medium", 101], ["Oak Medium", 202], ["Ash Medium", 303],
              ["Ash Medium", 404], ["Oak Large", 505]],
  conifer: [["Pine Medium", 61], ["Pine Large", 62], ["Pine Medium", 63], ["Pine Small", 64]],
  small: [["Oak Small", 11], ["Ash Small", 22], ["Bush 1", 33], ["Bush 2", 44]],
};

function fuller(o) {
  // At city scale a crown must read as a mass of foliage: larger leaf
  // clusters, a few more of them.
  o.leaves.size = o.leaves.size * 2.2;
  o.leaves.count = Math.round(o.leaves.count * 1.3);
}

// Far trees: a light lumpy crown + trunk (~250 triangles), shaded with
// position noise so it reads as foliage, not a ball. Height, crown size and
// colour are still the measured ones (instance transform + colour).
function farTreeGeometry(conifer) {
  const parts = [];
  const trunk = new THREE.CylinderGeometry(0.035, 0.05, 0.42, 6, 1, true);
  trunk.translate(0, 0.21, 0);
  parts.push([trunk, [0.24, 0.18, 0.13]]);
  if (conifer) {
    for (const [y, r, hgt] of [[0.30, 0.42, 0.42], [0.52, 0.32, 0.36], [0.72, 0.20, 0.30]]) {
      const c = new THREE.ConeGeometry(r, hgt, 9, 1);
      c.translate(0, y + hgt / 2, 0);
      parts.push([c, [1, 1, 1]]);
    }
  } else {
    for (const [x, y, z, r] of [[0, 0.66, 0, 0.42], [0.22, 0.58, 0.08, 0.30], [-0.2, 0.6, -0.1, 0.30],
                                [0.05, 0.82, -0.18, 0.26], [-0.08, 0.55, 0.24, 0.27]]) {
      // smooth, irregular lumps: jitter each vertex radially, keep smooth normals
      const ico = new THREE.IcosahedronGeometry(r, 1);
      const ip = ico.attributes.position;
      for (let i = 0; i < ip.count; i++) {
        const vx = ip.getX(i), vy = ip.getY(i), vz = ip.getZ(i);
        const j = 1 + 0.14 * Math.sin(vx * 53.1 + vy * 31.7 + vz * 47.3 + x * 11 + z * 7);
        ip.setXYZ(i, vx * j, vy * j * 0.92, vz * j);
      }
      ico.translate(x, y, z);
      parts.push([ico, [1, 1, 1]]);
    }
  }
  const pos = [], nor = [], col = [];
  for (const [g0, tint] of parts) {
    // smooth normals: merge coincident vertices first, then expand
    let g = g0;
    if (!g.index) {
      const map = new Map(), idx = [], pos2 = [];
      const pp = g.attributes.position;
      for (let i = 0; i < pp.count; i++) {
        const key = `${pp.getX(i).toFixed(4)}_${pp.getY(i).toFixed(4)}_${pp.getZ(i).toFixed(4)}`;
        let k = map.get(key);
        if (k === undefined) { k = pos2.length / 3; map.set(key, k); pos2.push(pp.getX(i), pp.getY(i), pp.getZ(i)); }
        idx.push(k);
      }
      g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos2, 3));
      g.setIndex(idx);
    }
    g.computeVertexNormals();
    g = g.toNonIndexed();
    const p = g.attributes.position, nn = g.attributes.normal;
    for (let i = 0; i < p.count; i++) {
      const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      pos.push(x, y, z); nor.push(nn.getX(i), nn.getY(i), nn.getZ(i));
      // leafy variation: darker low and inside, lighter on top
      const k = tint[0] === 1 ? 0.62 + 0.38 * Math.min(1, Math.max(0, (y - 0.35) / 0.6)) *
                (0.8 + 0.2 * Math.sin(x * 37 + z * 29 + y * 41)) : 1;
      col.push(tint[0] * k, tint[1] * k, tint[2] * k);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  return geo;   // unit tree: height ~1, crown radius ~0.5
}

function avgColour(tex) {
  // Mean opaque colour of a leaf texture (sRGB 0-1), so a crown's measured
  // colour can be divided by it to give the per-instance tint.
  try {
    const img = tex && tex.image;
    if (!img || !img.width) return [0.32, 0.45, 0.20];
    const cv = document.createElement("canvas"); cv.width = cv.height = 32;
    const c = cv.getContext("2d", { willReadFrequently: true });
    c.drawImage(img, 0, 0, 32, 32);
    const d = c.getImageData(0, 0, 32, 32).data;
    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < d.length; i += 4) if (d[i + 3] > 128) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
    return n ? [r / n / 255, g / n / 255, b / n / 255] : [0.32, 0.45, 0.20];
  } catch (e) { return [0.32, 0.45, 0.20]; }
}

const toLin = (v) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));

async function waitImage(tex, ms = 3000) {
  const t0 = performance.now();
  while (tex && !(tex.image && tex.image.width) && performance.now() - t0 < ms) {
    await new Promise((r) => setTimeout(r, 50));
  }
}

function flattenMounds(groundMesh, C) {
  // Lower the canopy bump under each crown to the crown's ground height.
  const g = groundMesh.geometry, p = g.attributes.position;
  if (!p || p.count < 4) return 0;
  let w = 1;
  while (w < p.count && p.getX(w) > p.getX(w - 1)) w++;
  const h = Math.floor(p.count / w);
  if (w < 2 || h < 2) return 0;
  const x0 = p.getX(0), dx = (p.getX(w - 1) - x0) / (w - 1);
  const zTop = p.getZ(0), dz = (zTop - p.getZ((h - 1) * w)) / (h - 1);
  if (!(dx > 0 && dz > 0)) return 0;
  const arr = p.array;
  let lowered = 0;
  for (let i = 0; i < C.n; i++) {
    if (!(C.yt[i] - C.y[i] > 0.8)) continue;              // no mound to remove here
    // Crowns cover well under half of the canopy area, so the bump is lowered
    // over a wider footprint than the crown itself, with a soft edge.
    const ra = Math.max(C.ra[i], C.r[i]) * 1.75, rb = Math.max(C.rb[i], C.r[i] * 0.6) * 1.75;
    const R = Math.max(ra, rb) + dx;
    const c0 = Math.max(0, Math.floor((C.x[i] - R - x0) / dx)), c1 = Math.min(w - 1, Math.ceil((C.x[i] + R - x0) / dx));
    const r0 = Math.max(0, Math.floor((zTop - (C.z[i] + R)) / dz)), r1 = Math.min(h - 1, Math.ceil((zTop - (C.z[i] - R)) / dz));
    const cs = Math.cos(C.rot[i]), sn = Math.sin(C.rot[i]);
    const floor = C.y[i] + 0.15;
    for (let rr = r0; rr <= r1; rr++) {
      for (let cc = c0; cc <= c1; cc++) {
        const k = (rr * w + cc) * 3;
        const ex = arr[k] - C.x[i], ez = arr[k + 2] - C.z[i];
        // three.js rotation.y: local +X maps to (cos, 0, -sin) in world x/z
        const u = (ex * cs - ez * sn) / ra, v = (ex * sn + ez * cs) / rb;
        const d = u * u + v * v;
        if (d >= 1 || arr[k + 1] <= floor) continue;
        const t = d < 0.4 ? 1 : 1 - (d - 0.4) / 0.6;         // soft edge
        arr[k + 1] = arr[k + 1] + (floor - arr[k + 1]) * t;
        lowered++;
      }
    }
  }
  if (lowered) {
    p.needsUpdate = true;
    g.computeVertexNormals();
    g.computeBoundingBox(); g.computeBoundingSphere();
  }
  return lowered;
}

export async function addTrees({ root, groundMesh, BASE, TSL, camera, quality = 1, lite = false }) {
  const _q = new URLSearchParams(location.search);
  const NEAR_CAP = _q.has("treenear") ? Math.max(0, +_q.get("treenear")) : NEAR_MAX;
  const NEAR_SHADOWS = _q.get("treeshadow") !== "0";
  const r = await fetch(`${BASE}/tree_crowns.json`, { cache: "no-store" }).catch(() => null);
  if (!r || !r.ok) return null;
  const J = await r.json();
  const n = J.count | 0;
  if (!n || !Array.isArray(J.x)) return null;
  const C = { n, x: J.x, z: J.z, y: J.y, h: J.h, r: J.r, ra: J.ra || J.r, rb: J.rb || J.r,
              rot: J.rot || new Array(n).fill(0), yt: J.yt || J.y, R: J.R, G: J.G, B: J.B };
  // The crown mean includes the crown's own shadow, and the 3D lighting shades
  // the model again, so trees came out darker than in the photograph. Lean on
  // the sunlit quartile (also measured): the leaves the sun lit at capture.
  if (Array.isArray(J.Rl) && J.Rl.length === n) {
    const mix = (a, b) => a.map((v, i) => 0.4 * v + 0.6 * b[i]);
    C.R = mix(J.R, J.Rl); C.G = mix(J.G, J.Gl); C.B = mix(J.B, J.Bl);
  }
  const datum = (J.frame && J.frame.datum_m) || 0;
  const hills = datum > CONIFER_ABOVE_M;
  if (lite) return addLightTrees({ root, groundMesh, TSL, camera, C, n, hills });

  const { Tree } = await import("./vendor/ez-tree/ez-tree.es.js");

  async function make(name, seed, light) {
    const t = new Tree();
    t.loadPreset(name);
    t.options.seed = seed;
    fuller(t.options);
    t.generate();
    const bg = t.branchesMesh.geometry, lg = t.leavesMesh.geometry;
    const bm = t.branchesMesh.material, lm = t.leavesMesh.material;
    await waitImage(lm.map); await waitImage(bm.map);
    bg.computeBoundingBox(); lg.computeBoundingBox();
    const lb = lg.boundingBox, bb = bg.boundingBox;
    const H = Math.max(lb.max.y, bb.max.y) - Math.min(0, bb.min.y);
    const R = Math.max(-lb.min.x, lb.max.x, -lb.min.z, lb.max.z, 0.5);
    return { bg, lg, barkMap: bm.map || null, barkNormal: bm.normalMap || null,
             leafMap: lm.map || null, alphaTest: lm.alphaTest || 0.5, H, R,
             leafRef: avgColour(lm.map), tris: ((bg.index ? bg.index.count : 0) + (lg.index ? lg.index.count : 0)) / 3 };
  }

  function leafMaterial(v) {
    const m = new THREE.MeshStandardNodeMaterial({ side: THREE.DoubleSide, roughness: 0.82, metalness: 0.0,
                                                   alphaTest: v.alphaTest, transparent: false });
    m.map = v.leafMap;
    if (TSL) m.positionNode = sway(v.H, 1.0);
    return m;
  }
  function barkMaterial(v) {
    const m = new THREE.MeshStandardNodeMaterial({ roughness: 0.92, metalness: 0.0 });
    m.map = v.barkMap;
    if (v.barkNormal) m.normalMap = v.barkNormal;
    if (TSL) m.positionNode = sway(v.H, 1.0);
    return m;
  }
  function sway(H, k) {
    const { positionLocal, time, sin, vec3, float } = TSL;
    const hn = positionLocal.y.div(float(H)).clamp(0, 1);
    const s = sin(time.mul(1.3).add(positionLocal.x.mul(0.07)).add(positionLocal.z.mul(0.05)))
      .mul(hn.mul(hn)).mul(float(0.015 * H * k));
    return positionLocal.add(vec3(s, float(0), s.mul(0.6)));
  }

  // --- variants -----------------------------------------------------------
  const groups = hills ? { tall: SPECIES.conifer, small: SPECIES.small.slice(2) }
                       : { tall: SPECIES.broadleaf, small: SPECIES.small };
  const variants = [];
  const farGeo = farTreeGeometry(hills);
  const farUnit = { H: 1.0, R: 0.52 };
  for (const [gname, list] of Object.entries(groups)) {
    for (const [name, seed] of list) {
      const near = await make(name, seed, false);
      variants.push({ group: gname, name, near, members: [] });
    }
  }
  const tallV = variants.filter((v) => v.group === "tall"), smallV = variants.filter((v) => v.group === "small");

  // --- assign crowns, precompute transforms and colours ---------------------
  const M = new Float32Array(n * 16), MF = new Float32Array(n * 16), COL = new Float32Array(n * 3), COLF = new Float32Array(n * 3);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < n; i++) {
    const hash = ((Math.floor(C.x[i] * 7.31) * 73856093) ^ (Math.floor(C.z[i] * 5.17) * 19349663)) >>> 0;
    const pool = C.h[i] < 4.5 && smallV.length ? smallV : tallV;
    const v = pool[hash % pool.length];
    v.members.push(i);
    const sy = C.h[i] / v.near.H;
    const sxz = Math.min(Math.max(Math.max(C.ra[i], C.r[i]) / v.near.R, sy * 0.55), sy * 1.8);
    pos.set(C.x[i], C.y[i] - 0.25, C.z[i]);
    q.setFromAxisAngle(up, (C.rot[i] || 0) + (hash % 628) / 100);
    scl.set(sxz, sy, sxz);
    m4.compose(pos, q, scl).toArray(M, i * 16);
    const fy = C.h[i] / farUnit.H, fxz = Math.min(Math.max(Math.max(C.ra[i], C.r[i]) / farUnit.R, fy * 0.35), fy * 1.4);
    scl.set(fxz, fy, fxz);
    m4.compose(pos, q, scl).toArray(MF, i * 16);
    {
      // measured crown colour, with the haze of the satellite view taken out:
      // saturation x1.35 around the crown's own grey, brightness x1.2
      const rgb = [C.R[i], C.G[i], C.B[i]].map((c) => c / 255);
      const grey = (rgb[0] + rgb[1] + rgb[2]) / 3;
      for (let k = 0; k < 3; k++) {
        const sat = Math.min(1, Math.max(0, grey + (rgb[k] - grey) * 1.35));
        COLF[i * 3 + k] = toLin(sat) * 1.2;
      }
    }
    // per-instance tint = measured crown colour / the leaf texture's own mean
    const ref = v.near.leafRef;
    for (let k = 0; k < 3; k++) {
      const meas = [C.R[i], C.G[i], C.B[i]][k] / 255;
      COL[i * 3 + k] = Math.min(Math.max(toLin(meas) / Math.max(toLin(ref[k]), 0.02), 0.25), 2.2);
    }
  }

  // --- meshes ---------------------------------------------------------------
  const group = new THREE.Group();
  group.name = "trees";
  let totalNearTris = 0;
  for (const v of variants) {
    const cap = Math.max(1, Math.min(v.members.length, NEAR_MAX));
    const mk = (lv, near) => {
      const bark = new THREE.InstancedMesh(lv.bg, barkMaterial(lv), cap);
      const leaf = new THREE.InstancedMesh(lv.lg, leafMaterial(lv), cap);
      for (const im of [bark, leaf]) {
        im.count = 0;
        im.castShadow = near && quality > 0 && NEAR_SHADOWS;
        im.receiveShadow = true;
        im.frustumCulled = false;
        im.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        group.add(im);
      }
      leaf.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(cap * 3), 3);
      return { bark, leaf };
    };
    v.nearMesh = mk(v.near, true);
    totalNearTris += v.near.tris;
  }
  // one instanced mesh draws every far tree
  const farMat = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.0 });
  if (TSL) farMat.positionNode = sway(1.0, 0.6);
  const farMesh = new THREE.InstancedMesh(farGeo, farMat, n);
  farMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  farMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  farMesh.castShadow = false; farMesh.receiveShadow = true; farMesh.frustumCulled = false;
  farMesh.count = 0;
  group.add(farMesh);
  root.add(group);

  // flatten the mounds the trees now stand in
  let lowered = 0;
  try { lowered = groundMesh ? flattenMounds(groundMesh, C) : 0; } catch (e) { console.warn("tree mound flatten skipped:", e); }

  // --- LOD assignment ---------------------------------------------------------
  let info = {};
  const camLocal = new THREE.Vector3();
  const order = new Int32Array(n), dist = new Float32Array(n), isNear = new Uint8Array(n);
  const owner = new Int32Array(n);
  variants.forEach((v, vi) => { for (const i of v.members) owner[i] = vi; });
  function assign() {
    if (!camera) return;
    camLocal.copy(camera.position);
    root.worldToLocal(camLocal);
    for (let i = 0; i < n; i++) {
      const dx = C.x[i] - camLocal.x, dz = C.z[i] - camLocal.z, dy = C.y[i] - camLocal.y;
      dist[i] = dx * dx + dz * dz + dy * dy; order[i] = i;
    }
    // the NEAR_MAX closest crowns get the detailed tree (per-variant capacity)
    const idx = Array.from(order).sort((a, b) => dist[a] - dist[b]);
    isNear.fill(0);
    const used = new Int32Array(variants.length);
    let nearTotal = 0;
    for (const i of idx) {
      if (nearTotal >= TIERS[tier].near || dist[i] > DETAIL_M * DETAIL_M) break;
      const vi = owner[i];
      const cap = variants[vi].nearMesh.leaf.instanceMatrix.count;
      if (used[vi] >= cap) continue;
      isNear[i] = 1; used[vi]++; nearTotal++;
    }
    const counts = new Int32Array(variants.length);
    const fm = farMesh.instanceMatrix.array, fcol = farMesh.instanceColor.array;
    let fc = 0;
    for (let i = 0; i < n; i++) {
      if (isNear[i]) {
        const v = variants[owner[i]], a = counts[owner[i]]++;
        const src = M.subarray(i * 16, i * 16 + 16);
        v.nearMesh.bark.instanceMatrix.array.set(src, a * 16);
        v.nearMesh.leaf.instanceMatrix.array.set(src, a * 16);
        v.nearMesh.leaf.instanceColor.array.set(COL.subarray(i * 3, i * 3 + 3), a * 3);
      } else {
        if (dist[i] > TIERS[tier].farR * TIERS[tier].farR) continue;
        fm.set(MF.subarray(i * 16, i * 16 + 16), fc * 16);
        fcol.set(COLF.subarray(i * 3, i * 3 + 3), fc * 3);
        fc++;
      }
    }
    variants.forEach((v, vi) => {
      const mesh = v.nearMesh;
      mesh.bark.count = counts[vi]; mesh.leaf.count = counts[vi];
      mesh.bark.instanceMatrix.needsUpdate = true; mesh.leaf.instanceMatrix.needsUpdate = true;
      mesh.leaf.instanceColor.needsUpdate = true;
    });
    farMesh.count = fc;
    farMesh.instanceMatrix.needsUpdate = true; farMesh.instanceColor.needsUpdate = true;
  }
  // --- automatic quality ------------------------------------------------------
  // Measures the real frame rate on THIS machine and steps tree detail down
  // below 28 fps (or back up above 50), so a basic laptop stays smooth.
  // Tier 0: 350 detailed trees with shadows | 1: 150, no tree shadows |
  // 2: light trees only | 3: light trees within 700 m only.
  const TIERS = [{ near: NEAR_CAP, shadow: NEAR_SHADOWS, farR: Infinity },
                 { near: Math.min(150, NEAR_CAP), shadow: false, farR: Infinity },
                 { near: 0, shadow: false, farR: Infinity },
                 { near: 0, shadow: false, farR: 700 }];
  let tier = _q.has("treetier") ? Math.min(3, Math.max(0, +_q.get("treetier"))) : 0;
  const autoQ = !_q.has("treetier");
  let frames = 0, t0 = performance.now(), lastChange = performance.now(), goodFor = 0;
  function applyTier() {
    for (const v of variants) {
      v.nearMesh.bark.castShadow = TIERS[tier].shadow;
      v.nearMesh.leaf.castShadow = TIERS[tier].shadow;
    }
    assign();
  }
  function frameProbe() {
    frames++;
    const now = performance.now();
    if (now - t0 >= 2000) {
      const fps = frames * 1000 / (now - t0);
      frames = 0; t0 = now;
      info.fps = Math.round(fps); info.tier = tier;
      if (autoQ && now - lastChange > 3000) {
        if (fps < 28 && tier < 3) { tier++; lastChange = now; goodFor = 0; applyTier(); }
        else if (fps > 50) { goodFor += 2; if (goodFor >= 6 && tier > 0) { tier--; lastChange = now; goodFor = 0; applyTier(); } }
        else goodFor = 0;
      }
    }
    if (!stopped) requestAnimationFrame(frameProbe);
  }
  let stopped = false;
  assign();
  requestAnimationFrame(frameProbe);
  const timer = setInterval(assign, UPDATE_MS);

  info = { crowns: n, variants: variants.length, hills, lowered_vertices: lowered,
                 near_tris_per_variant: variants.map((v) => Math.round(v.near.tris)),
                 far_tris_each: Math.round(farGeo.attributes.position.count / 3), near_max: NEAR_MAX };
  console.info("trees:", JSON.stringify(info));
  return { group, get info() { return info; }, setVisible: (on) => { group.visible = on; },
           setTier: (t) => { tier = Math.min(3, Math.max(0, t)); applyTier(); },
           dispose: () => { clearInterval(timer); stopped = true; } };
}


// Lite preset (integrated graphics): light trees only, drawn within 900 m of
// the camera, no procedural tree generation at start-up and no tree shadows.
// Same measured crowns: position, height, crown size and colour.
function addLightTrees({ root, groundMesh, TSL, camera, C, n, hills }) {
  const FAR_R = 900;
  const geo = farTreeGeometry(hills);
  const unitR = 0.52;
  const MF = new Float32Array(n * 16), COLF = new Float32Array(n * 3);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), pos = new THREE.Vector3(), scl = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < n; i++) {
    const hash = ((Math.floor(C.x[i] * 7.31) * 73856093) ^ (Math.floor(C.z[i] * 5.17) * 19349663)) >>> 0;
    const fy = C.h[i], fxz = Math.min(Math.max(Math.max(C.ra[i], C.r[i]) / unitR, fy * 0.35), fy * 1.4);
    pos.set(C.x[i], C.y[i] - 0.25, C.z[i]);
    q.setFromAxisAngle(up, (C.rot[i] || 0) + (hash % 628) / 100);
    scl.set(fxz, fy, fxz);
    m4.compose(pos, q, scl).toArray(MF, i * 16);
    const rgb = [C.R[i], C.G[i], C.B[i]].map((c) => c / 255);
    const grey = (rgb[0] + rgb[1] + rgb[2]) / 3;
    for (let k = 0; k < 3; k++) COLF[i * 3 + k] = toLin(Math.min(1, Math.max(0, grey + (rgb[k] - grey) * 1.35))) * 1.2;
  }
  const mat = new THREE.MeshStandardNodeMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.0 });
  const mesh = new THREE.InstancedMesh(geo, mat, n);
  mesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(n * 3), 3);
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.castShadow = false; mesh.receiveShadow = true; mesh.frustumCulled = false;
  const group = new THREE.Group(); group.name = "trees"; group.add(mesh); root.add(group);
  let lowered = 0;
  try { lowered = groundMesh ? flattenMounds(groundMesh, C) : 0; } catch (e) { console.warn("tree mound flatten skipped:", e); }
  const cam = new THREE.Vector3();
  function assign() {
    if (!camera) return;
    cam.copy(camera.position); root.worldToLocal(cam);
    const r2 = FAR_R * FAR_R, A = mesh.instanceMatrix.array, CC = mesh.instanceColor.array;
    let k = 0;
    for (let i = 0; i < n; i++) {
      const dx = C.x[i] - cam.x, dz = C.z[i] - cam.z;
      if (dx * dx + dz * dz > r2) continue;
      A.set(MF.subarray(i * 16, i * 16 + 16), k * 16); CC.set(COLF.subarray(i * 3, i * 3 + 3), k * 3); k++;
    }
    mesh.count = k; mesh.instanceMatrix.needsUpdate = true; mesh.instanceColor.needsUpdate = true;
  }
  assign();
  const timer = setInterval(assign, 700);
  const info = { crowns: n, mode: "lite", hills, lowered_vertices: lowered, far_tris_each: Math.round(geo.attributes.position.count / 3), radius_m: FAR_R };
  console.info("trees:", JSON.stringify(info));
  return { group, info, setVisible: (on) => { group.visible = on; }, setTier: () => {}, dispose: () => clearInterval(timer) };
}
