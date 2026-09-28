"""Build ONE self-contained HTML file that shows a scene with no server.

    python tools/make_offline.py scenes/<name> dist/DepthWizard_<name>.html [--max-mb 180]

Double-click the result: it opens from disk, offline. Chrome refuses both ES
modules and fetch() over file://, so everything is inside the page:

  * three.js: every module the viewer imports, statically or dynamically, is
    found by walking the import graph from viewer/index.html, its relative
    imports rewritten to bare names, and each mapped to a data: URL through
    the page's import map (one URL per module, so three.js is instantiated
    once);
  * the scene: each file the viewer can load is embedded as base64 and turned
    into a blob: URL the first time the viewer asks for it -- through fetch()
    and through three.js's loading manager (images and GLB);
  * the server: /api/ calls get an immediate "offline copy" answer, so upload,
    validation and the other server features say they are unavailable
    instead of hanging.

Large optional layers are dropped first when the page would exceed --max-mb
(the surrounding context city, then far-context imagery); the list of what
was included and skipped is printed and written beside the HTML.
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER = os.path.join(ROOT, "viewer")
VENDOR = os.path.join(VIEWER, "vendor", "three@0.180.0")

SPEC = re.compile(r"""(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])([^"']+)\2""")
NEVER = {".tif", ".tiff", ".log"}
DROP_ORDER = ["context_city.glb", "terrain_context_far.jpg", "terrain_context.jpg",
              "facade_windows.json", "terrain_texture.png", "tree_crowns.json", "buildings.geojson"]


def canon(path):
    """Bare import name for a vendored file."""
    if not os.path.abspath(path).startswith(os.path.abspath(VENDOR)):
        return "dwapp/" + os.path.relpath(path, VIEWER).replace("\\", "/")   # viewer's own modules
    rel = os.path.relpath(path, VENDOR).replace("\\", "/")
    if rel == "build/three.webgpu.js":
        return "three"
    if rel == "build/three.tsl.js":
        return "three/tsl"
    return "dwmod/" + rel


def resolve(spec, importer_dir, imap):
    if spec in ("three", "three/webgpu"):
        return os.path.join(VENDOR, "build", "three.webgpu.js")
    if spec == "three/tsl":
        return os.path.join(VENDOR, "build", "three.tsl.js")
    if spec.startswith("three/addons/"):
        return os.path.join(VENDOR, "examples", "jsm", spec[len("three/addons/"):])
    if spec.startswith(("./", "../")):
        return os.path.normpath(os.path.join(importer_dir or VIEWER, spec))
    return None


def build_modules(entry_src):
    """{canonical name: rewritten source} for every module reachable from entry."""
    mods, todo = {}, []

    def rewrite(src, importer_dir):
        def sub(m):
            p = resolve(m.group(3), importer_dir, None)
            if p is None:
                return m.group(0)
            if os.path.exists(p):
                todo.append(p)
            return f"{m.group(1)}{m.group(2)}{canon(p)}{m.group(2)}"
        return SPEC.sub(sub, src)

    entry = rewrite(entry_src, None)
    seen = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        with open(p, encoding="utf-8") as f:
            mods[canon(p)] = rewrite(f.read(), os.path.dirname(p))
    return entry, mods


SHIM = r"""<script>
(function () {
  var files = {}, blobs = {};
  document.querySelectorAll('script[data-dwpath]').forEach(function (s) {
    files[s.getAttribute('data-dwpath')] = s;
  });
  function key(u) {
    try {
      var p = decodeURIComponent(new URL(u, location.href).pathname);
      if (p.slice(-19) === '/viewer/places.json') return '/viewer/places.json';
      var i = p.indexOf('/scenes/');
      return i >= 0 ? p.slice(i) : null;
    } catch (e) { return null; }
  }
  function blobFor(k) {
    if (!(k in files)) return null;
    if (!blobs[k]) {
      var s = files[k], bin = atob(s.textContent), u8 = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
      blobs[k] = URL.createObjectURL(new Blob([u8], { type: s.getAttribute('data-type') }));
      s.textContent = '';
    }
    return blobs[k];
  }
  window.__dwMapUrl = function (u) {
    if (typeof u !== 'string' || u.indexOf('blob:') === 0 || u.indexOf('data:') === 0) return u;
    var k = key(u), b = k && blobFor(k);
    return b || u;
  };
  var realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    var u = typeof input === 'string' ? input : (input && input.url) || '';
    if (u.indexOf('/api/') >= 0) {
      return Promise.resolve(new Response(JSON.stringify({ error: 'This is the offline copy: server features (upload, validation) need the DepthWizard server.' }),
        { status: 503, headers: { 'Content-Type': 'application/json' } }));
    }
    var m = window.__dwMapUrl(u);
    if (m !== u) return realFetch(m, init);
    if (location.protocol === 'file:' && u.indexOf('/scenes/') >= 0) {
      return Promise.resolve(new Response('', { status: 404 }));
    }
    return realFetch(input, init);
  };
})();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("out")
    ap.add_argument("--max-mb", type=float, default=180.0)
    a = ap.parse_args()
    scene_dir = os.path.abspath(a.scene_dir)
    name = os.path.basename(scene_dir.rstrip("\\/"))

    html = open(os.path.join(VIEWER, "index.html"), encoding="utf-8").read()
    m = re.search(r'<script type="module">(.*?)</script>', html, re.S)
    entry, mods = build_modules(m.group(1))
    # Offline defaults: this scene, and the loading manager routed through the shim.
    entry = entry.replace('const _scene = _params.get("scene");',
                          'const _scene = _params.get("scene") || window.__DW_SCENE;', 1)
    entry = re.sub(r'(import \{ GLTFLoader \} from "[^"]+";)',
                   r'\1\nTHREE.DefaultLoadingManager.setURLModifier((u) => window.__dwMapUrl ? window.__dwMapUrl(u) : u);',
                   entry, count=1)
    imap = {"imports": {k: "data:text/javascript;base64," + base64.b64encode(v.encode("utf-8")).decode()
                        for k, v in mods.items()}}

    files = sorted(f for f in os.listdir(scene_dir)
                   if os.path.isfile(os.path.join(scene_dir, f)) and os.path.splitext(f)[1].lower() not in NEVER)
    size = {f: os.path.getsize(os.path.join(scene_dir, f)) for f in files}
    js_mb = sum(len(v) for v in imap["imports"].values()) / 1e6
    budget = a.max_mb * 1e6 / 1.34 - js_mb * 1e6
    skipped = []
    for f in DROP_ORDER:
        if sum(size[x] for x in files) <= budget:
            break
        if f in files:
            files.remove(f); skipped.append(f)
    while files and sum(size[x] for x in files) > budget:
        big = max((x for x in files if x not in ("scene.json", "terrain.glb")), key=lambda x: size[x], default=None)
        if big is None:
            break
        files.remove(big); skipped.append(big)

    blobs = []
    for f in files:
        with open(os.path.join(scene_dir, f), "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode()
        typ = mimetypes.guess_type(f)[0] or "application/octet-stream"
        if f.endswith(".glb"):
            typ = "model/gltf-binary"
        blobs.append(f'<script type="application/octet-stream" data-dwpath="/scenes/{name}/{f}" '
                     f'data-type="{typ}">{b64}</script>')

    # the place list: just this place, so the panel shows it rather than an error
    try:
        _pl = [x for x in json.load(open(os.path.join(VIEWER, "places.json"), encoding="utf-8"))
               if x.get("id") == name]
    except Exception:
        _pl = []
    blobs.append('<script type="application/octet-stream" data-dwpath="/viewer/places.json" '
                 'data-type="application/json">' + base64.b64encode(json.dumps(_pl).encode()).decode() + '</script>')

    html = re.sub(r'<script type="importmap">.*?</script>',
                  lambda _: '<script type="importmap">' + json.dumps(imap) + "</script>", html, count=1, flags=re.S)
    html = html.replace(m.group(0), '<script type="module">' + entry + "</script>", 1)
    head_inject = ("\n".join(blobs) + f"\n<script>window.__DW_SCENE = {json.dumps(name)};</script>\n" + SHIM)
    html = html.replace('<script type="importmap">', head_inject + '<script type="importmap">', 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    report = {"scene": name, "html_mb": round(os.path.getsize(a.out) / 1e6, 1),
              "modules": len(mods), "included": files, "skipped": skipped}
    json.dump(report, open(os.path.splitext(a.out)[0] + "_contents.json", "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
