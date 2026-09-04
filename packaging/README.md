# Standalone packaging (Tauri)

**Status: prepared but NOT BUILT.** `cargo` is not installed on the development
machine, so nothing here has been compiled or run. Everything below is written
from the Tauri documentation and the actual shape of this repo — treat it as a
starting point to verify, not a tested procedure.

That distinction matters: this project has removed several claims that turned out
to be untested, and a build recipe nobody has executed is exactly that kind of
claim.

## Why Tauri rather than Electron

The viewer is already a static page plus a WebGPU canvas. Tauri uses the OS
webview instead of bundling Chromium, so the installer is tens of MB rather than
hundreds, and the renderer needs no changes at all — it is the same
`viewer/index.html` served locally.

The one real risk is **WebGPU support in the host webview**: Tauri renders
through WebView2 on Windows and WebKitGTK on Linux, and WebGPU availability
varies by version. The viewer already falls back to WebGL2 automatically
(`WebGPURenderer` handles this), so the failure mode is reduced performance
rather than a blank window — but it must be tested on the target machine before
being relied on for a live demo.

## Structure

```
packaging/
  tauri.conf.json     window, bundle and security config
  Cargo.toml          Rust manifest (unverified)
  src/main.rs         entry point (unverified)
dist/                 static build: viewer/ + a built scene in output/
```

## Steps

```bash
# 1. Install the Rust toolchain (this is the missing prerequisite)
#    https://rustup.rs

# 2. Install the Tauri CLI
cargo install tauri-cli

# 3. Stage the frontend. Tauri serves a static directory, so the viewer and a
#    built scene are copied in together -- a scene must exist first, or the
#    packaged app opens on "Loading terrain..." forever.
python build_city.py JAX_165 --extent 640 --px 2560 --views 6 --dem
mkdir -p dist && cp -r viewer dist/

# 4. Build
cargo tauri build
```

## What ships, and what does not

The packaged app contains the **viewer and one or more pre-built scenes**. It
does NOT contain the reconstruction pipeline: that needs Python, PyTorch, a
1.3 GB depth model and ~15 minutes of CPU per tile, which is not something to
bundle into a desktop installer.

Two honest options for a standalone that can also *build* scenes:

1. **Viewer-only** (recommended for a demo): ship pre-built scenes. Opens
   instantly, needs no Python, cannot process new imagery.
2. **Sidecar backend**: bundle the FastAPI upload server as a Tauri sidecar
   binary via PyInstaller. Adds several hundred MB and inherits every CPU-time
   constraint documented in the report — an upload still takes 8–22 minutes.

Option 1 is what a live demo actually needs. Option 2 is what "standalone
application" implies in the problem statement, and its cost should be stated
rather than discovered during judging.
