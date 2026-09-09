"""Upload dashboard: drop a nadir satellite image, get a 3D model.

Standard library only, deliberately. This has to run on a laptop that may have
no network access when it matters most, and adding Flask or FastAPI would mean
a pip install standing between the demo and a working machine. ThreadingHTTPServer
serves server-sent events perfectly well.

Two design decisions worth stating, because both are visible to the user:

JOBS RUN ONE AT A TIME. The pipeline writes to fixed output directories
(final_out and viewer/output), so two concurrent builds would overwrite each
other's meshes and the second viewer to load would show the first job's city.
Rather than pretend to be concurrent and corrupt results, a second upload waits
and is told its position in the queue.

UPLOADS ARE POSTED AS A RAW BODY, not multipart. Python 3.13 removed the cgi
module, and a hand-rolled multipart parser is a fiddly, security-sensitive thing
to get right for no benefit here. The filename travels as a query parameter and
the bytes are the request body, which the browser's fetch() sends natively.
"""

from __future__ import annotations

import json
import mimetypes
import os
import queue
import shutil
import threading
import time
import traceback
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import progress as prog

ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(ROOT, "uploads")
VIEWER_DIR = os.path.join(ROOT, "viewer")

# Uploads are user-supplied files written to disk, so both the extension and the
# size are checked before anything touches them. The pipeline itself refuses
# non-nadir imagery, but that check happens after a full decode.
ALLOWED_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_pipeline_lock = threading.Lock()      # serialises builds; see module docstring


# ---------------------------------------------------------------------------
# Job plumbing
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str, path: str, name: str):
        self.id = job_id
        self.path = path
        self.name = name
        self.subscribers: list[queue.Queue] = []
        self.history: list[dict] = []
        self.lock = threading.Lock()
        self.finished = False

    def publish(self, ev: dict) -> None:
        with self.lock:
            self.history.append(ev)
            if ev.get("phase") in ("done", "error"):
                self.finished = True
            subs = list(self.subscribers)
        for q in subs:
            q.put(ev)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            # Replay what already happened. A browser that connects late -- or
            # reconnects after a dropped connection -- must not see an empty
            # panel for a job that is already three stages in.
            for ev in self.history:
                q.put(ev)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


def run_job(job: Job) -> None:
    import build_city_image as bci
    from PIL import Image

    try:
        with Image.open(job.path) as im:
            w, h = im.size
        mp = (w * h) / 1e6
    except Exception:
        mp = 4.0

    tracker = prog.StageTracker(job.id, megapixels=mp, sink=job.publish)

    waiting = _pipeline_lock.locked()
    if waiting:
        # Say so rather than showing a stalled 0% for an unexplained minute.
        job.publish({"job": job.id, "phase": "queued", "stage": None,
                     "message": "Another model is being built -- yours is next.",
                     "eta": None, "eta_known": False, "overrunning": False})

    with _pipeline_lock:
        try:
            bci.build(job.path, job.name, max_px=2048, tracker=tracker,
                      stage=True)
            tracker.finish(ok=True)
        except SystemExit as e:
            # The pipeline refuses obliques and undersized images on purpose,
            # and that refusal is a real answer, not a crash. Pass its reasoning
            # through verbatim.
            tracker.finish(ok=False, detail=str(e))
        except Exception as e:
            traceback.print_exc()
            tracker.finish(ok=False, detail=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass       # the pipeline's own output is the interesting log

    def handle_one_request(self):
        """Swallow the traceback when a browser drops an SSE stream.

        Closing a tab mid-build resets the connection, and the default handler
        prints a full stack trace for it. During a live demo that looks like the
        pipeline crashed when nothing went wrong at all.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True

    # -- helpers ------------------------------------------------------------

    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _file(self, path: str) -> None:
        if not os.path.isfile(path):
            self._json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self._send(200, data, ctype)

    # -- routes -------------------------------------------------------------

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        route = u.path

        if route in ("/", "/index.html"):
            self._file(os.path.join(ROOT, "dashboard.html"))
            return

        if route == "/api/gallery":
            import prebuild
            self._json(200, {"scenes": prebuild.list_scenes()})
            return

        if route.startswith("/scenes/"):
            # Pre-built scenes, served straight from disk. Each lives in its own
            # directory so a gallery can offer several finished cities at once --
            # scenes.py restores by copying over viewer/output, which allows only
            # one at a time.
            rel = route[len("/scenes/"):]
            root = os.path.join(ROOT, "scenes")
            safe = os.path.normpath(os.path.join(root, rel))
            if not safe.startswith(os.path.normpath(root)):
                self._json(403, {"error": "forbidden"})
                return
            self._file(safe)
            return

        if route == "/api/calibration":
            self._json(200, {"summary": prog.calibration_summary()})
            return

        if route.startswith("/api/events/"):
            self._sse(route.rsplit("/", 1)[-1])
            return

        if route.startswith("/viewer/"):
            # Serve the existing viewer and its staged output unchanged, so the
            # dashboard hands off to exactly the tool that is already tested.
            rel = route[len("/viewer/"):]
            safe = os.path.normpath(os.path.join(VIEWER_DIR, rel))
            if not safe.startswith(os.path.normpath(VIEWER_DIR)):
                self._json(403, {"error": "forbidden"})
                return
            self._file(safe)
            return

        self._json(404, {"error": "no such route"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/api/upload":
            self._json(404, {"error": "no such route"})
            return

        qs = urllib.parse.parse_qs(u.query)
        raw_name = (qs.get("name") or ["upload"])[0]
        ext = os.path.splitext(raw_name)[1].lower()
        if ext not in ALLOWED_EXT:
            self._json(400, {"error":
                             f"{ext or 'that file type'} is not an image this "
                             "pipeline can read. Use TIFF, PNG or JPEG."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            self._json(400, {"error": "empty upload"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(413, {"error":
                             f"file is {length/1e6:.0f} MB; the limit is "
                             f"{MAX_UPLOAD_BYTES/1e6:.0f} MB"})
            return

        job_id = uuid.uuid4().hex[:12]
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        # The stored name is derived from the job id, never from the uploaded
        # filename: a name arriving over the wire must not decide where bytes
        # land on disk.
        dest = os.path.join(UPLOAD_DIR, job_id + ext)

        remaining = length
        with open(dest, "wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)

        job = Job(job_id, dest, "upload_" + job_id)
        with _jobs_lock:
            _jobs[job_id] = {"job": job}
        threading.Thread(target=run_job, args=(job,), daemon=True).start()
        self._json(200, {"job": job_id})

    # -- SSE ----------------------------------------------------------------

    def _sse(self, job_id: str) -> None:
        with _jobs_lock:
            rec = _jobs.get(job_id)
        if not rec:
            self._json(404, {"error": "unknown job"})
            return
        job: Job = rec["job"]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q = job.subscribe()
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    # A comment frame keeps proxies and the browser from closing
                    # an idle stream during the long depth stage, which can run
                    # for minutes without a stage boundary.
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                payload = json.dumps(ev).encode("utf-8")
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()
                if ev.get("phase") in ("done", "error"):
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            job.unsubscribe(q)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="DepthWizard upload dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    print("DepthWizard dashboard")
    print(f"  http://{a.host}:{a.port}")
    print()
    print("stage timing calibration:")
    print(prog.calibration_summary())
    print()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
