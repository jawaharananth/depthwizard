"""The 3D build runs in its own process, never in the web server.

The server answers every request -- uploads, progress polling, the viewer's
files -- while a build is running. Twice, a build running as a thread of the
server froze the whole process: a native thread held the interpreter lock, so
no request could be answered and the page just hung (py-spy: every Python
thread waiting, one native thread "active+gil"). The same builds, run on their
own, never froze. So the build now runs in a separate worker process that
loads the model once and takes jobs from a queue:

  * the server stays responsive whatever a build does;
  * progress events stream back to the server as before;
  * a watchdog: a healthy build emits a heartbeat every 5 s (progress.py), so
    silence for WATCHDOG_S, a dead worker, or a job over its time budget means
    the worker is stuck -- it is killed, restarted and the job retried once.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
import traceback

WATCHDOG_S = float(os.environ.get("DW_WATCHDOG_S", "120"))


class Stalled(RuntimeError):
    pass


LOG_MAX_LINES = 600
_ROOT = os.path.dirname(os.path.abspath(__file__))


class _LogTee:
    """Passes the build's own printed lines through to the console and, one
    complete line at a time, to the job's event stream (phase "log"), so the
    upload page can show what the pipeline is actually doing. Local paths are
    trimmed to names; tqdm's carriage-return progress bars are skipped."""

    def __init__(self, real, emit):
        self.real, self.emit, self.buf, self.sent = real, emit, "", 0

    def write(self, s):
        try:
            self.real.write(s)
        except Exception:
            pass
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self._line(line)
        return len(s)

    def _line(self, line):
        if "\r" in line:
            return
        line = line.rstrip()
        if not line.strip() or self.sent >= LOG_MAX_LINES:
            return
        line = line.replace(_ROOT + os.sep, "").replace(_ROOT, "")
        if len(line) > 220:
            line = line[:217] + "..."
        self.sent += 1
        self.emit(line)

    def flush(self):
        try:
            self.real.flush()
        except Exception:
            pass


def _worker_main(tasks, events):
    import sys
    import depth_model
    import progress as prog
    t0 = time.time()
    depth_model.DepthBackbone()                 # the same instance every build uses
    import build_city_image as bci
    events.put({"t": "ready", "seconds": round(time.time() - t0, 1), "pid": os.getpid()})
    while True:
        task = tasks.get()
        if task is None:
            return
        jid = task["job"]

        def sink(ev, _j=jid):
            if ev.get("phase") in ("done", "error"):
                return                           # the server announces the end, after its own steps
            events.put({"t": "ev", "job": _j, "ev": ev})

        tr = prog.StageTracker(jid, megapixels=task["mp"], sink=sink)
        res = {"t": "result", "job": jid}
        _flag = os.environ.get("DW_SIMULATE_STALL_FLAG")
        if _flag and os.path.exists(_flag):
            # test hook: freeze exactly like the real incident -- a native call
            # holding the interpreter lock, so even the heartbeat thread stops
            os.remove(_flag)
            tr.enter("load")
            import ctypes
            ctypes.PyDLL("kernel32").Sleep(3600 * 1000)
        real_out = sys.stdout
        sys.stdout = _LogTee(real_out, lambda line, _j=jid: events.put(
            {"t": "ev", "job": _j, "ev": {"job": _j, "phase": "log", "log": line, "t": round(time.time(), 2)}}))
        try:
            bci.build(task["path"], task["name"], tracker=tr, stage=True, **task.get("kw", {}))
            res["ok"] = True
        except SystemExit as e:                  # a refusal (oblique photo, too small): a real answer
            res.update(ok=False, refused=True, detail=str(e))
        except Exception as e:
            traceback.print_exc()
            res.update(ok=False, detail=f"{type(e).__name__}: {e}")
        finally:
            sys.stdout = real_out
        res["final"] = tr.finish(ok=res["ok"], detail=res.get("detail", ""))
        events.put(res)


class Worker:
    def __init__(self):
        self._ctx = mp.get_context("spawn")
        self.proc = None
        self.ready = False
        self.ready_seconds = None
        self.start()

    def start(self) -> None:
        self.tasks = self._ctx.Queue()
        self.events = self._ctx.Queue()
        self.ready = False
        self.proc = self._ctx.Process(target=_worker_main, args=(self.tasks, self.events),
                                      name="dw-build-worker", daemon=True)
        self.proc.start()

    def wait_ready(self, timeout: float = 900.0) -> float:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                m = self.events.get(timeout=1.0)
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"build worker exited during start-up (code {self.proc.exitcode})")
                continue
            if m.get("t") == "ready":
                self.ready = True
                self.ready_seconds = m.get("seconds")
                return self.ready_seconds
        raise RuntimeError("build worker did not become ready")

    def restart(self) -> float:
        self.stop(kill=True)
        self.start()
        return self.wait_ready()

    def stop(self, kill: bool = False) -> None:
        if self.proc is None:
            return
        if self.proc.is_alive():
            if kill:
                self.proc.kill()
            else:
                try:
                    self.tasks.put(None)
                except Exception:
                    pass
            self.proc.join(10)
        self.ready = False

    def run(self, task: dict, on_event, max_seconds: float) -> dict:
        """Run one job; returns the worker's result. Raises Stalled if the worker
        goes silent, dies, or runs past max_seconds."""
        self.tasks.put(task)
        t0 = last = time.time()
        while True:
            try:
                m = self.events.get(timeout=1.0)
            except queue.Empty:
                now = time.time()
                if not self.proc.is_alive():
                    raise Stalled(f"the build process stopped (code {self.proc.exitcode})")
                if now - last > WATCHDOG_S:
                    raise Stalled(f"no progress for {WATCHDOG_S:.0f} s")
                if now - t0 > max_seconds:
                    raise Stalled(f"over the {max_seconds / 60:.0f} min budget")
                continue
            last = time.time()
            if m.get("job") != task["job"]:
                continue                        # a stale message from a killed run
            if m["t"] == "ev":
                on_event(m["ev"])
            elif m["t"] == "result":
                return m
