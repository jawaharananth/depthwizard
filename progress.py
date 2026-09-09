"""Stage tracking, honest ETAs, and status text for long pipeline runs.

The point of this module is that the number on screen must be earned. Three
rules follow from that, and they are what separate this from a decorative
countdown:

FIRST, an estimate needs evidence. Until a stage has been timed at least once
on real hardware there is no ETA, and the UI is told so rather than being handed
a guess. A fabricated number is worse than no number, because the user believes
it.

SECOND, the estimate scales with the work. Duration here is dominated by pixel
count -- monocular depth inference on a 2480 px image is not the same job as on
a 1024 px one -- so timings are stored per megapixel for the stages that are
compute-bound, and as flat costs for the stages that are not. A single averaged
constant, which is the obvious way to build this, would be wrong by a large
factor the moment someone uploads an image of a different size.

THIRD, the countdown must not lie at the end. When elapsed time passes the
prediction, the number does not sit at 0:00 pretending; the stage switches to an
overrun state and says so. Hitting zero while still working is precisely the
failure that makes a progress UI feel fake.

Timings accumulate in a JSON file as runs happen, so the estimates get better
with use instead of staying frozen at whatever was hard-coded.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "stage_timings.json")

# How far past the prediction a stage must run before the UI stops counting down
# and admits it is over time. A little slack absorbs normal variation without
# flipping the message on every run.
OVERRUN_TOLERANCE = 1.15

# ...and it must also be late by this many seconds in absolute terms. Without
# the floor, a stage predicted at 4 s trips the overrun state by running 5 s,
# and the display flips from a countdown to "taking longer than usual" and back
# within one second. That flicker reads as a bug, and it is meaningless anyway:
# on this pipeline depth inference is ~94% of the build, so a two-second
# overshoot elsewhere is invisible in the total.
OVERRUN_MIN_SECONDS = 10.0

# A stage is called notably fast or slow only if it falls outside this many
# multiples of the observed spread. Using a fixed "few seconds" instead would
# fire constantly on long stages and never on short ones.
DEVIATION_SIGMAS = 2.0

# ...and the deviation must also be this many seconds in absolute terms. The
# stages either side of depth inference are sub-second to a few seconds, and
# with a handful of samples their spread is tiny, so a 0.4 s wobble on a 0.9 s
# stage cleared the sigma test and announced "reading the image took longer than
# usual". A remark nobody can perceive is noise pretending to be transparency.
REMARK_MIN_SECONDS = 8.0

# How often a running stage re-emits its state.
#
# Without this the pipeline only speaks at stage boundaries, and depth inference
# runs for minutes between two of them. The client, ticking locally, drains to
# 0:00 and sits there while work continues -- the precise failure this module
# was written to prevent. It also means the overrun state is never re-evaluated
# mid-stage and the message pool never rotates, so a four-minute wait shows one
# frozen sentence. The heartbeat carries real recomputed state, not a ping.
HEARTBEAT_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------
#
# These mirror the [n/5] stages actually printed by build_city_image.py. They
# are not invented labels: if the pipeline's stage boundaries move, these move
# with them, or the ETA silently starts measuring the wrong spans.
#
# ``scales_with_px`` marks the stages whose cost tracks image area. Loading a
# file and fitting a scale factor do not; depth inference, segmentation,
# footprint extraction and mesh export all do.

STAGES: List[Dict] = [
    {
        "key": "load",
        "label": "Reading the image",
        "scales_with_px": False,
        "messages": [
            "Opening your image...",
            "Checking this is a top-down view...",
        ],
    },
    {
        "key": "height_field",
        "label": "Estimating height",
        "scales_with_px": True,
        "messages": [
            "Estimating how tall everything is...",
            "Reading depth from a single view -- the slow, careful part...",
            "Untangling perspective to recover elevation...",
            "Still working: this stage does most of the thinking...",
        ],
    },
    {
        "key": "scale",
        "label": "Fixing the scale",
        "scales_with_px": False,
        "messages": [
            "Working out how many metres one unit of depth is...",
            "Measuring shadows to anchor real-world height...",
        ],
    },
    {
        "key": "buildings",
        "label": "Finding buildings",
        "scales_with_px": True,
        "messages": [
            "Separating buildings from ground, roads and greenery...",
            "Tracing rooftop outlines...",
            "Straightening walls and squaring off corners...",
        ],
    },
    {
        "key": "export",
        "label": "Building the model",
        "scales_with_px": True,
        "messages": [
            "Standing the buildings up...",
            "Painting the model with your imagery...",
            "Packaging the 3D model...",
        ],
    },
]

STAGE_INDEX = {s["key"]: i for i, s in enumerate(STAGES)}
MESSAGE_PERIOD_S = 18.0     # how often the line changes inside a long stage


# ---------------------------------------------------------------------------
# Calibration store
# ---------------------------------------------------------------------------

def _load_calibration(path: str = CALIB_PATH) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # A corrupt timings file must not take the pipeline down with it. No
        # calibration simply means no ETA, which the UI already handles.
        return {}


def _save_calibration(data: Dict, path: str = CALIB_PATH) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def record_timing(stage_key: str, seconds: float, megapixels: float,
                  path: str = CALIB_PATH, keep: int = 40) -> None:
    """Append one observed stage duration to the calibration store."""
    data = _load_calibration(path)
    samples = data.setdefault(stage_key, [])
    samples.append({"s": round(float(seconds), 3),
                    "mp": round(float(megapixels), 4)})
    # Keep the window bounded and recent: hardware and code both change, and a
    # timing from fifty runs ago is not evidence about this one.
    del samples[:-keep]
    _save_calibration(data, path)


def predict_stage(stage_key: str, megapixels: float,
                  calib: Optional[Dict] = None) -> Optional[float]:
    """Predicted duration in seconds, or None when there is no evidence yet."""
    calib = _load_calibration() if calib is None else calib
    samples = calib.get(stage_key) or []
    if not samples:
        return None

    spec = STAGES[STAGE_INDEX[stage_key]] if stage_key in STAGE_INDEX else {}
    if spec.get("scales_with_px") and megapixels > 0:
        # Rescale each past run to the size of this one before averaging, so a
        # single 6 MP sample still predicts a 1 MP job correctly.
        rates = [s["s"] / s["mp"] for s in samples if s.get("mp", 0) > 0]
        if rates:
            return statistics.median(rates) * megapixels
    return statistics.median([s["s"] for s in samples])


def stage_spread(stage_key: str, calib: Optional[Dict] = None) -> Optional[float]:
    """Typical variation of a stage's duration, for the fast/slow remark."""
    calib = _load_calibration() if calib is None else calib
    samples = calib.get(stage_key) or []
    if len(samples) < 3:
        return None
    vals = [s["s"] for s in samples]
    med = statistics.median(vals)
    # Median absolute deviation: robust to the one pathological run that would
    # otherwise inflate a standard deviation and mute the remark entirely.
    mad = statistics.median([abs(v - med) for v in vals])
    return mad * 1.4826 or None


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class StageTracker:
    """Tracks stage progress for one run and emits events to a sink.

    The sink is any callable taking a dict. In the web app it pushes onto a
    queue that an SSE endpoint drains; on the command line it can print. The
    pipeline itself does not know or care which.
    """

    def __init__(self, job_id: str, megapixels: float,
                 sink: Optional[Callable[[Dict], None]] = None,
                 record: bool = True, calib_path: str = CALIB_PATH):
        # Tests must be able to point the calibration store somewhere else. A
        # synthetic run that writes into the real store is not a harmless test:
        # it becomes evidence, and every later prediction is computed from it.
        self.calib_path = calib_path
        self.job_id = job_id
        self._megapixels = float(megapixels)
        self.sink = sink
        self.record = record
        self.calib = _load_calibration(self.calib_path)
        self.started = time.time()
        self.current: Optional[str] = None
        self.stage_started = 0.0
        self.done: Dict[str, float] = {}
        self.note: Optional[str] = None
        self._lock = threading.Lock()

        self.predictions = {
            s["key"]: predict_stage(s["key"], self.megapixels, self.calib)
            for s in STAGES
        }
        # An ETA is only offered when EVERY stage has evidence. A total built
        # from three known stages and two unknowns is not an estimate of the
        # whole job, and presenting it as one is the lie this module exists to
        # avoid.
        self.can_estimate = all(v is not None for v in self.predictions.values())

        # Daemon so an abandoned job cannot keep the process alive.
        self._stop_hb = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    def _heartbeat(self) -> None:
        while not self._stop_hb.wait(HEARTBEAT_SECONDS):
            if self.current is not None:
                try:
                    self.emit("progress")
                except Exception:
                    # A dead subscriber must never take down the build thread.
                    pass

    @property
    def megapixels(self) -> float:
        return self._megapixels

    @megapixels.setter
    def megapixels(self, value: float) -> None:
        """Resize the job and re-derive every prediction from it.

        The caller supplies an image size up front, but the pipeline may resize
        the image before any real work starts (a 40 MP upload is capped to
        max_px). Predictions made against the original size would then be too
        large for the whole run. Assigning here recomputes them, so the estimate
        reflects the pixels actually processed.
        """
        self._megapixels = float(value)
        self.predictions = {
            spec["key"]: predict_stage(spec["key"], self._megapixels, self.calib)
            for spec in STAGES
        }
        self.can_estimate = all(v is not None for v in self.predictions.values())

    # -- estimation ---------------------------------------------------------

    def eta(self) -> Optional[float]:
        """Seconds remaining, or None when there is not enough evidence."""
        if not self.can_estimate or self.current is None:
            return None
        idx = STAGE_INDEX[self.current]
        elapsed = time.time() - self.stage_started
        remaining_current = max(self.predictions[self.current] - elapsed, 0.0)
        remaining_future = sum(self.predictions[s["key"]]
                               for s in STAGES[idx + 1:])
        return remaining_current + remaining_future

    def overrunning(self) -> bool:
        if not self.can_estimate or self.current is None:
            return False
        pred = self.predictions[self.current]
        elapsed = time.time() - self.stage_started
        return (elapsed > pred * OVERRUN_TOLERANCE
                and elapsed - pred > OVERRUN_MIN_SECONDS)

    def message(self) -> str:
        if self.current is None:
            return "Getting ready..."
        spec = STAGES[STAGE_INDEX[self.current]]
        pool = spec["messages"]
        elapsed = time.time() - self.stage_started
        return pool[int(elapsed // MESSAGE_PERIOD_S) % len(pool)]

    # -- event emission -----------------------------------------------------

    def snapshot(self, phase: str = "progress") -> Dict:
        idx = STAGE_INDEX[self.current] if self.current else -1
        eta = self.eta()
        over = self.overrunning()
        # Past the estimate but not yet confidently late. The overrun floor
        # deliberately waits before declaring a stage slow, which left a window
        # where the countdown had reached zero and was still reported as known
        # -- so the UI printed 0:00 while work continued. eta_known means "there
        # is a credible POSITIVE estimate"; at zero there is not one, whatever
        # the overrun state says.
        past = (eta is not None and eta <= 0.5 and not over)
        ev = {
            "job": self.job_id,
            "phase": phase,
            "stage": self.current,
            "past_estimate": past,
            "stage_label": (STAGES[idx]["label"] if idx >= 0 else None),
            "stage_index": idx,
            "stage_count": len(STAGES),
            "elapsed_total": round(time.time() - self.started, 1),
            "message": self.message(),
            # None means "no estimate available", which the UI must render as
            # such. It is deliberately not 0 or a placeholder number.
            "eta": (None if eta is None else round(eta)),
            "eta_known": self.can_estimate and not over and not past,
            "overrunning": over,
        }
        if self.note:
            ev["note"] = self.note
            self.note = None
        return ev

    def emit(self, phase: str = "progress") -> Dict:
        ev = self.snapshot(phase)
        if self.sink:
            self.sink(ev)
        return ev

    # -- stage lifecycle ----------------------------------------------------

    def enter(self, key: str) -> None:
        """Open a stage without a `with` block.

        The pipeline functions are long linear bodies; wrapping five spans of
        them in context managers would mean reindenting most of the file, and a
        reindent is exactly the kind of edit that quietly moves a line into or
        out of an `if`. enter/leave instrument the same spans with single-line
        insertions instead.
        """
        if key not in STAGE_INDEX:
            raise KeyError(f"unknown stage {key!r}; expected one of "
                           f"{list(STAGE_INDEX)}")
        with self._lock:
            self.current = key
            self.stage_started = time.time()
        self.emit("stage_start")

    def leave(self) -> None:
        """Close the open stage, recording its duration."""
        key = self.current
        if key is None:
            return
        took = time.time() - self.stage_started
        self.done[key] = took
        self._remark(key, took)
        if self.record:
            record_timing(key, took, self.megapixels, self.calib_path)
            self.calib = _load_calibration(self.calib_path)
        self.emit("stage_end")

    @contextmanager
    def stage(self, key: str):
        if key not in STAGE_INDEX:
            raise KeyError(f"unknown stage {key!r}; expected one of "
                           f"{list(STAGE_INDEX)}")
        with self._lock:
            self.current = key
            self.stage_started = time.time()
        self.emit("stage_start")
        try:
            yield self
        finally:
            took = time.time() - self.stage_started
            self.done[key] = took
            self._remark(key, took)
            if self.record:
                record_timing(key, took, self.megapixels)
                # Fold the new measurement into this run's own predictions so
                # later stages of THIS job benefit immediately.
                self.calib = _load_calibration()
            self.emit("stage_end")

    def _remark(self, key: str, took: float) -> None:
        """Explain a visible jump in the countdown instead of letting it lurch."""
        pred = self.predictions.get(key)
        if pred is None:
            return
        spread = stage_spread(key, self.calib)
        if spread is None:
            return
        delta = took - pred
        if abs(delta) < DEVIATION_SIGMAS * spread:
            return
        if abs(delta) < REMARK_MIN_SECONDS:
            return
        label = STAGES[STAGE_INDEX[key]]["label"].lower()
        self.note = (f"{label} finished faster than usual -- skipping ahead"
                     if delta < 0 else
                     f"{label} took longer than usual -- hang tight")

    def finish(self, ok: bool = True, detail: str = "") -> Dict:
        self._stop_hb.set()
        with self._lock:
            self.current = None
        ev = {
            "job": self.job_id,
            "phase": "done" if ok else "error",
            "elapsed_total": round(time.time() - self.started, 1),
            "message": ("Your 3D model is ready." if ok
                        else (detail or "The build failed.")),
            "eta": 0 if ok else None,
            "eta_known": ok,
            "overrunning": False,
            "stage_timings": {k: round(v, 1) for k, v in self.done.items()},
        }
        if self.sink:
            self.sink(ev)
        return ev


class NullTracker:
    """Accepts the instrumentation calls and does nothing.

    The pipeline is usable from the command line with no web app attached, and
    that path must not carry a `tracker is not None` test at every stage
    boundary -- each one would be another place for the instrumentation to be
    silently skipped.
    """

    megapixels = 0.0

    def enter(self, key: str) -> None:
        pass

    def leave(self) -> None:
        pass

    def emit(self, phase: str = "progress"):
        return {}

    def finish(self, ok: bool = True, detail: str = ""):
        return {}


def calibration_summary(path: str = CALIB_PATH) -> str:
    """Human-readable state of the timing evidence."""
    calib = _load_calibration(path)
    if not calib:
        return "no timing samples recorded yet -- ETAs unavailable"
    lines = []
    for s in STAGES:
        k = s["key"]
        samples = calib.get(k) or []
        if not samples:
            lines.append(f"  {s['label']:<22} no samples")
            continue
        med = statistics.median([x["s"] for x in samples])
        unit = ("s/MP" if s["scales_with_px"] else "s")
        if s["scales_with_px"]:
            rates = [x["s"] / x["mp"] for x in samples if x.get("mp", 0) > 0]
            med = statistics.median(rates) if rates else med
        lines.append(f"  {s['label']:<22} {med:7.2f} {unit}   n={len(samples)}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("stage timing calibration:")
    print(calibration_summary())
