"""The iGPU tile queue must not freeze other threads.

The first version handed results back through OpenVINO callbacks; inside the
multi-threaded upload server that deadlocked the process (every HTTP request
timed out). Here the heavy call runs in a worker thread, as a build does in
the server, while this thread keeps a heartbeat: it must never stall and the
work must finish. Skipped where there is no iGPU.
"""
import threading
import time

import numpy as np
import pytest
from PIL import Image

import depth_model as dm


@pytest.fixture(scope="module")
def bb():
    b = dm.DepthBackbone(metric=True)
    if getattr(b, "_ovg", None) is None:
        pytest.skip("no iGPU backend on this machine")
    return b


def test_worker_thread_does_not_freeze_server(bb):
    rng = np.random.RandomState(1)
    img = Image.fromarray((rng.rand(1400, 1400, 3) * 255).astype(np.uint8))
    done, err = [], []

    def work():
        try:
            for _ in range(3):
                done.append(bb.predict_tiled(img, raw_metric=True).shape)
        except Exception as e:           # pragma: no cover
            err.append(e)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    last, worst = time.time(), 0.0
    deadline = time.time() + 240
    while t.is_alive() and time.time() < deadline:
        time.sleep(0.05)
        now = time.time()
        worst = max(worst, now - last)
        last = now
    assert not t.is_alive(), "worker never finished (deadlock)"
    assert not err, err
    assert len(done) == 3
    assert worst < 5.0, f"main thread stalled {worst:.1f} s"
