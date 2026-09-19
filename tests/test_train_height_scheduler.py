"""Regression test for a real bug found this session (2026-09-19):
OneCycleLR(total_steps=4, pct_start=0.25) divides by zero on its very first
step, because the up-phase rounds to exactly 0 steps at that combination.
Hit for real by the combined GAMUS+DFC2019 --smoke --htc run (train_height.py),
whose exact step count landed on this boundary.

train_height.py's own fix: skip the scheduler entirely below 8 steps (see
its comment on `sched = None`). This test exercises that boundary directly
against the real torch.optim.lr_scheduler.OneCycleLR, not a mock, since the
whole bug was a property of that class's internals at small step counts.
"""
import torch


def _one_cycle_would_crash(total_steps: int) -> bool:
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    try:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=total_steps, pct_start=0.25)
        sched.step()
        return False
    except ZeroDivisionError:
        return True


def test_the_known_crash_reproduces_against_real_torch():
    # Documents WHY train_height.py's guard exists, against the real class --
    # if a future torch version fixes this upstream, this test simply starts
    # asserting False, which is a signal the guard could be relaxed, not a
    # silent bitrot.
    assert _one_cycle_would_crash(4), \
        "expected OneCycleLR(total_steps=4, pct_start=0.25) to raise " \
        "ZeroDivisionError on this torch version; if it no longer does, " \
        "train_height.py's `steps >= 8` guard may be over-cautious now, " \
        "not wrong -- re-check before loosening it"


def test_train_height_guard_threshold_is_actually_safe():
    # train_height.py skips OneCycleLR below 8 steps. Confirm 8 itself, and
    # a range above it, never hits the crash -- if the safe threshold were
    # actually higher than 8, this would catch it before a real run does.
    for steps in (8, 9, 10, 16, 32, 100):
        assert not _one_cycle_would_crash(steps), \
            f"total_steps={steps} crashed OneCycleLR; train_height.py's " \
            f"`steps >= 8` guard is not actually safe at this boundary"
