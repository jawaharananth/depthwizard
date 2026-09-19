"""Regression test for a real bug found and fixed this session (2026-09-19):
dfc2019_height.split_tiles() was putting 14 of rescore_baseline.py's 18
independent benchmark tiles into its own train split (4 more into val),
meaning any --use-dfc2019 training run trained directly on the tiles this
project's headline RMSE/MAE numbers are scored against.

These tests exist so that regression can never happen silently again --
if someone edits split_tiles() and reintroduces the leak, this fails loudly
instead of quietly inflating the next reported accuracy number.

Requires the local dfc2019_data/ directory (skipped, not failed, if absent --
this is data-dependent, not something to fake with a mock).
"""
import os

import pytest

import dfc2019_height as dfc

pytestmark = pytest.mark.skipif(
    not os.path.isdir(dfc.TRUTH_DIR),
    reason="dfc2019_data/truth not present locally")


def test_benchmark_tiles_are_stable_and_nonempty():
    bench = dfc.benchmark_tiles()
    assert len(bench) == 18
    assert len(set(bench)) == 18, "benchmark_tiles() must not contain duplicates"
    # Deterministic: same call twice must return the same set in the same
    # order, since rescore_baseline.py's own selection is order-dependent
    # (sorted stems, first 18) and this function exists specifically to
    # mirror that selection exactly.
    assert dfc.benchmark_tiles() == bench


def test_split_excludes_benchmark_tiles_by_default():
    bench = set(dfc.benchmark_tiles())
    train, val = dfc.split_tiles(dfc.available_tiles())
    assert not (set(train) & bench), \
        "benchmark tiles leaked into the train split"
    assert not (set(val) & bench), \
        "benchmark tiles leaked into the val split"


def test_split_train_val_disjoint():
    train, val = dfc.split_tiles(dfc.available_tiles())
    assert not (set(train) & set(val)), "a tile landed in both train and val"


def test_exclude_benchmark_false_actually_disables_the_guard():
    # The escape hatch exists for deliberate measurement (see its docstring)
    # and must still work -- if this silently ignored the flag, whoever ran
    # a deliberate ablation would draw the wrong conclusion from it.
    bench = set(dfc.benchmark_tiles())
    train, val = dfc.split_tiles(dfc.available_tiles(), exclude_benchmark=False)
    assert (set(train) | set(val)) & bench, \
        "exclude_benchmark=False should NOT remove benchmark tiles"
