"""A crashed shard process must leave nothing behind and its re-run must be the same search.

Found by killing a shard process of a concurrent SA run on purpose (R3 verification, 2026-11-25):
  1. its eval-pool workers and its (p, q) cache Manager were orphaned and never exited; they
     kept the multiprocessing resource tracker open, so the run's main process never exited;
  2. the re-run read the killed attempt's rows back as SA seed candidates, so it searched a
     different path (different rows, a different best δ).

Coverage:
  - start_parent_watchdog: a process exits when its parent dies; the eval-pool initializer,
    the cache Manager and the shard-process initializer install it.
  - load_seen_trajectories(max_bytes): only complete rows inside the prefix are read.
  - repair_trailing_fragment: a half-written last line is cut off (rows are kept).
  - End to end (real 2F1 log(2) shard): a run interrupted after a prefix of its rows (with a
    torn last line) and re-run with the seed snapshot ends byte-identical to an
    uninterrupted run.
"""

import multiprocessing as mp
import os
import signal
import time

import psutil
import pytest

from dreamer.configs import config
from dreamer.configs.system import sys_config


# ---------------------------------------------------------------------------
# parent watchdog (module-level targets: spawned processes import this module)
# ---------------------------------------------------------------------------

def _grandchild():
    from dreamer.utils.mp_context import start_parent_watchdog
    start_parent_watchdog(interval=0.2)
    time.sleep(120)


def _child(q):
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_grandchild)
    p.start()
    q.put(p.pid)
    time.sleep(120)


def test_watchdog_exits_when_parent_dies():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    parent = ctx.Process(target=_child, args=(q,))
    parent.start()
    grandchild = q.get(timeout=60)
    time.sleep(3.0)                                    # let the watchdog start
    # alive (not a zombie that crashed on start-up) — otherwise the test would be vacuous
    assert psutil.Process(grandchild).status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    os.kill(parent.pid, signal.SIGKILL)
    parent.join(10)
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            if psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.2)
    else:
        os.kill(grandchild, signal.SIGKILL)
        pytest.fail("orphaned grandchild did not exit after its parent was killed")


def test_initializers_install_the_watchdog(monkeypatch):
    calls = []
    import dreamer.utils.mp_context as mpc
    monkeypatch.setattr(mpc, "start_parent_watchdog", lambda *a, **k: calls.append("watchdog"))

    from dreamer.search.methods.flatland import parallel_eval_worker as w
    from dreamer.search.searchers import shard_concurrency as sc

    class _Shard:
        class cache:
            max_size = 4
    w._pool_init(config.export_configurations(), _Shard(), None)
    sc._shard_process_init(config.export_configurations(), None)
    assert calls == ["watchdog", "watchdog"]

    from dreamer.search.methods.flatland import parallel_eval as pe
    started = {}
    monkeypatch.setattr(pe.SyncManager, "start", lambda self, initializer=None, initargs=(): started.update(i=initializer))
    pe._start_manager()
    assert started["i"] is mpc.start_parent_watchdog


# ---------------------------------------------------------------------------
# prefix loading and torn last lines
# ---------------------------------------------------------------------------

def _row(i):
    import json
    return json.dumps({"trajectory_id": f"t{i}", "constant": "c", "delta": float(i)}) + "\n"


def test_load_seen_trajectories_prefix(tmp_path):
    from dreamer.utils.multi_processing import load_seen_trajectories
    p = tmp_path / "s.jsonl"
    p.write_text(_row(1) + _row(2) + _row(3))
    upto = len(_row(1) + _row(2))
    assert set(load_seen_trajectories(str(p), max_bytes=upto)) == {"t1", "t2"}
    assert set(load_seen_trajectories(str(p), max_bytes=upto + 5)) == {"t1", "t2"}   # partial 3rd row ignored
    assert set(load_seen_trajectories(str(p))) == {"t1", "t2", "t3"}


def test_repair_trailing_fragment(tmp_path, caplog):
    from dreamer.utils.multi_processing import repair_trailing_fragment
    p = tmp_path / "s.jsonl"
    p.write_text(_row(1) + _row(2) + _row(3)[:10])
    repair_trailing_fragment(str(p))
    assert p.read_text() == _row(1) + _row(2)
    assert "incomplete last line" in caplog.text
    repair_trailing_fragment(str(p))                   # complete file: untouched
    assert p.read_text() == _row(1) + _row(2)
    repair_trailing_fragment(str(tmp_path / "missing.jsonl"))   # no file: no error


# ---------------------------------------------------------------------------
# end to end: interrupted + re-run == uninterrupted
# ---------------------------------------------------------------------------

@pytest.mark.timeout(1500)
def test_rerun_after_crash_is_the_same_search(tmp_path, monkeypatch):
    from dreamer import log
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq
    from dreamer.search.searchers.annealing_mod import run_annealing_shard
    from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids

    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    monkeypatch.setattr(config.extraction, "STRATEGY", "heuristic")
    monkeypatch.setattr(config.extraction, "IGNORE_DUPLICATE_SEARCHABLES", True)
    monkeypatch.setattr(config.extraction, "LOAD_SHARD_CACHE", False)
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", None)
    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 12)
    monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 40)
    monkeypatch.setattr(config.search, "ENABLE_MICRO_HILL_CLIMB", False)
    monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
    monkeypatch.setattr(sys_config, "TOTAL_CORES", 1)     # serial: deterministic, no pool
    l2 = log(2)
    shards = ShardExtractor([l2], pFq(l2, 2, 1, -1).to_cmf()).extract()

    for shard in shards:
        out = tmp_path / derive_cmf_and_shard_ids(shard)[1][-8:]
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out))
        out.mkdir()
        sid = derive_cmf_and_shard_ids(shard)[1]
        path = out / f"{sid}.jsonl"
        run_annealing_shard(shard, [l2], True, seed_upto=0)
        full = path.read_text()
        lines = full.splitlines(keepends=True)
        identified = sum('"identified": true' in l for l in lines[: len(lines) // 2])
        if len(lines) < 8 or identified < 2:
            continue                                   # need identified rows in the prefix
        # "killed" after half its rows, mid-write of the next one
        path.write_text("".join(lines[: len(lines) // 2]) + lines[len(lines) // 2][:15])
        run_annealing_shard(shard, [l2], True, seed_upto=0)
        assert path.read_text() == full
        return
    pytest.skip("no 2F1 shard with identified rows in the first half of its SA run")
