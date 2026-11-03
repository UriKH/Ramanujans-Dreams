"""Tests for running simulated-annealing shards concurrently (``shard_concurrency``).

Coverage:
  - plan_shards: automatic K from cores and CMF dimension, the memory cap, a fixed K,
    never more than the number of shards, K = 1 → sequential plan.
  - run_shards: every shard runs once with its core / memory share; a shard whose
    search raises is reported and the others finish; a shard whose process dies is
    re-run alone; one that dies again is reported as failed; never-started shards
    are not penalised by another shard's death.
  - End to end: SimulatedAnnealingMod on all shards of a real 2F1 log(2) CMF writes
    byte-identical per-shard JSONL files sequentially and concurrently.
"""

import os

import pytest

from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.search.searchers import shard_concurrency as sc


# ---------------------------------------------------------------------------
# plan_shards
# ---------------------------------------------------------------------------

@pytest.fixture
def cores(monkeypatch):
    def set_cores(n, fit=None):
        monkeypatch.setattr(sys_config, "TOTAL_CORES", n)
        monkeypatch.setattr("dreamer.utils.memory_budget.memory_worker_cap",
                            lambda requested, log=True: requested if fit is None else min(requested, fit))
        monkeypatch.setattr(sc, "_memory_share", lambda k: 1000 // k)
    return set_cores


class TestPlan:
    def test_auto_from_cores_and_dimension(self, cores):
        cores(128)
        p = sc.plan_shards(18, [5] * 18)
        assert (p.concurrent, p.workers_per_shard) == (12, 10)     # 128 // (2·5)

    def test_never_more_than_shards(self, cores):
        cores(128)
        assert sc.plan_shards(3, [5, 5, 5]).concurrent == 3

    def test_memory_caps_k(self, cores):
        cores(128, fit=35)                                         # memory fits 35 workers
        assert sc.plan_shards(18, [5] * 18).concurrent == 3

    def test_requested_and_workers_override(self, cores):
        cores(64)
        p = sc.plan_shards(18, [5] * 18, requested=4, workers_per_shard=6)
        assert (p.concurrent, p.workers_per_shard, p.memory_mb_per_shard) == (4, 16, 250)

    def test_sequential_when_cores_allow_one(self, cores):
        cores(12)
        p = sc.plan_shards(18, [5] * 18)
        assert p.concurrent == 1 and p.workers_per_shard == 12 and p.memory_mb_per_shard is None


# ---------------------------------------------------------------------------
# run_shards (module-level tasks: shard processes import this module)
# ---------------------------------------------------------------------------

def task_report(sid, marker_dir):
    """Record the core / memory share this process got."""
    with open(os.path.join(marker_dir, sid), "w") as f:
        f.write(f"{config.system.TOTAL_CORES} {config.system.MEMORY_LIMIT_MB}")
    return sid


def task_raise(sid, marker_dir):
    if sid == "bad":
        raise ValueError("search failed")
    return task_report(sid, marker_dir)


def task_die_once(sid, marker_dir):
    """'flaky' kills its process the first time only; 'poison' every time."""
    flag = os.path.join(marker_dir, f"{sid}.tried")
    if sid == "poison" or (sid == "flaky" and not os.path.exists(flag)):
        open(flag, "w").close()
        os._exit(1)
    return task_report(sid, marker_dir)


PLAN = sc.ShardPlan(concurrent=2, workers_per_shard=3, memory_mb_per_shard=777)


@pytest.mark.timeout(300)
class TestRunShards:
    def test_all_run_with_their_share(self, tmp_path):
        done = []
        jobs = [(s, (s, str(tmp_path))) for s in ("a", "b", "c")]
        out = sc.run_shards(task_report, jobs, PLAN, on_done=done.append)
        assert out == {"a": "ok", "b": "ok", "c": "ok"} and sorted(done) == ["a", "b", "c"]
        assert {open(tmp_path / s).read() for s in "abc"} == {"3 777"}

    def test_raising_shard_reported_others_finish(self, tmp_path):
        jobs = [(s, (s, str(tmp_path))) for s in ("a", "bad", "c")]
        out = sc.run_shards(task_raise, jobs, PLAN)
        assert out["a"] == out["c"] == "ok" and out["bad"].startswith("failed: ValueError")

    def test_dead_process_rerun_alone_then_given_up(self, tmp_path):
        jobs = [(s, (s, str(tmp_path))) for s in ("a", "flaky", "poison", "d", "e")]
        out = sc.run_shards(task_die_once, jobs, PLAN)
        assert out["flaky"] == "ok"                     # died once, fine when re-run
        assert out["poison"] == "failed: process died twice"
        assert all(out[s] == "ok" for s in ("a", "d", "e"))
        assert all((tmp_path / s).exists() for s in ("a", "flaky", "d", "e"))


# ---------------------------------------------------------------------------
# End to end: identical rows sequentially and concurrently
# ---------------------------------------------------------------------------

def _log2_shards(monkeypatch):
    from dreamer import log
    from dreamer.extraction.extractor import ShardExtractor
    from dreamer.loading import pFq

    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    monkeypatch.setattr(config.extraction, "STRATEGY", "heuristic")
    monkeypatch.setattr(config.extraction, "IGNORE_DUPLICATE_SEARCHABLES", True)
    monkeypatch.setattr(config.extraction, "LOAD_SHARD_CACHE", False)
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", None)
    l2 = log(2)
    return l2, ShardExtractor([l2], pFq(l2, 2, 1, -1).to_cmf()).extract()


@pytest.mark.timeout(1500)
def test_sa_module_rows_identical_sequential_and_concurrent(tmp_path, monkeypatch):
    from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod

    monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", ())
    monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 6)
    monkeypatch.setattr(config.search, "ANNEAL_MAX_TOTAL_STEPS", 20)
    monkeypatch.setattr(config.search, "ENABLE_MICRO_HILL_CLIMB", True)
    monkeypatch.setattr(config.analysis, "STORE_TRAJECTORIES_SEPARATELY", False)
    monkeypatch.setattr(sys_config, "TOTAL_CORES", 6)
    texts = {}
    for mode, k in (("sequential", 1), ("concurrent", 3)):
        out = tmp_path / mode
        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(out))
        monkeypatch.setattr(config.search, "ANNEAL_CONCURRENT_SHARDS", k)
        l2, shards = _log2_shards(monkeypatch)
        shards = shards[:4]
        SimulatedAnnealingMod({l2: shards}, use_LIReC=True).execute()
        texts[mode] = {p.name: p.read_text() for p in sorted(out.glob("*.jsonl"))}
    assert texts["sequential"], "no shard files written — test is vacuous"
    assert sum(t.count("\n") for t in texts["sequential"].values()) > 10
    assert texts["concurrent"] == texts["sequential"]
