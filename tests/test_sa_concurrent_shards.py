"""Tests for running simulated-annealing shards concurrently (``shard_concurrency``).

Coverage:
  - plan_shards: W = max(D + 1, cores // #shards), K from cores (× oversubscription) and the
    memory check, a fixed K / W, never more than the number of shards, K = 1 → sequential plan.
  - run_shards: every shard runs once with its core / memory share; shard processes
    pass the run directory on to their pools; a shard whose search raises is reported
    and the others finish; a shard whose process dies is re-run alone; one that dies
    again is reported as failed, with the cause (exit code); never-started shards are
    not penalised by another shard's death.
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
    """Fix the core budget and the memory check (``fit`` = shards that fit, None = no limit)."""
    def set_cores(n, fit=None, oversubscribe=1.0):
        monkeypatch.setattr(sys_config, "TOTAL_CORES", n)
        monkeypatch.setattr(config.search, "ANNEAL_OVERSUBSCRIBE", oversubscribe)
        monkeypatch.setattr(sc, "_memory_shard_cap", lambda: 10 ** 6 if fit is None else fit)
        monkeypatch.setattr(sc, "_memory_share", lambda k: 1000 // k)
    return set_cores


class TestPlan:
    """Workers per shard W = max(D + 1, cores // #shards); K = min(#shards, cores·oversubscribe // W, memory).

    Measured (investigations/sa_idle_cores): one 5-D shard with 6 workers is as fast as with 9, and does 1.7x more
    work per core than with 18 — so many shards get small pools, and few shards share out all the cores.
    """

    def test_many_shards_small_pools(self, cores):
        cores(128)
        p = sc.plan_shards(3000, [11] * 3000)                      # 6F5-like: D = 11
        assert (p.concurrent, p.workers_per_shard) == (10, 12)     # W = D + 1, K = 128 // 12

    def test_fewer_shards_share_all_cores(self, cores):
        cores(18)
        assert (lambda p: (p.concurrent, p.workers_per_shard))(sc.plan_shards(2, [5, 5])) == (2, 9)
        assert (lambda p: (p.concurrent, p.workers_per_shard))(sc.plan_shards(4, [5] * 4)) == (3, 6)

    def test_never_more_than_shards(self, cores):
        cores(128)
        p = sc.plan_shards(18, [5] * 18)
        assert (p.concurrent, p.workers_per_shard) == (18, 7)      # W = max(6, 128 // 18)

    def test_largest_dimension_sets_the_floor(self, cores):
        cores(18)
        assert sc.plan_shards(10, [3, 5, 3]).workers_per_shard == 6

    def test_oversubscription_runs_more_shards(self, cores):
        cores(128, oversubscribe=1.25)
        p = sc.plan_shards(3000, [11] * 3000)
        assert (p.concurrent, p.workers_per_shard) == (13, 12)     # 160 // 12

    def test_memory_caps_k(self, cores, caplog):
        cores(128, fit=3)
        assert sc.plan_shards(3000, [11] * 3000).concurrent == 3
        assert "3 shard(s) at a time instead of 10" in caplog.text  # says why, and how to override

    def test_requested_and_workers_override(self, cores):
        cores(64)
        p = sc.plan_shards(18, [5] * 18, requested=4, workers_per_shard=6)
        assert (p.concurrent, p.workers_per_shard, p.memory_mb_per_shard) == (4, 16, 250)

    def test_sequential_when_cores_allow_one(self, cores):
        cores(12)
        p = sc.plan_shards(18, [11] * 18)                          # W = 12 = all cores
        assert p.concurrent == 1 and p.workers_per_shard == 12 and p.memory_mb_per_shard is None

    def test_default_oversubscription_is_measured_value(self):
        """Default 1.25: best on Zeus in two rounds with look-ahead on (investigations/sa_idle_cores)."""
        from dreamer.configs.search import SearchConfig
        assert SearchConfig().ANNEAL_OVERSUBSCRIBE == 1.25

    def test_one_shard_gets_everything(self, cores):
        cores(18)
        p = sc.plan_shards(1, [5])
        assert p.concurrent == 1 and p.workers_per_shard == 18


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


def task_monitor_dir(sid, marker_dir):
    """Record the run directory this shard process would hand to its own pools."""
    from dreamer.utils import monitoring
    with open(os.path.join(marker_dir, sid), "w") as f:
        f.write(str(monitoring.worker_run_dir()))
    return sid


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

    def test_shard_processes_report_to_the_run(self, tmp_path, monkeypatch):
        """A shard process joins the parent's run as a worker and passes the run
        directory on to its own eval pool (``worker_run_dir``)."""
        run = tmp_path / "run"
        run.mkdir()
        monkeypatch.setattr("dreamer.utils.monitoring.worker_run_dir", lambda: str(run))
        out = sc.run_shards(task_monitor_dir, [("a", ("a", str(tmp_path)))], PLAN)
        assert out == {"a": "ok"}
        assert open(tmp_path / "a").read() == str(run)

    def test_raising_shard_reported_others_finish(self, tmp_path):
        jobs = [(s, (s, str(tmp_path))) for s in ("a", "bad", "c")]
        out = sc.run_shards(task_raise, jobs, PLAN)
        assert out["a"] == out["c"] == "ok" and out["bad"].startswith("failed: ValueError")

    def test_dead_process_rerun_alone_then_given_up(self, tmp_path):
        jobs = [(s, (s, str(tmp_path))) for s in ("a", "flaky", "poison", "d", "e")]
        out = sc.run_shards(task_die_once, jobs, PLAN)
        assert out["flaky"] == "ok"                     # died once, fine when re-run
        assert out["poison"].startswith("failed: process died twice (")
        assert "exit code 1" in out["poison"]           # the real cause, not a guess
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
