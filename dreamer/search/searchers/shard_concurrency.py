"""
Run independent shard searches concurrently, one process per shard.

Why
---
A search optimiser handles shards one after another, and one shard's search can
use only a few workers (simulated annealing: ``2·D`` neighbour walks per step).
On a many-core machine most cores idle, and the run takes the **sum** of the
shards' times.  Shards are independent searches — each has its own seed, its own
RNG stream (derived from its ``shard_id``), its own ``(p, q)`` cache and its own
output file — so running ``K`` of them side by side changes no shard's result;
the run then takes about ``ceil(#shards / K)`` shard-times for the same work.

How
---
* ``K`` shard processes (``concurrent.futures`` workers, non-daemonic, so each
  can run its own eval pool).  Each executes the module's own per-shard function
  — the exact code of the sequential loop — with ``system.TOTAL_CORES`` set to
  its share of the cores and ``system.MEMORY_LIMIT_MB`` to its share of the
  memory, so its eval pool sizes itself with the existing machinery.
* Workers per shard ``W = max(D + 1, cores // #shards)`` and
  ``K = min(#shards, cores · ANNEAL_OVERSUBSCRIBE // W, shard processes memory fits)``
  (see :func:`plan_shards`); configured values override the automatic choice.
* Every shard runs in its own one-process executor, so a shard process that
  dies (e.g. killed for memory) takes down only that shard: it is re-queued at the
  front and restarts at once, its eval pool starting at one worker and growing
  with memory; after ``MAX_SHARD_ATTEMPTS`` deaths it is reported as failed, with
  the cause (executor message and exit code / signal).  Other shards are never
  re-run.  A re-run is the same search as an uninterrupted
  run: it seeds only from the rows its JSONL had before the first attempt (size
  recorded by the caller), while the killed attempt's rows serve as cached walks
  and a half-written last line is cut off.  The dead process's eval workers and
  (p, q) Manager exit on their own (``mp_context.start_parent_watchdog``), so the
  run still exits.
  A shard whose search *raises* is reported and not retried.
* Shard processes report to the run monitor as workers (events, watchdog), and
  their eval pools report to the same run directory.  Progress *nodes* exist only
  in the main process, so the web view shows the shards-finished bar, not each
  shard's own progress.

Caveat on "identical": a shard's rows are the same as in a sequential run as
far as identification is independent of evaluation order.  It is not fully: the
shard's (p, q) form cache is shared live between its eval workers, so which
cached form a trajectory gets can depend on timing (and on the worker count),
also without concurrency.  Seen once on an old code base; under investigation.

Only simulated annealing uses this so far (GitHub issue #74 tracks the others).
"""

from __future__ import annotations

import concurrent.futures as cf
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from dreamer.configs import config
from dreamer.utils.logger import Logger


@dataclass(frozen=True)
class ShardPlan:
    """How a concurrent shard run is laid out."""

    concurrent: int                  #: shards run at once (K)
    workers_per_shard: int           #: ``TOTAL_CORES`` given to each shard process
    memory_mb_per_shard: Optional[int]  #: ``MEMORY_LIMIT_MB`` per shard process (None: no split)


def plan_shards(n_shards: int, dims: Sequence[int], *, requested: int = 0,
                workers_per_shard: int = 0) -> ShardPlan:
    """Choose how many shards run at once and what each gets.

    Rules (measured on a laptop, ``context`` study ``sa_idle_cores``): one SA shard
    gains little from many workers — a 5-D shard with ``D + 1 = 6`` workers ran as fast
    as with 9, and did 1.7x more work per core than with 18 — so with many shards each
    gets a small pool and more shards run side by side; with few shards the cores are
    shared out among them.

    * workers per shard ``W = max(D + 1, cores // #shards)`` (``D`` = largest CMF
      dimension), at most all cores;
    * ``K = min(#shards, floor(cores · ANNEAL_OVERSUBSCRIBE) // W, shards memory fits)``.

    The memory check counts only what a shard needs to start (its process and one
    worker); each shard's eval pool then sizes itself within its memory share and
    shrinks when memory runs low.

    :param n_shards: Number of shards to search.
    :param dims: CMF dimension ``D`` of each shard.
    :param requested: ``> 0`` fixes ``K`` (``1`` = sequential); ``0`` = automatic.
    :param workers_per_shard: ``> 0`` overrides ``W``.
    :return: The :class:`ShardPlan`; ``concurrent == 1`` means "run sequentially".
    """
    from dreamer.utils.multi_processing import search_worker_budget

    total = max(1, int(search_worker_budget()))
    n_shards = max(1, int(n_shards))
    floor_w = max([int(d) for d in dims] or [1]) + 1
    want = int(workers_per_shard) or max(floor_w, total // n_shards)
    want = max(1, min(want, total))
    if requested and int(requested) > 0:
        k = int(requested)
    else:
        over = max(1.0, float(getattr(config.search, "ANNEAL_OVERSUBSCRIBE", 1.0) or 1.0))
        k_cores = max(1, int(total * over) // want)
        k_memory = _memory_shard_cap()
        k = min(k_cores, k_memory)
        if k_memory < k_cores and n_shards > k_memory:
            Logger(
                f"Concurrent shards: memory fits ~{k_memory} shard process(es) with one worker each, "
                f"so {k_memory} shard(s) at a time instead of {min(k_cores, n_shards)}; set "
                f"ANNEAL_CONCURRENT_SHARDS (and MEMORY_LIMIT_MB) to override.",
                Logger.Levels.info,
            ).log()
    k = max(1, min(k, n_shards))
    if k <= 1:
        return ShardPlan(1, total, None)
    # A fixed K splits all cores; otherwise W, topped up by cores left over (never below W,
    # so oversubscription keeps W workers per shard: K·W > cores).
    workers = max(1, total // k) if requested and int(requested) > 0 else max(want, total // k)
    return ShardPlan(k, workers, _memory_share(k))


def _memory_shard_cap() -> int:
    """Shard processes that fit in memory, each with its process and one worker (>= 1)."""
    if not getattr(config.system, "MEMORY_AWARE_WORKERS", True):
        return 10 ** 6
    from dreamer.utils.memory_budget import available_memory_mb, measured_worker_baseline_mb, per_worker_mb
    reserve = float(getattr(config.system, "MEMORY_RESERVE_MB", 1024))
    each = measured_worker_baseline_mb() + per_worker_mb()
    return max(1, int((available_memory_mb() - reserve) // each)) if each > 0 else 10 ** 6


def _memory_share(k: int) -> Optional[int]:
    """Each shard process's ``MEMORY_LIMIT_MB``: (budget − reserve) / k, or None."""
    if not getattr(config.system, "MEMORY_AWARE_WORKERS", True):
        return None
    from dreamer.utils.allocation import allocated_memory_mb
    from dreamer.utils.memory_budget import available_memory_mb
    budget = allocated_memory_mb() or available_memory_mb()
    reserve = float(getattr(config.system, "MEMORY_RESERVE_MB", 1024))
    return max(1, int((float(budget) - reserve) // k))


def _break_reason(exc: BaseException, executor) -> str:
    """Why a shard executor broke: the executor's message plus the dead processes'
    exit codes (a negative code is the signal; SIGKILL is typical of the OOM killer)."""
    import signal
    parts = [str(exc) or type(exc).__name__]
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause: {cause!r}")
    try:                                         # private attribute — best effort only
        codes = sorted({p.exitcode for p in getattr(executor, "_processes", {}).values()
                        if p.exitcode not in (None, 0)})
    except Exception:
        codes = []
    for code in codes:
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = f"signal {-code}"
            hint = " (e.g. out of memory)" if name == "SIGKILL" else ""
            parts.append(f"exit code {code} = {name}{hint}")
        else:
            parts.append(f"exit code {code}")
    return "; ".join(parts)


def _shard_process_init(config_overrides: dict, monitor_dir: Optional[str]) -> None:
    """Initializer of a shard process: the parent's config (with this process's
    core / memory share) and the run monitor as a worker."""
    config.configure(**config_overrides)
    from dreamer.utils import mp_context
    mp_context.start_parent_watchdog()          # exit if the run's main process dies
    if monitor_dir:
        try:
            from dreamer.utils import monitoring
            monitoring.init_worker(monitor_dir)
        except Exception:      # monitoring must never stop a search
            pass


#: Attempts per shard before it is reported as failed (its process died every time).
MAX_SHARD_ATTEMPTS = 2


def run_shards(task: Callable, jobs: Sequence[Tuple[str, tuple]], plan: ShardPlan, *,
               label: str = "shard search",
               on_done: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """Run ``task(*args)`` for every ``(shard_id, args)`` in *jobs*, ``plan.concurrent`` at a time.

    Every shard gets its **own** one-process executor, so a shard process that dies
    (e.g. killed for memory) takes down only that shard: it is re-queued at the
    front and restarts as soon as its slot is free, with its eval pool starting at
    one worker (``POOL_SLOW_START = 1``; it grows as memory allows).  After
    :data:`MAX_SHARD_ATTEMPTS` deaths it is reported as failed, with the cause
    (executor message, exit code / signal).  A shard whose search *raises* is
    reported at once (a bug, not a resource problem).  Other shards are never re-run.

    :param task: Module-level (picklable) per-shard function.
    :param jobs: ``(shard_id, args)`` pairs, in the order to start them.
    :param plan: From :func:`plan_shards` (``concurrent >= 2``).
    :param label: Name used in log lines.
    :param on_done: Called with each shard id as it finishes (progress).
    :return: ``{shard_id: "ok" | "failed: <reason>"}``.
    """
    from collections import deque
    from dreamer.utils import monitoring
    from dreamer.utils.mp_context import pool_context

    overrides = config.export_configurations()
    overrides["system"] = dict(overrides.get("system", {}), TOTAL_CORES=plan.workers_per_shard)
    if plan.memory_mb_per_shard is not None:
        overrides["system"]["MEMORY_LIMIT_MB"] = plan.memory_mb_per_shard
    retry_overrides = dict(overrides, system=dict(overrides["system"], POOL_SLOW_START=1))
    run_dir = monitoring.worker_run_dir()
    Logger(
        f"{label}: {len(jobs)} shard(s), {plan.concurrent} at a time, "
        f"{plan.workers_per_shard} worker(s) each"
        + (f", {plan.memory_mb_per_shard} MB each" if plan.memory_mb_per_shard else ""),
        Logger.Levels.info,
    ).log()

    outcome: Dict[str, str] = {}
    by_id = dict(jobs)
    attempts: Dict[str, int] = {sid: 0 for sid, _ in jobs}
    deaths: Dict[str, List[str]] = {sid: [] for sid, _ in jobs}
    pending = deque(sid for sid, _ in jobs)
    inflight: Dict[cf.Future, Tuple[str, cf.ProcessPoolExecutor]] = {}
    ctx = pool_context()

    def finish(sid: str, result: str) -> None:
        outcome[sid] = result
        if on_done is not None:
            on_done(sid)

    def start(sid: str) -> None:
        attempts[sid] += 1
        initargs = (overrides if attempts[sid] == 1 else retry_overrides, run_dir)
        ex = cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_shard_process_init, initargs=initargs)
        inflight[ex.submit(task, *by_id[sid])] = (sid, ex)

    try:
        while pending or inflight:
            while pending and len(inflight) < plan.concurrent:
                start(pending.popleft())
            done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                sid, ex = inflight.pop(fut)
                try:
                    fut.result()
                    finish(sid, "ok")
                except BrokenProcessPool as exc:
                    why = _break_reason(exc, ex)
                    deaths[sid].append(why)
                    if attempts[sid] < MAX_SHARD_ATTEMPTS:
                        Logger(
                            f"{label}: the process of shard {sid} died ({why}); restarting it now "
                            f"(attempt {attempts[sid] + 1} of {MAX_SHARD_ATTEMPTS}, its pool starts at one "
                            f"worker and grows with memory). Other shards are not affected.",
                            Logger.Levels.warning,
                        ).log()
                        pending.appendleft(sid)
                    else:
                        Logger(f"{label}: shard {sid} killed its process {attempts[sid]} times ({why}) — "
                               f"giving up on it.", Logger.Levels.exception).log()
                        finish(sid, f"failed: process died {attempts[sid]} times ({why})")
                except Exception as exc:         # the search itself raised — report, go on
                    Logger(f"{label}: shard {sid} failed: {exc!r}", Logger.Levels.exception).log()
                    finish(sid, f"failed: {exc!r}")
                finally:
                    ex.shutdown(wait=True, cancel_futures=True)
    finally:
        for _sid, ex in inflight.values():       # interrupted: do not leave shard processes behind
            ex.shutdown(wait=False, cancel_futures=True)
    return outcome
