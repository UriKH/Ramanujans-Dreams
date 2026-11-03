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
* ``K`` = min(#shards, cores // workers per shard, what memory allows for
  ``K × workers``); a configured value overrides the automatic choice.
* A shard process that dies (e.g. out of memory) breaks the executor: the
  shards that were running are re-run **one at a time** at the end (a shard
  re-reads its own JSONL, so finished walks are not repeated); a shard that
  kills its process again is reported as failed.  Other shards are unaffected.
  A shard whose search *raises* is reported and not retried.
* Shard processes report to the run monitor as workers, and their eval pools
  report to the same run directory.

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

    :param n_shards: Number of shards to search.
    :param dims: CMF dimension ``D`` of each shard; the default workers per shard
        is ``2·max(D)`` (the neighbour walks of one SA step).
    :param requested: ``> 0`` fixes ``K`` (``1`` = sequential); ``0`` = automatic.
    :param workers_per_shard: ``> 0`` overrides the per-shard worker count.
    :return: The :class:`ShardPlan`; ``concurrent == 1`` means "run sequentially".
    """
    from dreamer.utils.multi_processing import search_worker_budget
    from dreamer.utils.memory_budget import memory_worker_cap

    total = max(1, int(search_worker_budget()))
    want = int(workers_per_shard) or 2 * max([int(d) for d in dims] or [1])
    want = max(1, min(want, total))
    if requested and int(requested) > 0:
        k = int(requested)
    else:
        k_cores = total // want
        k_memory = max(1, memory_worker_cap(k_cores * want, log=False) // want)
        k = min(k_cores, k_memory)
    k = max(1, min(k, int(n_shards)))
    if k <= 1:
        return ShardPlan(1, total, None)
    return ShardPlan(k, max(1, total // k), _memory_share(k))


def _memory_share(k: int) -> Optional[int]:
    """Each shard process's ``MEMORY_LIMIT_MB``: (budget − reserve) / k, or None."""
    if not getattr(config.system, "MEMORY_AWARE_WORKERS", True):
        return None
    from dreamer.utils.allocation import allocated_memory_mb
    from dreamer.utils.memory_budget import available_memory_mb
    budget = allocated_memory_mb() or available_memory_mb()
    reserve = float(getattr(config.system, "MEMORY_RESERVE_MB", 1024))
    return max(1, int((float(budget) - reserve) // k))


def _shard_process_init(config_overrides: dict, monitor_dir: Optional[str]) -> None:
    """Initializer of a shard process: the parent's config (with this process's
    core / memory share) and the run monitor as a worker."""
    config.configure(**config_overrides)
    if monitor_dir:
        try:
            from dreamer.utils import monitoring
            monitoring.init_worker(monitor_dir)
        except Exception:      # monitoring must never stop a search
            pass


def run_shards(task: Callable, jobs: Sequence[Tuple[str, tuple]], plan: ShardPlan, *,
               label: str = "shard search",
               on_done: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    """Run ``task(*args)`` for every ``(shard_id, args)`` in *jobs*, ``plan.concurrent`` at a time.

    :param task: Module-level (picklable) per-shard function.
    :param jobs: ``(shard_id, args)`` pairs, in the order to start them.
    :param plan: From :func:`plan_shards` (``concurrent >= 2``).
    :param label: Name used in log lines.
    :param on_done: Called with each shard id as it finishes (progress).
    :return: ``{shard_id: "ok" | "failed: <reason>"}``.
    """
    from dreamer.utils import monitoring
    from dreamer.utils.mp_context import pool_context

    overrides = config.export_configurations()
    overrides["system"] = dict(overrides.get("system", {}), TOTAL_CORES=plan.workers_per_shard)
    if plan.memory_mb_per_shard is not None:
        overrides["system"]["MEMORY_LIMIT_MB"] = plan.memory_mb_per_shard
    initargs = (overrides, monitoring.worker_run_dir())
    Logger(
        f"{label}: {len(jobs)} shard(s), {plan.concurrent} at a time, "
        f"{plan.workers_per_shard} worker(s) each"
        + (f", {plan.memory_mb_per_shard} MB each" if plan.memory_mb_per_shard else ""),
        Logger.Levels.info,
    ).log()

    outcome: Dict[str, str] = {}
    by_id = dict(jobs)
    def finish(sid: str, result: str) -> None:
        outcome[sid] = result
        if on_done is not None:
            on_done(sid)

    def run_batch(ids: List[str], workers: int) -> List[str]:
        """Run *ids*, at most *workers* in flight; return those in flight when a process died.

        Shards are submitted only as slots free up, so a dead process implicates
        just the shards that were running; the rest continue on a fresh pool.
        """
        ctx = pool_context()
        pending, died = list(ids), []
        while pending:
            with cf.ProcessPoolExecutor(workers, mp_context=ctx, initializer=_shard_process_init,
                                        initargs=initargs) as ex:
                inflight: Dict[cf.Future, str] = {}
                broken = False
                while (pending or inflight) and not broken:
                    while pending and len(inflight) < workers:
                        sid = pending.pop(0)
                        inflight[ex.submit(task, *by_id[sid])] = sid
                    done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        sid = inflight.pop(fut)
                        try:
                            fut.result()
                            finish(sid, "ok")
                        except BrokenProcessPool:
                            broken = True
                            died.append(sid)
                        except Exception as exc:     # the search itself raised — report, go on
                            Logger(f"{label}: shard {sid} failed: {exc!r}", Logger.Levels.exception).log()
                            finish(sid, f"failed: {exc!r}")
                if broken:                           # everything still running went down with it
                    died.extend(inflight.values())
        return died

    suspects = run_batch([sid for sid, _ in jobs], plan.concurrent)
    if suspects:
        Logger(
            f"{label}: a shard process died (likely out of memory); re-running "
            f"{len(suspects)} shard(s) that were in flight, one at a time.",
            Logger.Levels.warning,
        ).log()
    for sid in suspects:
        if run_batch([sid], 1):
            Logger(f"{label}: shard {sid} killed its process again — giving up on it.",
                   Logger.Levels.exception).log()
            finish(sid, "failed: process died twice")
    return outcome
