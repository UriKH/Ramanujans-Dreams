"""
Process-based δ-evaluation for the Genetic Algorithm.

The GA evaluates a *batch* of new trajectory directions per generation.  The
per-direction cost is the symbolic walk (``TrajectoryAttributesHandler.from_cmf``
+ δ / LIReC identification) — heavy, GIL-holding work that thread pools barely
speed up.  A feasibility benchmark (``examples/benchmark_ga_parallel_eval.py``)
showed the walk is ~380 ms while pickling the result back is ~0.3% of that, so a
**persistent per-shard process pool** is a large, clean win.

Design
------
* The pool is created **once per shard** (the shard + start are constant-
  independent, so they are handed to the workers a single time via the
  ``initializer`` — copy-on-write / one pickle, not per task).
* Each task sends only the small per-walk payload (direction + constant +
  ids) and returns the exact ``(trajectory_matrix, value_sympy, dto)`` tuple
  the main process feeds to its ``sink`` — so the main process keeps sole
  ownership of ``seen_trajectories`` / ``handler_cache`` and the file sink,
  preserving the cross-run dedup semantics.

Only genuinely-new walks (Case C in :func:`evaluate_in_flatland`) are dispatched
here; Case A (δ already cached) and Case B (handler cached this run) are
resolved cheaply in the main process before dispatch.
"""

from multiprocessing import Pool
from typing import Optional

from dreamer.configs import config


# Per-worker context, populated once by the pool initializer.
_POOL_STATE: dict = {}


def _pool_init(config_overrides: dict, shard, start) -> None:
    """Pool initializer: re-apply config and stash the per-shard walk context.

    :param config_overrides: ``config.export_configurations()`` from the parent.
    :param shard: The shard being searched (constant-independent).
    :param start: Interior start :class:`Position` (constant-independent).
    """
    config.configure(**config_overrides)
    _POOL_STATE["shard"] = shard
    _POOL_STATE["start"] = start


def _pool_walk(args):
    """Worker task: perform one Case-C walk and return the sink tuple.

    :param args: ``(direction, constant, cmf_id, shard_id, shard_encoding_str)``.
    :return: ``(trajectory_matrix, value_sympy, dto)`` — identical to what the
        serial Case-C path passes to the sink.
    """
    from dreamer.utils.storage.trajectory_attributes import (
        TrajectoryAttributesHandler,
        build_trajectory_dto,
    )

    direction, constant, cmf_id, shard_id, shard_encoding_str = args
    shard = _POOL_STATE["shard"]
    start = _POOL_STATE["start"]
    handler = TrajectoryAttributesHandler.from_cmf(
        shard.cmf, direction, start, constant=None, searchable=shard
    )
    dto = build_trajectory_dto(
        handler,
        cmf_id=cmf_id,
        shard_id=shard_id,
        cmf_name=shard.cmf_name,
        shard_encoding_str=shard_encoding_str,
        start=start,
        direction=direction,
        constants=[constant],
    )
    return (handler.trajectory_matrix(), constant.value_sympy, dto)


def make_eval_pool(shard, start, workers: int) -> Optional[Pool]:
    """Create a persistent per-shard evaluation pool, or ``None`` if disabled.

    :param shard: The shard being searched (handed to workers once).
    :param start: Interior start :class:`Position`.
    :param workers: Desired worker count; ``<= 1`` returns ``None`` (serial).
    :return: A ready :class:`multiprocessing.Pool`, or ``None``.
    """
    if workers is None or workers <= 1:
        return None
    return Pool(
        processes=workers,
        initializer=_pool_init,
        initargs=(config.export_configurations(), shard, start),
    )
