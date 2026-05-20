"""
Multiprocessing utilities for the Ramanujan Agent pipeline.

Provides:
  - ``create_process_pool_executor`` / ``create_pool``: process pools that
    propagate the global config to child workers via ``_init_worker``.
  - ``background_attribute_worker``: consumer process that receives
    ``(trajectory_matrix, TrajectoryDTO | patch_dict)`` tasks, computes
    Tier-2 trajectory attributes (the asynchronous ones — eigenvalues,
    spectral gap, etc.) via ``TrajectoryAttributesHandler``, and forwards
    enriched DTOs to the writer queue.
  - ``dedicated_file_writer``: sink process that is the only writer to a given
    JSONL file, eliminating file-lock races.
  - ``load_seen_trajectories`` / ``load_seen_trajectory_ids``: helpers to
    read existing JSONL trajectory records for cross-run deduplication and
    merge-on-read (later records patch earlier ones).

MPMC flow (one pipeline per shard in SearcherModV1, only when
``search_config.TIER2_ATTRIBUTES`` is non-empty):
    Producer (main) ──► task_queue (bounded) ──► workers (N) ──► results_queue ──► writer

When ``TIER2_ATTRIBUTES`` is empty the searcher skips the pipeline entirely
and writes Tier-1 records directly from the main thread.
"""

import json
import queue
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import multiprocessing as mp
from multiprocessing import Pool

from dreamer.configs import config
from dreamer.utils.logger import Logger


# ---------------------------------------------------------------------------
# Config propagation for child processes
# ---------------------------------------------------------------------------

def _init_worker(config_overrides: dict) -> None:
    """Initialise the global config in a freshly-spawned worker process.

    ``mp.Process`` and ``ProcessPoolExecutor`` workers start with a clean
    namespace; calling this at the top of each worker re-applies the same
    config overrides that were active in the parent.
    """
    config.configure(**config_overrides)


def create_process_pool_executor() -> ProcessPoolExecutor:
    """Return a ``ProcessPoolExecutor`` whose workers inherit the current config."""
    return ProcessPoolExecutor(
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
    )


def create_pool() -> Pool:
    """Return a ``multiprocessing.Pool`` whose workers inherit the current config."""
    return Pool(
        initializer=_init_worker,
        initargs=(config.export_configurations(),),
    )


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------

def load_seen_trajectories(jsonl_path: str) -> Dict[str, dict]:
    """Read and merge all trajectory records from an existing JSONL file.

    Records sharing the same ``trajectory_id`` are merged left-to-right so
    that patch records (which contain only ``trajectory_id`` and a partial
    ``extended_metrics`` dict) are folded into the base record.
    ``extended_metrics`` is deep-merged; all other top-level keys are
    shallow-merged with later records winning.

    Returns an empty dict if the file does not exist.
    """
    merged: Dict[str, dict] = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    tid = record.get("trajectory_id")
                    if tid is None:
                        continue
                    if tid not in merged:
                        merged[tid] = record
                    else:
                        existing_metrics = dict(merged[tid].get("extended_metrics") or {})
                        new_metrics = dict(record.get("extended_metrics") or {})
                        merged[tid].update(record)
                        merged[tid]["extended_metrics"] = {**existing_metrics, **new_metrics}
                except (json.JSONDecodeError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return merged


def load_seen_shards(jsonl_path: str) -> Dict[str, dict]:
    """Read analyzer JSONL records keyed by ``shard_id``.

    Used by the analysis stage for cross-run deduplication: a shard that
    already has a record (with cached ``best_delta`` and ``identified_pct``)
    is skipped on subsequent runs.  Unlike :func:`load_seen_trajectories`,
    records are not deep-merged — the last record for a given shard wins,
    since the analyzer never emits patches.

    Returns an empty dict if the file does not exist.
    """
    merged: Dict[str, dict] = {}
    try:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    sid = record.get("shard_id")
                    if sid is not None:
                        merged[sid] = record
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return merged


def load_seen_trajectory_ids(jsonl_path: str) -> set:
    """Return the set of ``trajectory_id`` values in an existing JSONL file.

    Thin wrapper around :func:`load_seen_trajectories` kept for backward
    compatibility with any call site that only needs the id set.
    """
    return set(load_seen_trajectories(jsonl_path).keys())


# ---------------------------------------------------------------------------
# Background attribute worker  (Tier-2 consumer)
# ---------------------------------------------------------------------------

def background_attribute_worker(
    worker_id: int,
    task_queue: mp.Queue,
    results_queue: mp.Queue,
    config_overrides: dict,
) -> None:
    """Consumer process: compute the configured Tier-2 attributes asynchronously.

    Receives ``(trajectory_matrix, TrajectoryDTO | patch_dict)`` items from
    *task_queue*; a ``None`` item signals shutdown.  For each item, builds a
    ``TrajectoryAttributesHandler`` from the trajectory matrix and computes
    every attribute listed in ``search_config.TIER2_ATTRIBUTES`` that is
    **not already present** in ``extended_metrics``.  Results are written
    back into the same ``extended_metrics`` mapping.

    Patch-dict payloads (``trajectory_id`` + partial ``extended_metrics``)
    are treated identically to full DTOs — the same missing-attribute filter
    applies, so patches stay minimal.  When ``traj_matrix`` is ``None`` the
    worker simply forwards the payload without computing anything (used by
    the producer to short-circuit when no Tier-2 work is needed).

    Individual attribute failures are stored as ``<name>_error`` in
    ``extended_metrics`` rather than crashing the worker (see
    ``compute_attributes``).  A fatal error during handler construction is
    recorded under ``worker_error`` and the partial payload is still
    forwarded so the writer can flush it.
    """
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler
    from dreamer.utils.storage.attribute_registry import compute_attributes
    from dreamer.configs import config

    _init_worker(config_overrides)
    Logger(f"Worker {worker_id} started.", Logger.Levels.debug).log()

    attrs_to_compute = config.search.TIER2_ATTRIBUTES

    while True:
        try:
            item = task_queue.get(timeout=3)
        except queue.Empty:
            # No item yet — keep looping until sentinel arrives
            continue

        if item is None:
            Logger(f"Worker {worker_id} stopping.", Logger.Levels.debug).log()
            break

        traj_matrix, dto_or_patch = item
        # Patch dicts (plain dict with trajectory_id + partial extended_metrics)
        # are handled the same as full DTOs — only missing attributes are computed.
        is_patch = isinstance(dto_or_patch, dict)
        extended_metrics = (
            dto_or_patch["extended_metrics"]
            if is_patch
            else dto_or_patch.extended_metrics
        )
        tid = (
            dto_or_patch.get("trajectory_id", "?")
            if is_patch
            else dto_or_patch.trajectory_id
        )

        missing = [a for a in attrs_to_compute if a not in extended_metrics]
        if missing and traj_matrix is not None:
            try:
                handler = TrajectoryAttributesHandler(traj_matrix)
                extended_metrics.update(
                    compute_attributes(handler, missing, on_error="store")
                )
            except Exception as e:
                Logger(
                    f"Worker {worker_id} error on trajectory {tid}: {e}",
                    Logger.Levels.warning,
                ).log()
                extended_metrics["worker_error"] = str(e)

        results_queue.put(dto_or_patch)


# ---------------------------------------------------------------------------
# Dedicated file writer  (sink)
# ---------------------------------------------------------------------------

def dedicated_file_writer(
    results_queue: mp.Queue,
    output_file_path: str,
    config_overrides: dict,
) -> None:
    """Sink process: the only process that writes to *output_file_path*.

    Receives enriched ``TrajectoryDTO`` objects from *results_queue* and
    appends them as JSON Lines.  A ``None`` sentinel signals shutdown.

    Being the sole writer eliminates file-lock races in the MPMC pipeline.
    """
    _init_worker(config_overrides)
    Logger("Writer started.", Logger.Levels.debug).log()

    with open(output_file_path, "a") as f:
        while True:
            dto_or_patch = results_queue.get()
            if dto_or_patch is None:
                Logger("Writer stopping.", Logger.Levels.debug).log()
                break
            # Patch dicts are written as plain JSON; full DTOs use to_json_line().
            if isinstance(dto_or_patch, dict):
                line = json.dumps(dto_or_patch)
            else:
                line = dto_or_patch.to_json_line()
            f.write(line + "\n")
            f.flush()
