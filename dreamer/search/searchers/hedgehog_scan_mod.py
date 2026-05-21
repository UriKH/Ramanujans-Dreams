"""
SearcherModV1 — search-stage module.

Replaces the old ``SerialSearcher.search()`` / ``DataManager`` flow with a
handler-based pipeline:

  Producer (main thread)
    For each (trajectory, start) pair:
      1. Compute ``trajectory_id`` from (cmf_name, encoding, start, direction)
         — cheap; no symbolic work, no trajectory walk.
      2. If the id is already in the JSONL and every configured Tier-2
         attribute is present, skip immediately — no handler, no walk.
      3. Otherwise build the handler and either a patch dict (partial
         coverage) or a full Tier-1 DTO (new trajectory), and call
         ``push(item)``.

  push(item) — provided by the generic ``worker_pool`` context manager:
    * ``TIER2_ATTRIBUTES`` empty (default) → ``push`` is a synchronous
      writer; the JSONL is written from the main thread, no subprocesses
      created.
    * ``TIER2_ATTRIBUTES`` non-empty → ``push`` enqueues to a worker pool
      that runs ``compute_tier2_for_item`` in background subprocesses and
      a dedicated writer subprocess that appends to the JSONL.

Output files are written to:
    ``sys_config.EXPORT_SEARCH_RESULTS / <constant_name> / <cmf>__<shard_id>.jsonl``
"""

import os
from typing import Callable, List

from dreamer.utils.schemes.searcher_scheme import SearcherModScheme
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.configs.system import sys_config
from dreamer.utils.logger import Logger
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    _stable_id,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
)
from dreamer.utils.multi_processing import (
    compute_tier2_for_item,
    load_seen_trajectories,
    worker_pool,
    write_jsonl_line,
)

search_config = config.search


class SearcherModV1(SearcherModScheme):
    """Search module — deep trajectory search with optional asynchronous Tier-2
    attribute computation.

    All process/queue management lives in :func:`worker_pool`; the searcher
    only supplies the per-shard producer.  When ``TIER2_ATTRIBUTES`` is
    empty no subprocesses are spawned (direct-write fallback).
    """

    def __init__(self, shards: List[Shard], use_LIReC: bool):
        """
        :param shards: Prioritised shards to search.
        :param use_LIReC: Whether to use LIReC for constant identification.
        """
        super().__init__(
            shards,
            use_LIReC,
            description='Search module — deep search with Tier-1 DTO output',
            version='1.3.0',
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> None:
        """Run the search pipeline over all shards."""
        if not self.searchables:
            return

        dir_path = os.path.join(
            sys_config.EXPORT_SEARCH_RESULTS,
            self.searchables[0].const.name,
        )
        os.makedirs(dir_path, exist_ok=True)

        num_workers = sys_config.NUM_BACKGROUND_WORKERS
        config_overrides = config.export_configurations()

        for shard in SmartTQDM(
            self.searchables,
            desc='Searching in shards: ',
            **sys_config.TQDM_CONFIG,
        ):
            self._run_shard(shard, dir_path, num_workers, config_overrides)

    # ------------------------------------------------------------------
    # Per-shard pipeline
    # ------------------------------------------------------------------

    def _run_shard(
        self,
        shard: Shard,
        dir_path: str,
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Run the search for a single shard.

        The ``worker_pool`` context manager chooses MPMC vs direct-write
        based on whether ``TIER2_ATTRIBUTES`` is non-empty.
        """
        cmf_id, shard_id, shard_encoding_str = derive_cmf_and_shard_ids(shard)
        output_path = os.path.join(dir_path, f"{shard.cmf_name}__{shard_id}.jsonl")
        seen_trajectories = load_seen_trajectories(output_path)

        # ``compute_tier2_for_item`` unpacks the ``(traj_matrix, payload)`` tuple
        # the producer pushes and is a fast no-op when no Tier-2 attrs are
        # configured.  We still want it on the main thread in that case so the
        # writer sees the unwrapped payload — so the only thing that depends on
        # ``TIER2_ATTRIBUTES`` is whether to spawn subprocesses at all.
        with worker_pool(
            num_workers=num_workers,
            worker_fn=compute_tier2_for_item,
            writer_fn=write_jsonl_line,
            output_path=output_path,
            config_overrides=config_overrides,
            parallel=bool(search_config.TIER2_ATTRIBUTES),
        ) as push:
            self._produce(
                shard=shard,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=push,
                seen_trajectories=seen_trajectories,
            )

    # ------------------------------------------------------------------
    # Producer
    # ------------------------------------------------------------------

    def _produce(
        self,
        shard: Shard,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
    ) -> None:
        """Iterate over trajectory pairs and hand work to *sink*.

        Three cases per trajectory:

        1. **Complete** — trajectory_id is already in *seen_trajectories* with
           every configured Tier-2 attribute present.  Skip *before* any
           handler construction so no trajectory walk happens.
        2. **Partial** — trajectory_id is known but some Tier-2 attributes
           are missing.  Build the handler (cheap: just the symbolic
           trajectory matrix, no walks), emit a patch dict.
        3. **New** — trajectory_id is unknown.  Build the full Tier-1 DTO
           (this triggers the trajectory walks for delta/limit/etc.) and
           emit it.

        ``sink`` receives ``(trajectory_matrix, payload)`` where *payload*
        is either a ``TrajectoryDTO`` (case 3) or a patch ``dict`` (case 2).
        """
        searcher = SerialSearcher(shard, shard.const, use_LIReC=self.use_LIReC)

        try:
            pairs = searcher.sample_pairs(
                trajectory_generator=search_config.NUM_TRAJECTORIES_FROM_DIM,
            )
        except ValueError as e:
            Logger(
                f"Skipping shard {shard_id}: {e}",
                Logger.Levels.warning,
            ).log()
            return

        desired = set(search_config.TIER2_ATTRIBUTES)

        for traj, start in pairs:
            # Cheap: derive trajectory_id without symbolic work or any walk.
            start_t = _position_to_tuple(start)
            dir_t = _position_to_tuple(traj)
            trajectory_id = _stable_id(
                shard.cmf_name, shard_encoding_str, str(start_t), str(dir_t)
            )

            seen_record = seen_trajectories.get(trajectory_id)
            if seen_record is not None:
                existing_keys = set((seen_record.get("extended_metrics") or {}).keys())
                missing = desired - existing_keys
                if not missing:
                    # Case 1: fully covered — skip without building the handler.
                    continue

                # Case 2: partial coverage.  Build handler (no walks needed —
                # only the symbolic trajectory matrix), emit patch for workers.
                try:
                    handler = TrajectoryAttributesHandler.from_cmf(
                        shard.cmf, traj, start,
                    )
                except Exception as e:
                    Logger(
                        f"Handler error — shard {shard_id}, traj={traj}, start={start}: {e}",
                        Logger.Levels.warning,
                    ).log()
                    continue

                patch: dict = {
                    "trajectory_id": trajectory_id,
                    "extended_metrics": {},
                }
                sink((handler.trajectory_matrix(), patch))
                seen_trajectories[trajectory_id] = {
                    "extended_metrics": dict.fromkeys(existing_keys | missing),
                }
                continue

            # Case 3: new trajectory.  Build the full Tier-1 DTO; this walks.
            try:
                handler = TrajectoryAttributesHandler.from_cmf(
                    shard.cmf, traj, start,
                )
                dto = build_trajectory_dto(
                    handler,
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    cmf_name=shard.cmf_name,
                    shard_encoding_str=shard_encoding_str,
                    start=start,
                    direction=traj,
                )
            except Exception as e:
                Logger(
                    f"Handler error — shard {shard_id}, traj={traj}, start={start}: {e}",
                    Logger.Levels.warning,
                ).log()
                continue

            # Track in-flight locally so duplicate (traj, start) pairs within
            # the same run don't push duplicate work.
            seen_trajectories[trajectory_id] = {
                "extended_metrics": dict.fromkeys(desired),
            }
            sink((handler.trajectory_matrix(), dto))
