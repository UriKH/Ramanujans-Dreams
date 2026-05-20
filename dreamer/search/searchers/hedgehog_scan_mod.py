"""
SearcherModV1 — search-stage module.

Replaces the old ``SerialSearcher.search()`` / ``DataManager`` flow with a
handler-based pipeline:

  Producer (main thread)
    For each (trajectory, start) pair:
      1. Compute ``trajectory_id`` from (cmf_name, encoding, start, direction)
         — this is cheap; no symbolic work, no trajectory walk.
      2. If the id is already in the JSONL and every configured Tier-2
         attribute is present, *skip immediately* — no handler, no walk.
      3. Otherwise build the handler, Tier-1 DTO (or patch dict for partial
         coverage), and hand it to the sink.

  Sink — depends on ``search_config.TIER2_ATTRIBUTES``:
    * **empty** → main thread writes the DTO directly to the JSONL.
      No worker or writer subprocesses are created — they would be pure
      overhead since no Tier-2 work is required.
    * **non-empty** → records are pushed onto a bounded task queue consumed
      by ``NUM_BACKGROUND_WORKERS`` worker processes that compute the
      missing Tier-2 attributes, then forwarded to a dedicated writer
      process that owns the JSONL file (the sole writer eliminates
      file-lock races).

A ``try/finally`` guarantees shutdown sentinels are sent even if the
producer raises, so worker and writer processes never hang.

Output files are written to:
    ``sys_config.EXPORT_SEARCH_RESULTS / <constant_name> / <cmf>__<shard_id>.jsonl``
"""

import json
import multiprocessing as mp
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
    background_attribute_worker,
    dedicated_file_writer,
    load_seen_trajectories,
)

search_config = config.search


class SearcherModV1(SearcherModScheme):
    """Search module — deep trajectory search with optional asynchronous Tier-2
    attribute computation.

    When ``search_config.TIER2_ATTRIBUTES`` is non-empty, an MPMC pipeline
    per shard runs ``NUM_BACKGROUND_WORKERS`` worker processes and a
    dedicated writer.  When empty (the default), the main thread writes
    Tier-1 records directly to the JSONL — no subprocesses are spawned.
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
            version='1.2.0',
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

        Chooses between the direct-write path (no Tier-2 work configured)
        and the full MPMC pipeline.
        """
        cmf_id, shard_id, shard_encoding_str = derive_cmf_and_shard_ids(shard)
        output_path = os.path.join(dir_path, f"{shard.cmf_name}__{shard_id}.jsonl")
        seen_trajectories = load_seen_trajectories(output_path)

        if search_config.TIER2_ATTRIBUTES:
            self._run_shard_mpmc(
                shard, shard_id, shard_encoding_str, cmf_id,
                output_path, seen_trajectories, num_workers, config_overrides,
            )
        else:
            self._run_shard_direct(
                shard, shard_id, shard_encoding_str, cmf_id,
                output_path, seen_trajectories,
            )

    # -- direct-write path (no Tier-2 attributes configured) ----------

    def _run_shard_direct(
        self,
        shard: Shard,
        shard_id: str,
        shard_encoding_str: str,
        cmf_id: str,
        output_path: str,
        seen_trajectories: dict,
    ) -> None:
        """Main thread writes DTOs straight to the JSONL.

        Used when ``TIER2_ATTRIBUTES`` is empty so no async work is needed.
        Avoids the per-shard cost of spawning worker and writer subprocesses.
        """
        with open(output_path, "a") as fout:
            def sink(_traj_matrix, payload) -> None:
                line = (
                    json.dumps(payload) if isinstance(payload, dict)
                    else payload.to_json_line()
                )
                fout.write(line + "\n")
                fout.flush()

            self._produce(
                shard=shard,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=sink,
                seen_trajectories=seen_trajectories,
            )

    # -- MPMC pipeline path (Tier-2 attributes configured) ------------

    def _run_shard_mpmc(
        self,
        shard: Shard,
        shard_id: str,
        shard_encoding_str: str,
        cmf_id: str,
        output_path: str,
        seen_trajectories: dict,
        num_workers: int,
        config_overrides: dict,
    ) -> None:
        """Full MPMC pipeline: producer → task queue → workers → writer."""
        # Bounded task queue prevents producers outrunning consumers (RAM safety).
        # Size is large enough that the producer rarely stalls waiting for workers.
        task_queue: mp.Queue = mp.Queue(maxsize=max(32, 4 * num_workers))
        results_queue: mp.Queue = mp.Queue()

        workers = [
            mp.Process(
                target=background_attribute_worker,
                args=(i, task_queue, results_queue, config_overrides),
            )
            for i in range(num_workers)
        ]
        writer = mp.Process(
            target=dedicated_file_writer,
            args=(results_queue, output_path, config_overrides),
        )

        for w in workers:
            w.start()
        writer.start()

        try:
            def sink(traj_matrix, payload) -> None:
                task_queue.put((traj_matrix, payload))

            self._produce(
                shard=shard,
                cmf_id=cmf_id,
                shard_id=shard_id,
                shard_encoding_str=shard_encoding_str,
                sink=sink,
                seen_trajectories=seen_trajectories,
            )
        finally:
            # Send one sentinel per worker, then signal the writer.
            # Guaranteed even if the producer raised — prevents hanging processes.
            for _ in workers:
                task_queue.put(None)
            for w in workers:
                w.join()
            results_queue.put(None)
            writer.join()

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
        """Iterate over trajectory pairs, hand work to *sink*.

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
                sink(handler.trajectory_matrix(), patch)
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
            sink(handler.trajectory_matrix(), dto)
