"""
AnalyzerModV1 — analysis-stage module.

For each constant, samples trajectories in every candidate shard and
records the resulting Tier-1 attributes (`delta`, `identified`, plus the
rest of the ``TrajectoryDTO`` core fields) as one JSONL line per
trajectory.  Filters and ranks shards by best observed delta.

**Per-trajectory dedup (never per-shard)** — sampling itself is cheap, so
it always runs.  The expensive trajectory walk happens only for sampled
pairs whose Tier-1 fields are not yet present in the shard's JSONL.
Records already in the file (e.g. from a prior analyzer run or a previous
search-stage pass) are reused.  This lets you re-run with a different
sampling strategy and pick up incremental data without losing or
recomputing anything.

**Single canonical store** — the analyzer's per-trajectory output is
written to the same per-shard JSONL the searcher uses:
``EXPORT_SEARCH_RESULTS/<constant>/<cmf>__<shard_id>.jsonl``.  This is the
canonical location for any per-trajectory record (analyzer + search +
post-process all read/write it via merge-on-read).
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from dreamer.utils.schemes.analysis_scheme import AnalyzerModScheme
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.module import CatchErrorInModule
from dreamer.utils.constants.constant import Constant
from dreamer.configs.system import sys_config
from dreamer.configs import config
from dreamer.extraction.shard import Shard
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    _stable_id,
    build_trajectory_dto,
    derive_cmf_and_shard_ids,
)
from dreamer.utils.multi_processing import load_seen_trajectories
from dreamer.search.methods.hedgehog_scan import SerialSearcher

analysis_config = config.analysis


class AnalyzerModV1(AnalyzerModScheme):
    """Analysis module: filters and ranks shards by Tier-1 trajectory attributes.

    For each shard, samples trajectories and computes ``delta`` + ``identified``
    per trajectory.  Records are appended to the shared per-shard JSONL
    (also used by the searcher) so subsequent runs — or the search stage
    itself — can dedup at the trajectory level.

    Shards passing the identified-percentage threshold are sorted by best
    observed delta and returned for deeper search.
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        """
        :param cmf_data: Mapping from each constant to its list of shards.
        """
        super().__init__(
            cmf_data,
            desc='Analysis module — per-trajectory dedup, ranks shards by best delta',
            version='3',
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """Filter and rank shards for every constant.

        Returns a mapping from constant → shards sorted by best delta
        (descending), then by dimension (ascending, as a tie-breaker).
        """
        result: Dict[Constant, List[Searchable]] = {c: [] for c in self.cmf_data.keys()}

        for constant, shards in SmartTQDM(
            self.cmf_data.items(),
            desc='Analyzing constants and their CMFs',
            **sys_config.TQDM_CONFIG,
        ):
            Logger(
                Logger.buffer_print(
                    sys_config.LOGGING_BUFFER_SIZE,
                    f'Analyzing for {constant.name}',
                    '=',
                ),
                Logger.Levels.message,
            ).log()

            const_dir = os.path.join(sys_config.EXPORT_SEARCH_RESULTS, constant.name)
            os.makedirs(const_dir, exist_ok=True)

            shard_best_delta: Dict[Shard, float] = {}

            for shard in shards:
                cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)
                shard_jsonl_path = os.path.join(
                    const_dir, f"{shard.cmf_name}__{shard_id}.jsonl",
                )
                seen_trajectories = load_seen_trajectories(shard_jsonl_path)

                best_delta, identified_pct, sampled = self._analyze_shard(
                    shard,
                    constant,
                    cmf_id=cmf_id,
                    shard_id=shard_id,
                    encoding_str=encoding_str,
                    jsonl_path=shard_jsonl_path,
                    seen_trajectories=seen_trajectories,
                )
                if not sampled:
                    # ``_analyze_shard`` already logged a sampling failure;
                    # skip the shard entirely (don't include in priorities).
                    continue

                if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                    bd = f'{best_delta:.4f}' if best_delta is not None else 'N/A'
                    Logger(
                        f"Shard {shard_id[:8]}…  "
                        f"best_delta={bd}  identified={identified_pct * 100:.1f}%",
                        Logger.Levels.info,
                    ).log()

                if (
                    identified_pct >= analysis_config.IDENTIFY_THRESHOLD
                    and best_delta is not None
                ):
                    shard_best_delta[shard] = best_delta

            # Sort by best delta descending; tie-break by dimension ascending
            result[constant] = sorted(
                shard_best_delta.keys(),
                key=lambda s: (-shard_best_delta[s], s.dim),
            )

        return result

    # ------------------------------------------------------------------
    # Per-shard analysis
    # ------------------------------------------------------------------

    def _analyze_shard(
        self,
        shard: Shard,
        constant: Constant,
        *,
        cmf_id: str,
        shard_id: str,
        encoding_str: str,
        jsonl_path: str,
        seen_trajectories: dict,
    ) -> Tuple[Optional[float], float, bool]:
        """Sample trajectories in *shard* and aggregate Tier-1 stats.

        For every sampled ``(traj, start)`` pair:

        * Derive the deterministic ``trajectory_id`` (cheap — no walk).
        * If the existing JSONL record already carries both ``delta_estimate``
          and ``identified``, reuse the cached values — no handler is built.
        * Otherwise build a ``TrajectoryAttributesHandler`` and a full
          ``TrajectoryDTO`` (this triggers the walk), append the line to the
          JSONL, and use the freshly computed values for aggregation.

        Returns ``(best_delta, identified_pct, sampled)``.  ``sampled`` is
        ``False`` when ``sample_pairs`` itself raised — the caller treats
        that as "skip this shard".
        """
        searcher = SerialSearcher(shard, constant, use_LIReC=False)
        try:
            pairs = searcher.sample_pairs(
                trajectory_generator=analysis_config.NUM_TRAJECTORIES_FROM_DIM,
            )
        except ValueError as e:
            Logger(
                f"Skipping shard {shard_id}: {e}",
                Logger.Levels.warning,
            ).log()
            return None, 0.0, False

        best_delta: Optional[float] = None
        total = 0
        identified_count = 0

        with open(jsonl_path, "a") as fout:
            for traj, start in pairs:
                # Trajectory id derived without any symbolic / walk work.
                start_t = _position_to_tuple(start)
                dir_t = _position_to_tuple(traj)
                tid = _stable_id(
                    shard.cmf_name, encoding_str, str(start_t), str(dir_t),
                )

                cached = seen_trajectories.get(tid)
                if (
                    cached is not None
                    and "delta_estimate" in cached
                    and "identified" in cached
                ):
                    # Reuse — no handler, no walk.
                    delta = float(cached["delta_estimate"])
                    identified = bool(cached["identified"])
                else:
                    try:
                        handler = TrajectoryAttributesHandler.from_cmf(
                            shard.cmf, traj, start,
                        )
                        dto = build_trajectory_dto(
                            handler,
                            cmf_id=cmf_id,
                            shard_id=shard_id,
                            cmf_name=shard.cmf_name,
                            shard_encoding_str=encoding_str,
                            start=start,
                            direction=traj,
                        )
                    except Exception as e:
                        Logger(
                            f"Handler error — shard {shard_id}, "
                            f"traj={traj}, start={start}: {e}",
                            Logger.Levels.warning,
                        ).log()
                        continue

                    delta = float(dto.delta_estimate)
                    identified = bool(dto.identified)
                    line = dto.to_json_line()
                    fout.write(line + "\n")
                    fout.flush()
                    # Track in-run so duplicate (traj, start) pairs don't trigger
                    # another walk within the same execute() call.
                    seen_trajectories[tid] = json.loads(line)

                total += 1
                if identified:
                    identified_count += 1
                if best_delta is None or delta > best_delta:
                    best_delta = delta

        identified_pct = identified_count / total if total else 0.0
        return best_delta, identified_pct, True
