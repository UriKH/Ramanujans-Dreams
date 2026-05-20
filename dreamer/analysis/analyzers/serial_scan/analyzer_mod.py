"""
AnalyzerModV1 — analysis-stage module.

For each constant, samples trajectories in each shard using
``TrajectoryAttributesHandler`` (Tier-1: delta + identified), filters and
ranks shards by best observed delta, and writes a JSONL audit record per
shard.

Cross-run dedup: before sampling, the per-constant JSONL is loaded and
keyed by ``shard_id``.  Any shard whose ``shard_id`` is already present
reuses the cached ``best_delta`` and ``identified_pct`` — no resampling
or trajectory walks happen for that shard.

The old ``Analyzer`` / ``Analyzer.prioritize()`` path is intentionally not
called here; it is preserved in ``serial_scan_analyzer.py`` for future use.
"""

import json
import os
from typing import Dict, List

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
    derive_cmf_and_shard_ids,
)
from dreamer.utils.storage.dtos import ShardDTO
from dreamer.utils.multi_processing import load_seen_shards
from dreamer.search.methods.hedgehog_scan import SerialSearcher

analysis_config = config.analysis


class AnalyzerModV1(AnalyzerModScheme):
    """Analysis module: filters and ranks shards by Tier-1 trajectory attributes.

    For each shard, ``TrajectoryAttributesHandler`` computes delta and
    ``identified()`` (currently a stub that returns ``True``) for every
    sampled trajectory.  Shards passing the identified-percentage threshold
    are sorted by best delta and returned for deeper search.

    A per-constant JSONL file is appended under
    ``sys_config.EXPORT_ANALYSIS_RESULTS/shards/`` with one record per
    shard.  On subsequent runs, shards already represented in this file
    are skipped (cached values reused).
    """

    def __init__(self, cmf_data: Dict[Constant, List[Searchable]]):
        """
        :param cmf_data: Mapping from each constant to its list of shards.
        """
        super().__init__(
            cmf_data,
            desc='Analysis module — handler-based shard filtering and prioritization',
            version='2',
        )

    @CatchErrorInModule(with_trace=sys_config.MODULE_ERROR_SHOW_TRACE, fatal=True)
    def execute(self) -> Dict[Constant, List[Searchable]]:
        """Filter and rank shards for every constant.

        Returns a mapping from constant → shards sorted by best delta
        (descending), then by dimension (ascending, as a tie-breaker).
        """
        out_root = os.path.join(sys_config.EXPORT_ANALYSIS_RESULTS, "shards")
        os.makedirs(out_root, exist_ok=True)

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

            output_path = os.path.join(out_root, f"{constant.name}.jsonl")
            # Cross-run cache: shard_id → previous analysis record.
            cached_shards = load_seen_shards(output_path)
            # shard → best delta for passing shards only
            shard_best_delta: Dict[Shard, float] = {}

            with open(output_path, "a") as fout:
                for shard in shards:
                    cmf_id, shard_id, encoding_str = derive_cmf_and_shard_ids(shard)

                    cached = cached_shards.get(shard_id)
                    if cached is not None:
                        # Shard already analyzed for this constant — reuse.
                        best_delta = cached.get("best_delta")
                        identified_pct = cached.get("identified_pct", 0.0)
                        if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
                            bd = f'{best_delta:.4f}' if isinstance(best_delta, (int, float)) else 'N/A'
                            Logger(
                                f"Shard {shard_id[:8]}…  cached  "
                                f"best_delta={bd}  identified={identified_pct * 100:.1f}%",
                                Logger.Levels.info,
                            ).log()
                    else:
                        best_delta, identified_pct, sampled = self._analyze_shard(
                            shard, constant, shard_id,
                        )
                        if not sampled:
                            # _analyze_shard logged a sampling failure.
                            continue
                        self._write_record(
                            fout,
                            shard=shard,
                            constant=constant,
                            shard_id=shard_id,
                            cmf_id=cmf_id,
                            encoding_str=encoding_str,
                            best_delta=best_delta,
                            identified_pct=identified_pct,
                        )

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
    # Helpers
    # ------------------------------------------------------------------

    def _analyze_shard(self, shard: Shard, constant: Constant, shard_id: str):
        """Sample trajectories in *shard* and return ``(best_delta, identified_pct, sampled)``.

        ``sampled`` is ``False`` when ``sample_pairs`` itself raised — the
        caller treats that as "skip this shard entirely" (no record written).
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

        best_delta = None
        total = 0
        identified_count = 0

        for traj, start in pairs:
            try:
                handler = TrajectoryAttributesHandler.from_cmf(
                    shard.cmf, traj, start,
                )
                total += 1
                delta = float(handler.delta())
                if handler.identified():
                    identified_count += 1
                if best_delta is None or delta > best_delta:
                    best_delta = delta
            except Exception as e:
                Logger(
                    f"Handler error — shard {shard_id}, "
                    f"traj={traj}, start={start}: {e}",
                    Logger.Levels.warning,
                ).log()

        identified_pct = identified_count / total if total else 0.0

        if analysis_config.PRINT_FOR_EVERY_SEARCHABLE:
            bd = f'{best_delta:.4f}' if best_delta is not None else 'N/A'
            Logger(
                f"Shard {shard_id[:8]}…  best_delta={bd}  "
                f"identified={identified_pct * 100:.1f}%",
                Logger.Levels.info,
            ).log()

        return best_delta, identified_pct, True

    @staticmethod
    def _write_record(
        fout,
        *,
        shard: Shard,
        constant: Constant,
        shard_id: str,
        cmf_id: str,
        encoding_str: str,
        best_delta,
        identified_pct: float,
    ) -> None:
        """Append one analysis record for *shard* to the open JSONL file."""
        dto = ShardDTO(
            shard_id=shard_id,
            cmf_id=cmf_id,
            # shard_encoding kept empty until ShardDTO schema firms up;
            # the full inequality string is stored as an auxiliary key.
            shard_encoding=(),
            dimensionality=shard.dim,
            found_constants=[constant.name] if identified_pct > 0.0 else [],
        )
        record = json.loads(dto.to_json_line())
        record["shard_encoding_str"] = encoding_str
        record["best_delta"] = best_delta
        record["identified_pct"] = identified_pct
        fout.write(json.dumps(record) + "\n")
        fout.flush()
