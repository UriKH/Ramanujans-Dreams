"""
Shared flatland trajectory evaluator.

Provides :func:`evaluate_in_flatland` — the three-case (A/B/C) walk-reuse
logic shared by SmallAngleSearch, GeneticSearch, and SimulatedAnnealingSearch.
All three methods evaluate a flatland integer vector ``z``, emit a Tier-1 DTO
to a ``sink`` callable, and return ``(delta, identified)`` for one constant.
"""

import dataclasses
from typing import Callable, Dict, Optional, Tuple

from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.storage.attribute_registry import attribute_name
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_trajectory_id,
)
from dreamer.configs import config

search_config = config.search


def evaluate_in_flatland(
    z,
    *,
    geom: FlatlandGeometry,
    shard: Shard,
    start,
    constant: Constant,
    cmf_id: str,
    shard_id: str,
    shard_encoding_str: str,
    sink: Callable,
    seen_trajectories: dict,
    handler_cache: Dict[str, "TrajectoryAttributesHandler"],
) -> Tuple[float, bool]:
    """Compute δ/identified for *constant* at flatland direction *z*, emitting a DTO.

    Returns ``(delta, identified)`` for *constant*.  Three cases, each cheaper
    than the next:

    **Case A — delta already cached (on-disk or in-memory):**
    ``seen_trajectories`` contains a record whose ``delta_estimate`` already
    includes this constant → return immediately, no handler built, no walk.

    **Case B — handler cached (another constant evaluated this trajectory this
    run):**
    A :class:`TrajectoryAttributesHandler` for this trajectory_id is in
    *handler_cache* → call ``compute_for_constant`` only; build a merged DTO
    and emit it.

    **Case C — new trajectory:**
    Build handler from scratch, full walk, emit Tier-1 DTO.

    In all cases the handler (if available) is stored in *handler_cache* for
    future same-shard cross-constant reuse.
    """
    direction = geom.to_real(z)
    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)
    trajectory_id = derive_trajectory_id(
        shard_id, shard.cmf_name, shard_encoding_str, start_t, dir_t
    )

    desired = {attribute_name(s) for s in search_config.TIER2_ATTRIBUTES}
    seen_record = seen_trajectories.get(trajectory_id)

    # --- Case A: delta already known for this constant ---
    if seen_record is not None:
        delta_map = seen_record.get("delta_estimate") or {}
        if constant.name in delta_map:
            ided_map = seen_record.get("identified") or {}
            return float(delta_map[constant.name]), bool(ided_map.get(constant.name, False))

    # --- Case B: handler cached — reuse walk, only compute new constant ---
    cached_handler = handler_cache.get(trajectory_id)
    if cached_handler is not None:
        try:
            new_dto = build_trajectory_dto(
                cached_handler,
                cmf_id=cmf_id,
                shard_id=shard_id,
                cmf_name=shard.cmf_name,
                shard_encoding_str=shard_encoding_str,
                start=start,
                direction=direction,
                constants=[constant],
            )
            existing_delta = dict(seen_record.get("delta_estimate") or {}) if seen_record else {}
            existing_ided = dict(seen_record.get("identified") or {}) if seen_record else {}
            existing_p = dict(seen_record.get("p_vector") or {}) if seen_record else {}
            existing_q = dict(seen_record.get("q_vector") or {}) if seen_record else {}
            merged_dto = dataclasses.replace(
                new_dto,
                delta_estimate={**existing_delta, **new_dto.delta_estimate},
                identified={**existing_ided, **new_dto.identified},
                p_vector={**existing_p, **(new_dto.p_vector or {})},
                q_vector={**existing_q, **(new_dto.q_vector or {})},
            )
        except Exception as exc:
            Logger(
                f"Flatland evaluator handler-cache error — shard {shard_id}, "
                f"constant={constant.name}: {exc}",
                Logger.Levels.warning,
            ).log()
            return float("-inf"), False

        sink((cached_handler.trajectory_matrix(), constant.value_sympy, merged_dto))
        seen_trajectories[trajectory_id] = {
            "extended_metrics": dict.fromkeys(desired),
            "delta_estimate": dict(merged_dto.delta_estimate),
            "identified": dict(merged_dto.identified),
        }
        delta = float(merged_dto.delta_estimate.get(constant.name, float("-inf")))
        identified = bool(merged_dto.identified.get(constant.name, False))
        return delta, identified

    # --- Case C: new trajectory — full walk ---
    try:
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
    except Exception as exc:
        Logger(
            f"Flatland evaluator handler error — shard {shard_id}, "
            f"direction={direction}: {exc}",
            Logger.Levels.warning,
        ).log()
        return float("-inf"), False

    handler_cache[trajectory_id] = handler
    sink((handler.trajectory_matrix(), constant.value_sympy, dto))
    seen_trajectories[trajectory_id] = {
        "extended_metrics": dict.fromkeys(desired),
        "delta_estimate": dict(dto.delta_estimate),
        "identified": dict(dto.identified),
    }

    delta = float(dto.delta_estimate.get(constant.name, float("-inf")))
    identified = bool(dto.identified.get(constant.name, False))
    return delta, identified
