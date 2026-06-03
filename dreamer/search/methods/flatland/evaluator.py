"""
Shared flatland trajectory evaluator.

Provides :func:`evaluate_in_flatland` — the three-case (A/B/C) walk-reuse
logic shared by SmallAngleSearch, GeneticSearch, and SimulatedAnnealingSearch.
All three methods evaluate a flatland integer vector ``z``, emit a Tier-1 DTO
to a ``sink`` callable, and return ``(delta, identified)`` for one constant.
"""

import dataclasses
import threading
from contextlib import nullcontext
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
    tier1_config_fingerprint,
    walk_depth_for,
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
    lock: Optional[threading.Lock] = None,
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

    Thread-safety
    -------------
    The parallel-neighbour evaluators (Simulated Annealing, Genetic) call this
    from a :class:`~concurrent.futures.ThreadPoolExecutor`, so the shared
    ``seen_trajectories`` and ``handler_cache`` dicts are read/written
    concurrently.  Pass a :class:`threading.Lock` via *lock* to make the
    read-snapshot and the cache updates atomic — without it, concurrent callers
    can lose updates to ``seen_trajectories``.  The expensive walk
    (``from_cmf`` / ``compute_for_constant``) runs **outside** the lock so any
    real parallelism is preserved.  Two threads racing on the *same* unseen
    ``trajectory_id`` may both walk it (duplicate work + duplicate JSONL line);
    that is harmless because :func:`load_seen_trajectories` merges records by id
    on read.  Single-threaded callers pass ``lock=None`` (a no-op context).
    """
    guard = lock if lock is not None else nullcontext()
    # Always walk the GCD-reduced (primitive) ray: δ depends on the direction's
    # angle, not its length, so scaled/doubled copies of ``z`` map to the same
    # ray — same ``trajectory_id`` — and reuse the cached walk (Case A/B).
    direction = geom.to_real_primitive(z)
    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)

    trajectory_id = derive_trajectory_id(
        shard_id, shard.cmf_name, shard_encoding_str, start_t, dir_t
    )

    desired = {attribute_name(s) for s in search_config.TIER2_ATTRIBUTES}
    with guard:
        seen_record = seen_trajectories.get(trajectory_id)
        cached_handler = handler_cache.get(trajectory_id)

    # Fingerprint of the config knobs that influence this trajectory's Tier-1
    # values, including the walk depth it will use.  A cached record is only
    # reusable when its stored fingerprint matches — otherwise the config (e.g.
    # walk depth / walk type / identification tolerances) changed and the stored
    # δ / identification are stale and must be recomputed below.
    current_fp = tier1_config_fingerprint(walk_depth_for(shard.cmf, direction))

    # --- Case A: delta already known for this constant (same config) ---
    if seen_record is not None and seen_record.get("config_fingerprint") == current_fp:
        delta_map = seen_record.get("delta_estimate") or {}
        if constant.name in delta_map:
            ided_map = seen_record.get("identified") or {}
            return float(delta_map[constant.name]), bool(ided_map.get(constant.name, False))

    # --- Case B: handler cached — reuse walk, only compute new constant ---
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
            # Only fold in previously-stored per-constant data when it was
            # computed under the *same* config — a stale record's δ/identified
            # must not leak into the freshly-recomputed merged DTO.
            fresh = seen_record if (seen_record and seen_record.get("config_fingerprint") == current_fp) else {}
            existing_delta = dict(fresh.get("delta_estimate") or {})
            existing_ided = dict(fresh.get("identified") or {})
            existing_p = dict(fresh.get("p_vector") or {})
            existing_q = dict(fresh.get("q_vector") or {})
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
        with guard:
            seen_trajectories[trajectory_id] = {
                "extended_metrics": dict.fromkeys(desired),
                "delta_estimate": dict(merged_dto.delta_estimate),
                "identified": dict(merged_dto.identified),
                "config_fingerprint": current_fp,
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

    sink((handler.trajectory_matrix(), constant.value_sympy, dto))
    with guard:
        handler_cache[trajectory_id] = handler
        seen_trajectories[trajectory_id] = {
            "extended_metrics": dict.fromkeys(desired),
            "delta_estimate": dict(dto.delta_estimate),
            "identified": dict(dto.identified),
            "config_fingerprint": current_fp,
        }

    delta = float(dto.delta_estimate.get(constant.name, float("-inf")))
    identified = bool(dto.identified.get(constant.name, False))
    return delta, identified
