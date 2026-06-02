"""
Small Angle Search — per-shard local hill-climb on the trajectory *direction*.

See ``context/SEARCH_ALGORITHMS.md``.  The method:

0. Conditions the shard into flatland space (:class:`FlatlandGeometry`).
1. Draws a small reservoir of candidate trajectories from the shard sampler,
   sorted by ascending L2 norm (start close to the origin), and uses the first
   one that **identifies** the constant as the climb seed.  If none identifies,
   :class:`NoInitialIdentification` is raised (caught by the module).
2. Computes δ for the current direction (attributes are computed in the *real*
   shard space).
3. Probes the ±1 perturbations of each flatland coordinate; keeps those that
   stay inside the shard cone and re-centers on the best δ.
4. If no perturbation stays inside, doubles the trajectory length (no GCD
   reduction) and retries.  Stops at ``SA_MAX_DEPTH`` iterations or after
   ``SA_PATIENCE`` iterations without δ improvement above ``SA_IMPROVE_THRESHOLD``.

This method is iterative/stateful (δ at one step decides the next), so unlike the
hedgehog searcher it computes δ inline via :class:`TrajectoryAttributesHandler`
and pushes one DTO per evaluated trajectory to an injected ``sink`` callable.

Walk-reuse
----------
The expensive step is building ``TrajectoryAttributesHandler`` (the matrix walk).
Once a handler is built its walk matrices are cached internally; calling
``compute_for_constant(B)`` on the same handler is cheap.  :meth:`_evaluate`
therefore accepts an optional *handler_cache* dict (keyed by ``trajectory_id``)
shared across all per-constant climbs for the same shard.  This means:

* Within a single run, if constant-A's climb evaluates trajectory T and constant-
  B's climb later encounters T, the walk is **not** repeated — ``compute_for_constant``
  is called on the already-built handler, and a merged DTO (containing both
  constants' Tier-1 data) is emitted so the JSONL record is complete.

* Across runs (from JSONL): if ``delta_estimate[constant.name]`` is already
  present in the on-disk record it is returned immediately, no handler built.
"""

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.small_angle.flatland import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.attribute_registry import attribute_name
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    _position_to_tuple,
    build_trajectory_dto,
    derive_trajectory_id,
)

search_config = config.search


class NoInitialIdentification(Exception):
    """Raised when no reservoir trajectory identifies the constant in a shard."""

    def __init__(self, shard_id: str, constant: Constant):
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Small Angle Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


class SmallAngleSearch(SearchMethod):
    """Local hill-climb search over trajectory directions, single constant."""

    def __init__(
        self,
        space: Shard,
        constant: Constant,
        use_LIReC: bool = True,
    ):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this climb optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """Standalone entry point: collect emitted DTOs into a list and return them.

        The module drives :meth:`run` directly with the worker-pool sink; this
        wrapper exists for the abstract-method contract and for testing.
        """
        collected: list = []
        self.run(
            constant=self.constant,
            cmf_id="",
            shard_id=getattr(self.space, "cmf_name", "shard"),
            shard_encoding_str=",".join(str(e) for e in self.space.encoding),
            sink=lambda item: collected.append(item),
            seen_trajectories={},
        )
        return collected

    def run(
        self,
        *,
        constant: Constant,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
        handler_cache: Optional[Dict[str, "TrajectoryAttributesHandler"]] = None,
    ) -> None:
        """Run the hill-climb for a single constant, emitting DTOs to *sink*.

        :param handler_cache: Optional dict shared across all per-constant climbs
            for the same shard.  Maps ``trajectory_id → handler`` so that
            trajectories already evaluated (for another constant or in a previous
            step) skip the expensive walk and reuse ``compute_for_constant`` only.
            Callers (``SmallAngleSearchMod._run_shard``) should create one dict per
            shard and pass it to every constant's ``run()`` call.

        :raises NoInitialIdentification: if no reservoir seed identifies *constant*.
        """
        if handler_cache is None:
            handler_cache = {}

        shard: Shard = self.space
        geom = FlatlandGeometry(shard)
        start = shard.get_interior_point()

        # Context shared by every _evaluate call.
        ctx = dict(
            geom=geom,
            start=start,
            constant=constant,
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=shard_encoding_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
            handler_cache=handler_cache,
        )

        z = self._select_seed(geom, ctx, shard_id, constant)

        best_delta, _ = self._evaluate(z, **ctx)
        no_improve = 0
        doublings = 0

        for _ in range(search_config.SA_MAX_DEPTH):
            best_z, best_score = self._best_inside_perturbation(z, ctx)

            if best_z is None:
                # No perturbation stays inside: lengthen the trajectory and retry.
                if doublings >= search_config.SA_MAX_DOUBLINGS:
                    break
                z = z * 2  # no GCD reduction (spec)
                doublings += 1
                continue

            doublings = 0
            z = best_z
            if best_score - best_delta >= search_config.SA_IMPROVE_THRESHOLD:
                best_delta = max(best_delta, best_score)
                no_improve = 0
            else:
                best_delta = max(best_delta, best_score)
                no_improve += 1
                if no_improve >= search_config.SA_PATIENCE:
                    break

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_seed(
        self, geom: FlatlandGeometry, ctx: dict, shard_id: str, constant: Constant
    ) -> np.ndarray:
        """Pick the first reservoir trajectory (ascending L2 norm) that identifies."""
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.SA_RESERVOIR_SIZE
        )
        # Sort by ascending L2 norm so we start as close to the origin as possible.
        candidates: List[Tuple[float, Position]] = []
        for t in trajectories:
            norm = float(np.linalg.norm([float(t[s]) for s in geom.symbols]))
            candidates.append((norm, t))
        candidates.sort(key=lambda pair: pair[0])

        for _, t in candidates:
            z = geom.to_flatland(t)
            if not np.any(z):
                continue
            _, identified = self._evaluate(z, **ctx)
            if identified:
                return z

        raise NoInitialIdentification(shard_id, constant)

    def _best_inside_perturbation(
        self, z: np.ndarray, ctx: dict
    ) -> Tuple[Optional[np.ndarray], float]:
        """Evaluate all in-shard ±1 perturbations; return the best (z, δ)."""
        geom: FlatlandGeometry = ctx["geom"]
        best_z: Optional[np.ndarray] = None
        best_score = float("-inf")
        for cand in geom.perturbations(z):
            if not geom.is_inside(cand):
                continue
            delta, _ = self._evaluate(cand, **ctx)
            if delta > best_score:
                best_score = delta
                best_z = cand
        return best_z, best_score

    def _evaluate(
        self,
        z: np.ndarray,
        *,
        geom: FlatlandGeometry,
        start: Position,
        constant: Constant,
        cmf_id: str,
        shard_id: str,
        shard_encoding_str: str,
        sink: Callable,
        seen_trajectories: dict,
        handler_cache: Dict[str, "TrajectoryAttributesHandler"],
    ) -> Tuple[float, bool]:
        """Compute δ/identified for *constant* and direction *z*, emitting a DTO.

        Returns ``(delta, identified)`` for *constant*.  Three cases, each cheaper
        than the next:

        **Case A — delta already cached (on-disk or in-memory):**
        ``seen_trajectories`` contains a record whose ``delta_estimate`` already
        includes this constant → return immediately, no handler built, no walk.
        This covers both previous-run JSONL records and intra-run re-visits.

        **Case B — handler cached (another constant evaluated this trajectory this
        run):**
        A :class:`TrajectoryAttributesHandler` for this trajectory_id is in
        *handler_cache* → call ``compute_for_constant`` only (walk matrices
        already cached inside the handler); build a merged DTO and emit it.

        **Case C — new trajectory:**
        Build handler from scratch, full walk, emit Tier-1 DTO.

        In all cases the handler (if available) is stored in *handler_cache* for
        future same-shard cross-constant reuse.
        """
        shard: Shard = self.space
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
                # Merge with any already-known constants from the existing record.
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
                    f"Small Angle Search handler-cache error — shard {shard_id}, "
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
                f"Small Angle Search handler error — shard {shard_id}, "
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
