"""
Simulated Annealing Search — trajectory optimisation with Metropolis acceptance.

Algorithm is faithful to ``context/resources/code/algos/annealing.py`` and
``positions.py``.  Key design choices:

* Genomes are flatland integer vectors (via :class:`FlatlandGeometry`).
* Neighbours = raw (non-reduced) ±1 unit steps in flatland, filtered by
  ``geom.is_inside`` and excluding the tabu list.
* Temperature decreases only on accepted moves (reference semantics).
* On a rejected step the current genome is doubled (length-doubling, no GCD
  reduce); a doubling counter is incremented; on exceeding
  ``ANNEAL_MAX_DOUBLINGS`` the counter is reset and a fresh seed direction is
  drawn (fixing the reference's dead "give up" branch).
* Output uses the modern ``worker_pool`` sink / Tier-1 DTO pipeline.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ramanujantools import Position

from dreamer.configs import config
from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.search.methods.flatland.evaluator import evaluate_in_flatland
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.utils.constants.constant import Constant
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

search_config = config.search


class NoInitialIdentification(Exception):
    """Raised when no reservoir trajectory identifies the constant in a shard."""

    def __init__(self, shard_id: str, constant: Constant):
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Simulated Annealing Search: no initial trajectory identified "
            f"'{constant.name}' in shard {shard_id}."
        )


def _get_temp(T0: float, k: int, schedule: str) -> float:
    """Cooling schedule (reference ``get_temp``)."""
    if schedule == "log":
        return T0 / math.log(k + 1) if k > 0 else T0
    return T0 / (k + 1)


class SimulatedAnnealingSearch(SearchMethod):
    """Simulated annealing search over flatland trajectory directions, single constant."""

    def __init__(
        self,
        space: Shard,
        constant: Constant,
        use_LIReC: bool = True,
    ):
        """
        :param space: The shard to search in.
        :param constant: The (single) constant this search optimises δ for.
        :param use_LIReC: Use LIReC to identify constants within the shard.
        """
        super().__init__(space, constant, use_LIReC)
        self.constant = constant

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def search(self, starts=None):
        """Standalone entry point — collect emitted DTOs into a list."""
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
        """Run SA for a single constant, emitting DTOs to *sink*.

        :raises NoInitialIdentification: if no reservoir seed identifies *constant*.
        """
        if handler_cache is None:
            handler_cache = {}

        shard: Shard = self.space
        geom = FlatlandGeometry(shard)
        start = shard.get_interior_point()

        eval_ctx = dict(
            geom=geom,
            shard=shard,
            start=start,
            constant=constant,
            cmf_id=cmf_id,
            shard_id=shard_id,
            shard_encoding_str=shard_encoding_str,
            sink=sink,
            seen_trajectories=seen_trajectories,
            handler_cache=handler_cache,
        )

        cur_z = self._select_seed(geom, eval_ctx, shard_id, constant)
        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
        best_delta = cur_delta

        T0 = search_config.ANNEAL_T0
        Tmin = search_config.ANNEAL_TMIN
        schedule = search_config.ANNEAL_SCHEDULE
        max_iters = search_config.ANNEAL_MAX_ITERS
        max_doublings = search_config.ANNEAL_MAX_DOUBLINGS
        tabu_size = search_config.ANNEAL_TABU_SIZE

        T = T0
        iter_left = max_iters
        doubling_count = 0

        # Tabu: bounded recent-position list (reference update_old_list_neighs).
        old_pos_list: List[bytes] = [cur_z.tobytes()]

        while iter_left > 0 and T > Tmin:
            # Generate in-cone, non-tabu neighbours (raw ±1, no GCD reduction).
            neighbours: List[np.ndarray] = []
            for cand in geom.perturbations(cur_z, reduce=False):
                if not geom.is_inside(cand):
                    continue
                if cand.tobytes() in old_pos_list:
                    continue
                neighbours.append(cand)

            if not neighbours:
                # No valid neighbour: double and continue (reference adaptive scaling).
                if doubling_count >= max_doublings:
                    doubling_count = 0
                    fresh = self._try_reseed(geom, eval_ctx, shard_id, constant)
                    if fresh is not None:
                        cur_z = fresh
                        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
                else:
                    cur_z = cur_z * 2  # no GCD reduce
                    doubling_count += 1
                continue

            # Evaluate all neighbours; pick best (reference picks best of batch).
            neighbour_deltas: List[Tuple[float, np.ndarray]] = []
            for nb in neighbours:
                d, _ = evaluate_in_flatland(nb, **eval_ctx)
                neighbour_deltas.append((d, nb))
            neighbour_deltas.sort(key=lambda x: x[0], reverse=True)
            new_delta, new_z = neighbour_deltas[0]

            # Update tabu list with neighbours + current (reference update_old_list_neighs).
            for _, nb in neighbour_deltas:
                old_pos_list.append(nb.tobytes())
            old_pos_list.append(cur_z.tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]

            accepted = False
            if new_delta >= cur_delta:
                cur_z = new_z
                cur_delta = new_delta
                accepted = True
                iter_left -= 1
            else:
                diff = new_delta - cur_delta
                if random.random() < math.exp(diff / T):
                    cur_z = new_z
                    cur_delta = new_delta
                    accepted = True
                    iter_left -= 1

            if accepted:
                doubling_count = 0
                if cur_delta > best_delta:
                    best_delta = cur_delta
                # Temperature decreases only on accepted moves (reference).
                T = _get_temp(T0, max_iters - iter_left, schedule)
            else:
                # Adaptive scaling: double on rejection (reference).
                if doubling_count >= max_doublings:
                    doubling_count = 0
                    fresh = self._try_reseed(geom, eval_ctx, shard_id, constant)
                    if fresh is not None:
                        cur_z = fresh
                        cur_delta, _ = evaluate_in_flatland(cur_z, **eval_ctx)
                else:
                    cur_z = cur_z * 2  # no GCD reduce
                    doubling_count += 1

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_seed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
    ) -> np.ndarray:
        """Pick the first reservoir trajectory (ascending L2 norm) that identifies."""
        trajectories = ShardSamplingOrchestrator(self.space).sample_trajectories(
            search_config.ANNEAL_RESERVOIR_SIZE
        )
        candidates: List[Tuple[float, Position]] = []
        for t in trajectories:
            norm = float(np.linalg.norm([float(t[s]) for s in geom.symbols]))
            candidates.append((norm, t))
        candidates.sort(key=lambda pair: pair[0])

        for _, t in candidates:
            z = geom.to_flatland(t)
            if not np.any(z):
                continue
            _, identified = evaluate_in_flatland(z, **eval_ctx)
            if identified:
                return z

        raise NoInitialIdentification(shard_id, constant)

    def _try_reseed(
        self,
        geom: FlatlandGeometry,
        eval_ctx: dict,
        shard_id: str,
        constant: Constant,
    ) -> Optional[np.ndarray]:
        """Attempt to find a fresh seed when the doubling budget is exhausted."""
        try:
            return self._select_seed(geom, eval_ctx, shard_id, constant)
        except NoInitialIdentification:
            return None
