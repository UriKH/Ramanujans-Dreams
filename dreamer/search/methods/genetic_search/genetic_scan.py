"""
Genetic Search — population-based trajectory optimisation in flatland space.

Algorithm is faithful to ``context/resources/code/algos/genetic.py``.
Key differences from the legacy ``GeneticSearchMethod`` (which predates the
DTO/JSONL refactor):

* Genomes are flatland integer vectors (via :class:`FlatlandGeometry`) so that
  in-cone membership is fast and exact.
* Output uses the modern ``worker_pool`` sink / Tier-1 DTO pipeline.
* No GCD reduction of genome vectors — raw integer coords so that ``2*z ± 1``
  refinement and magnitude growth are meaningful (reference behaviour).
* Single-point crossover (reference), ``random.choice(elites)`` parent
  selection, refine_prob 0.7/0.3 child asymmetry.
* Early-stop on unchanged-best generations retained (user decision).
"""

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


class NoInitialPopulation(Exception):
    """Raised when no in-cone seed genome can be found for the shard."""

    def __init__(self, shard_id: str, constant: Constant):
        self.shard_id = shard_id
        self.constant = constant
        super().__init__(
            f"Genetic Search: could not build an initial in-cone population "
            f"for constant '{constant.name}' in shard {shard_id}."
        )


# ---------------------------------------------------------------------------
# Reference operators (translated from resources/code/algos/genetic.py)
# ---------------------------------------------------------------------------

def _crossover(z1: np.ndarray, z2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover over the flatland coordinate vector."""
    d = len(z1)
    if d < 2:
        return z1.copy(), z2.copy()
    point = random.randint(1, d - 1)
    c1 = np.concatenate([z1[:point], z2[point:]])
    c2 = np.concatenate([z2[:point], z1[point:]])
    return c1, c2


def _mutate(
    z: np.ndarray,
    *,
    max_step: int,
    mutation_prob: float,
    refine_prob: float,
    refine_coord_prob: float,
) -> np.ndarray:
    """Mutate a flatland genome vector.

    Faithful to ``mutate_position`` in the reference: no GCD reduction.

    * Refine mode (prob ``refine_prob``): ``z' = 2*z``, then per-coord add ±1
      with ``refine_coord_prob``; guarantee ≥ 1 coord changes.
    * Coarse mode (otherwise): per-coord add ``randint(-max_step, max_step)``
      with ``mutation_prob``.
    """
    if random.random() < refine_prob:
        new_z = 2 * z.copy()
        changed = False
        for i in range(len(new_z)):
            if random.random() < refine_coord_prob:
                new_z[i] += random.choice([-1, 1])
                changed = True
        if not changed and len(new_z) > 0:
            idx = random.randrange(len(new_z))
            new_z[idx] += random.choice([-1, 1])
        return new_z

    new_z = z.copy()
    for i in range(len(new_z)):
        if random.random() < mutation_prob:
            new_z[i] += random.randint(-max_step, max_step)
    return new_z


# ---------------------------------------------------------------------------
# GeneticSearch
# ---------------------------------------------------------------------------

class GeneticSearch(SearchMethod):
    """Population-based search over flatland trajectory directions, single constant."""

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
        """Run the GA for a single constant, emitting DTOs to *sink*.

        :raises NoInitialPopulation: if no in-cone seed genome can be built.
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

        # Resolve GA schedule (callable or int).
        dim = geom.d_flat
        generations = (
            search_config.GA_GENERATIONS(dim)
            if callable(search_config.GA_GENERATIONS)
            else search_config.GA_GENERATIONS
        )
        pop_size = (
            search_config.GA_POPULATION_SIZE(dim)
            if callable(search_config.GA_POPULATION_SIZE)
            else search_config.GA_POPULATION_SIZE
        )
        pop_size = max(pop_size, 2)

        population = self._init_population(geom, pop_size, shard_id, constant)

        # Evaluate initial population; delta=None entries are filled below.
        deltas = [self._eval_genome(z, eval_ctx) for z in population]

        last_best = None
        unchanged_count = 0
        max_unchanged = max(int(0.1 * generations), 5)

        for _ in range(generations):
            # Sort by fitness descending.
            ranked = sorted(zip(deltas, population), key=lambda p: p[0], reverse=True)
            deltas, population = zip(*ranked) if ranked else ([], [])
            deltas, population = list(deltas), list(population)

            best_delta = deltas[0]
            if last_best is not None and abs(best_delta - last_best) < 1e-7:
                unchanged_count += 1
            else:
                unchanged_count = 0
            last_best = best_delta

            if unchanged_count >= max_unchanged:
                Logger(
                    f"Genetic Search: early stop after {unchanged_count} "
                    f"unchanged generations (shard {shard_id})",
                    Logger.Levels.debug,
                ).log()
                break

            # Elitism.
            elite_count = max(1, int(search_config.GA_ELITE_FRACTION * pop_size))
            elites = population[:elite_count]
            elite_deltas = deltas[:elite_count]

            next_pop: List[np.ndarray] = list(elites)
            next_deltas: List[float] = list(elite_deltas)

            # Reproduction until full.
            while len(next_pop) < pop_size:
                p1 = random.choice(elites)
                p2 = random.choice(elites)

                # Single-point crossover (reference child asymmetry).
                if random.random() < search_config.GA_CROSSOVER_PROB:
                    c1, c2 = _crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Child-1: higher refine_prob (0.7 per reference).
                c1 = _mutate(
                    c1,
                    max_step=search_config.GA_MUTATION_STEP,
                    mutation_prob=search_config.GA_MUTATION_PROB,
                    refine_prob=0.7,
                    refine_coord_prob=search_config.GA_REFINE_COORD_PROB,
                )
                c1 = self._repair(c1, geom)
                next_pop.append(c1)
                next_deltas.append(None)

                if len(next_pop) < pop_size:
                    # Child-2: lower refine_prob (0.3 per reference).
                    c2 = _mutate(
                        c2,
                        max_step=search_config.GA_MUTATION_STEP,
                        mutation_prob=search_config.GA_MUTATION_PROB,
                        refine_prob=0.3,
                        refine_coord_prob=search_config.GA_REFINE_COORD_PROB,
                    )
                    c2 = self._repair(c2, geom)
                    next_pop.append(c2)
                    next_deltas.append(None)

            population = next_pop
            # Evaluate new children (elites already have cached deltas).
            for i in range(len(population)):
                if next_deltas[i] is None:
                    next_deltas[i] = self._eval_genome(population[i], eval_ctx)
            deltas = next_deltas

        # Final evaluation pass.
        for i, z in enumerate(population):
            if deltas[i] is None:
                deltas[i] = self._eval_genome(z, eval_ctx)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _init_population(
        self,
        geom: FlatlandGeometry,
        pop_size: int,
        shard_id: str,
        constant: Constant,
    ) -> List[np.ndarray]:
        """Build an initial population of in-cone flatland genomes."""
        orchestrator = ShardSamplingOrchestrator(self.space)
        # Oversample: request more than pop_size to account for out-of-cone.
        n_sample = max(pop_size * 3, 10)
        trajectories = orchestrator.sample_trajectories(n_sample)

        population: List[np.ndarray] = []
        for t in trajectories:
            z = geom.to_flatland(t)
            if not np.any(z):
                continue
            if geom.is_inside(z):
                population.append(z)
            if len(population) >= pop_size:
                break

        # Top up with small random perturbations of found seeds when short.
        attempts = 0
        while len(population) < pop_size and attempts < pop_size * 10:
            attempts += 1
            if not population:
                break
            base = random.choice(population).copy()
            cand = _mutate(
                base,
                max_step=1,
                mutation_prob=0.5,
                refine_prob=0.0,
                refine_coord_prob=0.0,
            )
            if np.any(cand) and geom.is_inside(cand):
                population.append(cand)

        if not population:
            raise NoInitialPopulation(shard_id, constant)

        # Pad with copies if still short (will be repaired/mutated next gen).
        while len(population) < pop_size:
            population.append(random.choice(population).copy())

        return population[:pop_size]

    def _repair(self, z: np.ndarray, geom: FlatlandGeometry) -> np.ndarray:
        """Return *z* if in-cone, else resample a valid genome from the shard."""
        if geom.is_inside(z):
            return z
        orchestrator = ShardSamplingOrchestrator(self.space)
        for _ in range(search_config.GA_MAX_RETRIES * 3):
            trajectories = orchestrator.sample_trajectories(5)
            for t in trajectories:
                cand = geom.to_flatland(t)
                if np.any(cand) and geom.is_inside(cand):
                    return cand
        # Fall back to the zero-ish original; downstream eval will handle.
        return z

    def _eval_genome(self, z: np.ndarray, eval_ctx: dict) -> float:
        """Evaluate a genome, returning δ (−∞ if invalid)."""
        delta, _ = evaluate_in_flatland(z, **eval_ctx)
        return delta
