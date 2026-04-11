import random
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, cast

from dreamer.extraction.samplers import ShardSamplingOrchestrator
from dreamer.extraction.shard import Shard
from dreamer.utils.logger import Logger
from dreamer.utils.multi_processing import create_pool
from dreamer.utils.schemes.searcher_scheme import SearchMethod
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector
from dreamer.configs import search_config
from ramanujantools import Position


INVALID_DELTA = -2.0


def _delta_from_search_data(sd: Optional[SearchData]) -> float:
    if sd is None or sd.delta is None or isinstance(sd.delta, str):
        return INVALID_DELTA
    try:
        return float(sd.delta)
    except (TypeError, ValueError):
        return INVALID_DELTA


def _to_position(data: Dict[Any, Any]) -> Position:
    return Position(list(data.items()))

def _crossover_positions(parent1: Position, parent2: Position) -> Tuple[Position, Position]:
    keys = sorted(set(parent1.keys()) | set(parent2.keys()), key=str)
    if len(keys) < 2:
        return parent1, parent2

    point = random.randint(1, len(keys) - 1)
    child1_dict: Dict[Any, Any] = {}
    child2_dict: Dict[Any, Any] = {}
    for i, key in enumerate(keys):
        if i < point:
            child1_dict[key] = parent1.get(key, 0)
            child2_dict[key] = parent2.get(key, 0)
        else:
            child1_dict[key] = parent2.get(key, 0)
            child2_dict[key] = parent1.get(key, 0)

    return _to_position(child1_dict), _to_position(child2_dict)


def _mutate_position(
    pos: Position,
    *,
    max_step: int,
    mutation_prob: float,
    refine_prob: float,
    refine_coord_prob: float,
) -> Position:
    if random.random() < refine_prob:
        new_pos = 2 * pos
        changed = False
        for key in list(new_pos.keys()):
            if random.random() < refine_coord_prob:
                new_pos[key] = new_pos[key] + random.choice([-1, 1])
                changed = True
        if not changed and new_pos:
            key = random.choice(list(new_pos.keys()))
            new_pos[key] = new_pos[key] + random.choice([-1, 1])
        return new_pos

    new_pos = _to_position(dict(pos))
    for key in list(new_pos.keys()):
        if random.random() < mutation_prob:
            new_pos[key] = new_pos[key] + random.randint(-max_step, max_step)
    return new_pos


class GeneticSearchMethod(SearchMethod):
    """Genetic trajectory search over a searchable space."""

    def __init__(
        self,
        space: Shard,
        constant,
        generations: int = 25,
        pop_size: int = 40,
        max_coord_init: int = 10,
        elite_fraction: float = 0.2,
        mutation_prob: float = 0.3,
        mutation_step: int = 1,
        crossover_prob: float = 0.5,
        max_retries: int = 3,
        refine_prob: float = 0.5,
        refine_coord_prob: float = 0.5,
        parallel_eval: bool = True,
        data_manager: DataManager = None,
        share_data: bool = True,
        use_LIReC: bool = True,
    ):
        """Initialize GA hyperparameters and backing storage."""
        super().__init__(space, constant, use_LIReC, data_manager, share_data)
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if generations < 1:
            raise ValueError("generations must be at least 1")
        if max_coord_init < 1:
            raise ValueError("max_coord_init must be at least 1")

        self.generations = generations
        self.pop_size = pop_size
        self.max_coord_init = max_coord_init
        self.elite_fraction = elite_fraction
        self.mutation_prob = mutation_prob
        self.mutation_step = mutation_step
        self.crossover_prob = crossover_prob
        self.max_retries = max_retries
        self.refine_prob = refine_prob
        self.refine_coord_prob = refine_coord_prob
        self.parallel_eval = parallel_eval
        self.space = cast(Shard, self.space)

        if self.data_manager is None:
            self.data_manager = DataManager(use_LIReC=self.use_LIReC)

    def _resolve_start(self, starts: Optional[Position | List[Position]]) -> Position:
        if starts is None:
            start_point = self.space.get_interior_point()
        elif isinstance(starts, list):
            start_point = starts[0] if starts else None
        else:
            start_point = starts

        if start_point is None:
            raise ValueError("Genetic search requires a valid start point or left as None")
        return start_point

    def _resolve_template(self, provided: Optional[Position]) -> Position:
        if provided is not None:
            return provided

        sample_set = ShardSamplingOrchestrator(self.space).sample_trajectories(lambda dim: max(10, 2 * dim))
        return next(iter(sample_set))

    def _sample_valid_trajectories(self, *, count: int, template_pos: Position) -> List[Position]:
        if count <= 0:
            return []

        sampled: List[Position] = []
        max_sampling_rounds = max(3, self.max_retries + 1)

        try:
            sampler = ShardSamplingOrchestrator(self.space)
        except Exception:
            sampler = None

        if sampler is None:
            Logger("No sampler available!!!").log()

        for _ in range(max_sampling_rounds):
            if len(sampled) >= count:
                break

            if sampler is not None:
                needed = count - len(sampled)
                candidates = list(
                    sampler.sample_trajectories(lambda dim: max(needed, 2 * dim), exact=True)
                )

                random.shuffle(candidates)
                for traj in candidates:
                    if self.space.is_valid_trajectory(traj):
                        sampled.append(traj)
                        if len(sampled) >= count:
                            break

        if len(sampled) < count:
            raise ValueError("Genetic search could not sample enough valid trajectories satisfying A v <= 0")
        return sampled[:count]

    def _repair_trajectory(self, trajectory: Position, template_pos: Position) -> Position:
        if self.space.is_valid_trajectory(trajectory):
            return trajectory
        return self._sample_valid_trajectories(count=1, template_pos=template_pos)[0]

    def _compute_missing_search_data(self, pairs: List[Tuple[Position, Position]]) -> None:
        missing_pairs = [pair for pair in pairs if SearchVector(pair[1], pair[0]) not in self.data_manager]
        if not missing_pairs:
            return

        traj_list = [traj for traj, _ in missing_pairs]
        start_list = [start for _, start in missing_pairs]

        if self.parallel_eval and len(missing_pairs) > 1:
            with create_pool() as pool:
                results = list(
                    pool.map(
                        partial(self.space.compute_trajectory_data, use_LIReC=self.use_LIReC),
                        traj_list,
                        start_list,
                        chunksize=search_config.SEARCH_VECTOR_CHUNK,
                    )
                )
        else:
            results = [
                self.space.compute_trajectory_data(traj, start, use_LIReC=self.use_LIReC)
                for traj, start in missing_pairs
            ]

        for sd in results:
            if sd is not None:
                self.data_manager[sd.sv] = sd

    def _evaluate_population(
        self,
        population: List[Dict[str, Any]],
        *,
        start: Position,
        template_pos: Position,
    ) -> List[Dict[str, Any]]:
        for ind in population:
            ind["trajectory"] = self._repair_trajectory(ind["trajectory"], template_pos)

        to_eval_indices = [i for i, ind in enumerate(population) if ind["delta"] is None]
        if not to_eval_indices:
            return population

        pairs = [(population[i]["trajectory"], start) for i in to_eval_indices]
        self._compute_missing_search_data(pairs)

        for i in to_eval_indices:
            traj = population[i]["trajectory"]
            sv = SearchVector(start, traj)
            sd = self.data_manager.get(sv)
            delta = _delta_from_search_data(sd)

            if delta == INVALID_DELTA:
                for _ in range(self.max_retries):
                    new_traj = self._sample_valid_trajectories(count=1, template_pos=template_pos)[0]
                    new_sd = self.space.compute_trajectory_data(new_traj, start, use_LIReC=self.use_LIReC)
                    if new_sd is not None:
                        self.data_manager[new_sd.sv] = new_sd
                    delta = _delta_from_search_data(new_sd)
                    if delta != INVALID_DELTA:
                        population[i]["trajectory"] = new_traj
                        population[i]["delta"] = delta
                        population[i]["sd"] = new_sd
                        break
                else:
                    population[i]["delta"] = INVALID_DELTA
                    population[i]["sd"] = sd
            else:
                population[i]["delta"] = delta
                population[i]["sd"] = sd

        return population

    def search(
        self,
        starts: Optional[Position | List[Position]] = None,
        *,
        template_trajectory: Optional[Position] = None,
    ) -> DataManager:
        """Perform GA search and return the data manager with evaluated trajectories."""
        start_point = self._resolve_start(starts)
        template = self._resolve_template(template_trajectory)

        initial_trajectories = self._sample_valid_trajectories(count=self.pop_size, template_pos=template)
        population: List[Dict[str, Any]] = [
            {"trajectory": traj, "delta": None, "sd": None} for traj in initial_trajectories
        ]

        for gen in range(self.generations):
            population = self._evaluate_population(population, start=start_point, template_pos=template)
            population.sort(key=lambda ind: ind["delta"], reverse=True)

            best_delta = population[0]["delta"]
            Logger(f"GA: generation={gen}, best_delta={best_delta}", Logger.Levels.debug).log()

            elite_count = max(1, int(self.elite_fraction * self.pop_size))
            elites = population[:elite_count]
            next_population: List[Dict[str, Any]] = [
                {"trajectory": ind["trajectory"], "delta": ind["delta"], "sd": ind["sd"]}
                for ind in elites
            ]

            while len(next_population) < self.pop_size:
                parent1 = random.choice(elites)
                parent2 = random.choice(elites)

                if random.random() < self.crossover_prob:
                    child1, child2 = _crossover_positions(parent1["trajectory"], parent2["trajectory"])
                else:
                    child1 = _to_position(dict(parent1["trajectory"]))
                    child2 = _to_position(dict(parent2["trajectory"]))

                if random.random() < self.mutation_prob:
                    child1 = _mutate_position(
                        child1,
                        max_step=self.mutation_step,
                        mutation_prob=self.mutation_prob,
                        refine_prob=self.refine_prob,
                        refine_coord_prob=self.refine_coord_prob,
                    )
                child1 = self._repair_trajectory(child1, template)
                next_population.append({"trajectory": child1, "delta": None, "sd": None})

                if len(next_population) < self.pop_size:
                    if random.random() < self.mutation_prob:
                        child2 = _mutate_position(
                            child2,
                            max_step=self.mutation_step,
                            mutation_prob=self.mutation_prob,
                            refine_prob=self.refine_prob,
                            refine_coord_prob=self.refine_coord_prob,
                        )
                    child2 = self._repair_trajectory(child2, template)
                    next_population.append({"trajectory": child2, "delta": None, "sd": None})

            population = next_population

        population = self._evaluate_population(population, start=start_point, template_pos=template)
        population.sort(key=lambda ind: ind["delta"], reverse=True)
        # Logger(f"GA: Search complete. Best Delta Found: {population[0]['delta']}", Logger.Levels.info).log()
        return self.data_manager
