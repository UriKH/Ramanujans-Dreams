from __future__ import annotations
from typing import List, Dict, Any, Tuple
import random
import sympy as sp
from ramanujantools import Position
from .metrics import trajDelta


def get_avg_delta(data):
    deltas = [d.get("delta", 0) for d in data]
    total_sum = sum(deltas)
    count = len(deltas)
    return total_sum / count

import random
import sympy as sp
from ramanujantools import Position

def random_position_like(
    template_pos: Position,
    p: int,
    q: int,
    max_coord: int,
    max_tries: int = 10_000,
) -> Position:
    """
    Random generator that:
      * keeps the same sign pattern as template_pos
      * uses template_pos keys as coordinate system
    No constraint between x_i and y_j.
    """
    # All coordinate symbols come from template_pos
    coords = list(template_pos.keys())
    tmpl = dict(template_pos)

    def sample_with_same_sign(sym):
        s = tmpl.get(sym, 0)
        if s > 0:
            return random.randint(1, max_coord)
        elif s < 0:
            return random.randint(-max_coord, -1)
        else:
            return 0

    for _ in range(max_tries):
        step_dict = {sym: sample_with_same_sign(sym) for sym in coords}
        return Position(step_dict)   # no need for retries now

    raise RuntimeError("Could not generate Position with required constraints")


def random_position(
    p: int,
    q: int,
    max_coord: int,
    max_tries: int = 10_000,
) -> Position:

    x_symbols = sp.symbols(f"x0:{p}") if p > 0 else []
    y_symbols = sp.symbols(f"y0:{q}") if q > 0 else []
    coords = list(x_symbols) + list(y_symbols)

    for _ in range(max_tries):
        # sample each coordinate independently and uniformly
        step = {
            sym: random.randint(-max_coord, max_coord)
            for sym in coords
        }

        # enforce x_i != y_j like before
        x_vals = [step[x] for x in x_symbols]
        y_vals = [step[y] for y in y_symbols]
        if set(x_vals).isdisjoint(y_vals):
            return Position(step)

    raise RuntimeError("Could not generate Position with required constraints")


def crossover_positions(parent1: Position, parent2: Position) -> Tuple[Position, Position]:
    """
    Single-point crossover in the coordinate list.
    """
    keys = sorted(set(parent1.keys()) | set(parent2.keys()), key=str)
    if len(keys) < 2:
        return parent1, parent2

    point = random.randint(1, len(keys) - 1)
    child1_dict = {}
    child2_dict = {}
    for i, k in enumerate(keys):
        if i < point:
            child1_dict[k] = parent1.get(k, 0)
            child2_dict[k] = parent2.get(k, 0)
        else:
            child1_dict[k] = parent2.get(k, 0)
            child2_dict[k] = parent1.get(k, 0)

    return Position(child1_dict), Position(child2_dict)


def mutate_position(
    pos: Position,
    max_step: int,
    mutation_prob: float,
    refine_prob: float = 0.3,          # chance to use the 2*pos ± 1 move
    refine_coord_prob: float = 0.5,    # per-coordinate chance in refine mode
) -> Position:
    """
    Mutation operator using delta(n*pos) = delta(pos).

    - With probability `refine_prob`:
        pos' = 2 * pos, then for each coordinate (with prob `refine_coord_prob`)
        add either +1 or -1. This is the "2*pos ± 1" refinement move.

    - Otherwise:
        For each coordinate, with probability `mutation_prob`,
        add a random integer step in [-max_step, max_step] (coarse move).
    """

    # ----------------------------
    # 1) Refinement mode: 2*pos ± 1
    # ----------------------------
    if random.random() < refine_prob:
        # Start from scaled Position
        new_pos = 2 * pos        # uses Position.__rmul__
        changed = False

        for k in list(new_pos.keys()):
            if random.random() < refine_coord_prob:
                step = random.choice([-1, 1])
                new_pos[k] = new_pos[k] + step
                changed = True

        # Ensure at least one coordinate changes
        if not changed and new_pos:
            k = random.choice(list(new_pos.keys()))
            new_pos[k] = new_pos[k] + random.choice([-1, 1])

        return new_pos

    # ---------------------------------
    # 2) Coarse mode: usual random step
    # ---------------------------------
    new_pos = pos.copy()  # returns a Position

    for k in list(new_pos.keys()):
        if random.random() >= mutation_prob:
            continue
        step = random.randint(-max_step, max_step)
        new_pos[k] = new_pos[k] + step

    return new_pos


def evaluate_population(
    population: List[Dict[str, Any]],
    initPoint: Position,
    template_pos: Position,
    ccmf,
    p: int,
    q: int,
    expr,
    trueConst,
    depth: int,
    cores: int,
    walk_type: int = 1,
    max_coord_init: int = 10,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Evaluate δ for all individuals with delta == None, resampling if δ = -2.
    """
    to_eval = [(i, ind["trajectory"]) for i, ind in enumerate(population) if ind["delta"] is None]

    if not to_eval:
        return population

    with __import__("multiprocessing").Pool(processes=cores) as pool:
        async_results = [
            pool.apply_async(
                trajDelta,
                args=(initPoint, traj, ccmf, p, q, expr, trueConst, depth,walk_type),
            )
            for _, traj in to_eval
        ]
        evaluated = [res.get() for res in async_results]

    for (idx, _), res in zip(to_eval, evaluated):
        delta_val = res["delta"]

        if delta_val == -2:
            # resample
            new_res = res
            for _ in range(max_retries):
                new_traj = random_position_like(template_pos, p, q, max_coord_init)
                new_res = trajDelta(initPoint, new_traj, ccmf, p, q, expr, trueConst, depth,walk_type)
                if new_res["delta"] != -2:
                    population[idx]["trajectory"] = new_traj
                    population[idx]["delta"] = new_res["delta"]
                    break
            else:
                population[idx]["delta"] = -2
        else:
            population[idx]["delta"] = delta_val

    return population


def search_traj_ga(
    initPoint: Position,
    template_pos: Position,
    ccmf,
    p: int,
    q: int,
    expr,
    trueConst,
    depth: int,
    cores: int,
    generations: int,
    pop_size: int,
    max_coord_init: int,
    elite_fraction: float = 0.2,
    mutation_prob: float = 0.3,
    mutation_step: int = 1,
    crossover_prob: float = 0.5,
    walk_type: int = 1,
) -> Dict[str, Any]:
    """
    Genetic search over trajectories (Positions).

    Population individuals are dicts:
        {"trajectory": Position, "delta": Optional[float]}

    - Initialization uses random_position() based on template_pos.
    - Evaluation uses evaluate_population(), which calls trajDelta().
    - Mutation uses mutate_position() with a refinement move 2*pos ± 1.
    """

    # --------------------
    # 0) Initialize population
    # --------------------
    population: List[Dict[str, Any]] = []
    for _ in range(pop_size):
        traj = random_position_like( template_pos,p, q, max_coord_init)
        population.append({"trajectory": traj, "delta": None})

    history: List[Tuple[int, float]] = []

    # --------------------
    # 1) GA generations loop
    # --------------------
    for gen in range(generations):
        print(f"Generation {gen}...")

        # Evaluate individuals that still have delta=None (and resample bad ones)
        population = evaluate_population(
            population=population,
            initPoint=initPoint,
            template_pos=template_pos,
            ccmf=ccmf,
            p=p,
            q=q,
            expr=expr,
            trueConst=trueConst,
            depth=depth,
            cores=cores,
            walk_type=walk_type,
            max_coord_init=max_coord_init,
        )

        # Sort by fitness (delta), best first
        population.sort(key=lambda ind: ind["delta"], reverse=True)
        best = population[0]
        history.append((gen, best["delta"]))

        print("  Best delta this generation:", best["delta"])
        print("  Best trajectory:", best["trajectory"])
        print("  Average delta:", get_avg_delta(population))
        print("  Current population:", population)


        # --------------------
        # Selection (elitism)
        # --------------------
        elite_count = max(1, int(elite_fraction * pop_size))
        elites = population[:elite_count]

        new_population: List[Dict[str, Any]] = []
        # keep elites (copy structure but keep trajectory reference)
        for ind in elites:
            new_population.append({"trajectory": ind["trajectory"], "delta": ind["delta"]})

        # --------------------
        # Reproduction (crossover + mutation)
        # --------------------
        while len(new_population) < pop_size:
            parent1 = random.choice(elites)
            parent2 = random.choice(elites)

            # Crossover
            if random.random() < crossover_prob:
                child1_traj, child2_traj = crossover_positions(
                    parent1["trajectory"], parent2["trajectory"]
                )
            else:
                # no crossover: children are copies of parents
                child1_traj = Position(parent1["trajectory"].copy())
                child2_traj = Position(parent2["trajectory"].copy())

            # Mutation on child1
            if random.random() < mutation_prob:
                child1_traj = mutate_position(
                    child1_traj,
                    max_step=mutation_step,
                    mutation_prob=mutation_prob,
                    # refine move 2*pos ± 1 with some probability
                    refine_prob=0.7,
                    refine_coord_prob=0.5,
                )

            new_population.append({"trajectory": child1_traj, "delta": None})

            # Mutation + add child2 if we still have room
            if len(new_population) < pop_size:
                if random.random() < mutation_prob:
                    child2_traj = mutate_position(
                        child2_traj,
                        max_step=mutation_step,
                        mutation_prob=mutation_prob,
                        refine_prob=0.3,
                        refine_coord_prob=0.5,
                    )
                new_population.append({"trajectory": child2_traj, "delta": None})

        # Move to next generation
        population = new_population

    # --------------------
    # 2) Final evaluation and best result
    # --------------------
    population = evaluate_population(
        population=population,
        initPoint=initPoint,
        template_pos=template_pos,
        ccmf=ccmf,
        p=p,
        q=q,
        expr=expr,
        trueConst=trueConst,
        depth=depth,
        cores=cores,
        walk_type=walk_type,
        max_coord_init=max_coord_init,
    )
    population.sort(key=lambda ind: ind["delta"], reverse=True)
    best = population[0]

    print("Final best delta:", best["delta"])
    print("Final best trajectory:", best["trajectory"])


    return {
        "best": best,          # {"trajectory": Position, "delta": float}
        "history": history,    # list of (generation, best_delta)
    }