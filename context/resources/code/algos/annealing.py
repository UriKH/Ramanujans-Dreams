from typing import List, Dict, Any
import math
from ramanujantools import Position
from cmf_search.positions import next_steps
from cmf_search.evaluation import evaluate_trajectories


def update_old_list_neighs(old_pos_list: List[Position], neighs: List[Position]) -> List[Position]:
    list_size = 14 * 5
    ret = old_pos_list + neighs
    if len(ret) > list_size:
        ret = ret[len(ret) - list_size :]
    return ret


def update_old_list(old_pos_list: List[Position], cur_traj: Position) -> List[Position]:
    list_size = 10
    ret = old_pos_list + [cur_traj]
    if len(ret) > list_size:
        ret = ret[1:]
    return ret


def get_temp(T0: float, k: int, schedule_type: str = "linear") -> float:
    if schedule_type == "log":
        return T0 / math.log(k + 1)
    elif schedule_type == "linear":
        return T0 / (k + 1)
    else:
        raise ValueError(f"Unknown schedule_type {schedule_type}")


def search_traj_sa(
    init_point: Position,
    traj: Position,
    iterations: int,
    maxRes: int,
    ccmf,
    p: int,
    q: int,
    expr,
    true_const,
    depth: int,
    cores: int,
    T0: float = 1.0,
    Tmin: float = 1e-3,
    walk_type: int = 1,
) -> List[Dict[str, Any]]:
    """
    Simulated annealing over trajectories.

    Returns a list of dicts {"trajectory": Position, "delta": float}.
    """
    cur_traj = traj
    trajMul = 1
    T = T0
    iter_left = iterations
    data: List[Dict[str, Any]] = []

    # initial δ
    from .metrics import trajDelta
    cur_delta = trajDelta(init_point, traj, ccmf, p, q, expr, true_const, depth,walk_type)["delta"]
    best_res = cur_delta
    old_pos_list: List[Position] = [traj]

    while iter_left and T > Tmin:
        neighs = next_steps(p, q, cur_traj, old_pos_list)

        neighs_results = evaluate_trajectories(
            init_point, neighs, ccmf, p, q, expr, true_const, depth, cores,walk_type
        )
        data.extend(neighs_results)

        accepted = False
        old_pos_list = update_old_list_neighs(old_pos_list, neighs + [cur_traj])

        neighs_results.sort(key=lambda d: d["delta"], reverse=True)
        new_delta = neighs_results[0]["delta"]
        print(neighs_results)
        if new_delta >= cur_delta:
            cur_traj = neighs_results[0]["trajectory"]
            cur_delta = new_delta
            accepted = True
            iter_left -= 1
        else:
            # classic Metropolis step
            diff = new_delta - cur_delta
            prob = math.exp(diff / T)
            from random import random

            if random() < prob:
                cur_traj = neighs_results[0]["trajectory"]
                cur_delta = new_delta
                accepted = True
                iter_left -= 1

        # adaptive trajectory scaling
        if not accepted:
            cur_traj = 2*cur_traj
            give_up = trajMul > maxRes
            if give_up:
                # reset multiplier and try a new random direction
                trajMul = 1
        else:
            trajMul = 1

        if cur_delta > best_res:
            best_res = cur_delta

        print("best delta", best_res)
        print("cur delta", cur_delta)

        T = get_temp(T0, iterations - iter_left, schedule_type="linear")
        print("T =", T)

    return data
