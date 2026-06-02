from itertools import product
import sympy as sp
from ramanujantools import Position


def unit_steps_for_traj(traj: Position) -> list[Position]:
    """
    Return all +/-1 unit steps along the coordinates that actually appear in traj.
    """
    steps: list[Position] = []

    for coord in traj.keys():  # coord is a sympy.Symbol
        # +1 step
        s_plus = Position()
        s_plus[coord] = 1
        steps.append(s_plus)

        # -1 step
        s_minus = Position()
        s_minus[coord] = -1
        steps.append(s_minus)

    return steps


def next_steps(p: int, q: int, traj: Position, old_pos_list: list[Position]=list()) -> list[Position]:
    """
    Generate neighbors of traj that are not in old_pos_list.
    """
    steps = unit_steps_for_traj(traj)
    next_list: list[Position] = []

    for step in steps:
        cur_pos = traj + step
        if cur_pos not in old_pos_list:
            next_list.append(cur_pos)
    return next_list


def iter_positions_xy(p: int, q: int, n: int):
    """
    Iterate all positions with coordinates in [-n, n].
    Used for your grid search experiments.
    """
    xs = sp.symbols(f"x0:{p}") if p > 0 else []
    ys = sp.symbols(f"y0:{q}") if q > 0 else []
    keys = list(xs) + list(ys)

    rng = range(-n, n + 1)
    for vals in product(rng, repeat=len(keys)):
        yield Position({k: v for k, v in zip(keys, vals)})
