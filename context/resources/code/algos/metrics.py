from typing import Dict, Any
import math
import sympy as sp
from sympy import Q, ask
from sympy.abc import n
from sympy.matrices.dense import Matrix
from ramanujantools import Position

import math
from ramanujantools import Position


def reduce_position(pos: Position) -> Position:
    """
    Return a 'reduced' Position where all coordinates are divided
    by the gcd of all coordinates.

    Example:
        Position({x0: 10, x1: 15, y0: -5}) -> Position({x0: 2, x1: 3, y0: -1})
    """

    #d = dict(pos)

    g = 0
    for v in pos.values():
        g = math.gcd(g, int(v))
    if g == 0 or g == 1:
        return pos

    reduced_dict = {k: int(v) // g for k, v in pos.items()}
    return Position(reduced_dict)



def delta(estimated: sp.Expr, limit: sp.Expr | float) -> sp.Expr:

    error = sp.Abs(estimated - limit)
    denominator = sp.denom(estimated)
    d = -1 - sp.log(error) / sp.log(denominator)
    return d.evalf()




def trajDelta(
    init: Position,
    trajectory: Position,
    ccmf,
    p: int,
    q: int,
    expr,
    true_const,
    depth: int,
    walk_type: int = 1,  # 1 = inv().T, 2 = raw walk
) -> Dict[str, Any]:
    """
    Compute δ for a single trajectory.

    walk_type:
      1 -> use walk.inv().T (canonical mode)
      2 -> use raw walk (no inverse / transpose)
    """
    x = sp.symbols(f"x:{p+1}")
    y = sp.symbols(f"y:{q+1}")
    a = sp.symbols(f"a:{p+1}")
    b = sp.symbols(f"b:{q+1}")
    c = sp.symbols(f"c:{p+1}")
    trajectory = reduce_position(trajectory)
    try:
        trajMat = ccmf.trajectory_matrix(trajectory, init)
        # Build the walk matrix according to type
        if walk_type == 1:
            # canonical mode: inverse-transpose, then normalize
            walk = trajMat.walk({n: 1}, depth, {n: 0}).inv().T
        elif walk_type == 2:
            # raw mode: original walk, then normalize
            walk = trajMat.walk({n: 1}, depth, {n: 0})
        else:
            raise ValueError(f"Unknown walk_type={walk_type}")

        # normalize so [0,0] = 1
        walk = walk / walk[0, 0]
        col1 = walk.col(0)
        # substitute c_i with entries of col1 (starting from c1)
        subs_dict = {
            c[i]: col1[i]
            for i in range(1, min(len(col1), len(c)))
        }

        estimated = expr.subs(subs_dict)
        delta_curr = delta(estimated, true_const)

        if not ask(Q.finite(delta_curr)):
            print(trajectory)
            delta_curr = -2

    except Exception as e:
        print(trajectory)
        print("error: ",e)
        delta_curr = -2
        return {"trajectory": trajectory, "delta": float(delta_curr)}

    print({"trajectory": trajectory, "delta": float(delta_curr)})
    return {"trajectory": trajectory, "delta": float(delta_curr)}



def trajDeltaMulti(
        init: Position,
        trajectory: Position,
        ccmf,
        p: int,
        q: int,
        exprs,
        true_consts,
        names,
        depth: int,
        walk_type: int = 1,  # 1 = inv().T, 2 = raw walk
) -> Dict[str, Any]:
    """
    Compute δ for a single trajectory.

    walk_type:
      1 -> use walk.inv().T (canonical mode)
      2 -> use raw walk (no inverse / transpose)
    """
    x = sp.symbols(f"x:{p + 1}")
    y = sp.symbols(f"y:{q + 1}")
    c = sp.symbols(f"c:{p + 1}")
    deltas = []
    trajectory = reduce_position(trajectory)

    # try:
    trajMat = ccmf.trajectory_matrix(trajectory, init)

    # Build the walk matrix according to type
    if walk_type == 1:
        # canonical mode: inverse-transpose, then normalize
        walk = trajMat.walk({n: 1}, depth, {n: 0}).inv().T
    elif walk_type == 2:
        # raw mode: original walk, then normalize
        walk = trajMat.walk({n: 1}, depth, {n: 0})
    else:
        raise ValueError(f"Unknown walk_type={walk_type}")

    # normalize so [0,0] = 1
    walk = walk / walk[0, 0]
    col1 = walk.col(0)

    subs_dict = {
        c[i]: col1[i]
        for i in range(1, min(len(col1), len(c)))
    }

    for i in range(len(true_consts)):
        estimated = exprs[i].subs(subs_dict)
        d = delta(estimated, true_consts[i])
        print(d.evalf())
        if not ask(Q.finite(d)):
            deltas.append(-2)
        else:
            deltas.append(d)
        print({"trajectory": trajectory, "deltas": deltas, "const": names})


    # except Exception:
    #     delta_curr = -2

    return {"trajectory": trajectory, "deltas": deltas, "const": names}
