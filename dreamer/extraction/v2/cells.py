"""
Sign-pattern (cell) enumeration for a hyperplane arrangement.

Given ``N`` hyperplanes ``A x + c = 0`` in :math:`\\mathbb{R}^D`, the
arrangement partitions space into open cells; each cell is uniquely
labelled by a sign vector ``s in {-1, +1}^N``.  Iterating over all
``2^N`` sign vectors is infeasible past ``N ~ 25``, but the number of
*non-empty* cells is bounded by :math:`O(N^D)`.

We enumerate non-empty cells via BFS in the "flip graph":

1. Pick a starting feasible sign vector via random sampling.
2. From each visited cell, flip bit ``i`` (for every ``i``) and ask an
   LP whether the resulting cell has non-empty interior.  Enqueue when
   feasible and unseen.

Interior feasibility is decided by maximising the slack ``t`` in

    s_i * (A_i . x + c_i) >= t   for all i,
    -B <= x_d <= B               (large box keeps the LP bounded)

and accepting iff the optimum ``t`` is strictly positive.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog


SignTuple = Tuple[int, ...]


def _interior_slack(
    A: np.ndarray,
    c: np.ndarray,
    sign_vector: np.ndarray,
    *,
    bound: float = 1e6,
) -> float:
    """
    Return the maximum interior slack of a cell, or ``-inf`` if the LP
    fails.

    Solves ``max t`` subject to ``s_i (A_i . x + c_i) - t >= 0`` and
    ``|x_d| <= bound``.  A strictly positive optimum proves the cell
    has non-empty interior.
    """
    n, d = A.shape
    # Decision variables: [x_1, ..., x_d, t].  linprog minimises -> use -t.
    obj = np.zeros(d + 1, dtype=np.float64)
    obj[-1] = -1.0
    A_ub = np.zeros((n, d + 1), dtype=np.float64)
    A_ub[:, :d] = -(sign_vector[:, None] * A)  # s_i A_i x >= t  =>  -s_i A_i x + t <= 0
    A_ub[:, -1] = 1.0
    b_ub = (sign_vector * c).astype(np.float64)  # s_i c_i
    bounds = [(-bound, bound)] * d + [(None, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        return float("-inf")
    return float(res.x[-1])


def _sign_at(A: np.ndarray, c: np.ndarray, x: np.ndarray) -> Optional[np.ndarray]:
    """
    Return the sign vector at ``x`` or :data:`None` if ``x`` lies on any
    hyperplane.
    """
    vals = A @ x + c
    if np.any(vals == 0):
        return None
    return np.where(vals > 0, 1, -1).astype(np.int64)


def _find_start_cell(
    A: np.ndarray,
    c: np.ndarray,
    *,
    rng: np.random.Generator,
    max_attempts: int = 64,
    radius: int = 1000,
) -> np.ndarray:
    """
    Find any non-empty cell by random integer sampling.

    :raises RuntimeError: If no off-hyperplane point is found in
        ``max_attempts`` tries.
    """
    d = A.shape[1]
    for _ in range(max_attempts):
        x = rng.integers(-radius, radius + 1, size=d, dtype=np.int64)
        sig = _sign_at(A, c, x)
        if sig is not None:
            return sig
    raise RuntimeError(
        f"Could not find a starting cell after {max_attempts} random samples"
    )


def enumerate_cells(
    A: np.ndarray,
    c: np.ndarray,
    *,
    max_cells: int = 100_000,
    seed: Optional[int] = 0,
    epsilon: float = 1e-9,
) -> List[SignTuple]:
    """
    Enumerate every non-empty cell of the arrangement.

    :param A: Hyperplane coefficient matrix, shape ``(N, D)``.
    :param c: Hyperplane constants, shape ``(N,)``.
    :param max_cells: Safety cap.  Raises if exceeded -- prevents
        runaway BFS on pathological arrangements.
    :param seed: RNG seed used to find the starting cell.
    :param epsilon: Slack threshold above which a cell is considered to
        have non-empty interior.
    :return: List of sign-encoding tuples (``+1`` / ``-1``).
    :raises RuntimeError: If ``max_cells`` is exceeded or no starting
        cell can be found.
    :raises ValueError: For inconsistent input shapes.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")

    n = A.shape[0]
    rng = np.random.default_rng(seed)
    start = _find_start_cell(A, c, rng=rng)

    seen: Set[SignTuple] = {tuple(start.tolist())}
    queue: deque = deque([start])

    while queue:
        sig = queue.popleft()
        for i in range(n):
            flipped = sig.copy()
            flipped[i] = -flipped[i]
            key = tuple(flipped.tolist())
            if key in seen:
                continue
            slack = _interior_slack(A, c, flipped)
            if slack > epsilon:
                seen.add(key)
                queue.append(flipped)
                if len(seen) > max_cells:
                    raise RuntimeError(
                        f"Cell enumeration exceeded max_cells={max_cells}"
                    )
    return sorted(seen)
