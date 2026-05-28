r"""
Sign-pattern (cell) enumeration for a hyperplane arrangement.

Given ``N`` hyperplanes ``A x + c = 0`` in :math:`\mathbb{R}^D`, the
arrangement partitions space into open cells; each cell is uniquely
labelled by a sign vector ``s in {-1, +1}^N``.  Iterating over all
``2^N`` sign vectors is infeasible past ``N ~ 25``, but the number of
*non-empty* cells is bounded by :math:`O(N^D)`.

We enumerate non-empty cells via BFS in the "flip graph":

1. Pick a starting feasible sign vector via random sampling.
2. From each visited cell, flip bit ``i`` (for every ``i``) and ask an
   LP whether the resulting cell has non-empty interior.  Enqueue when
   feasible and unseen.

Feasibility backend (Phase 1 — hot-started stateful solver)
===========================================================

The dominant cost of the BFS is the per-neighbour feasibility LP.  The
old implementation rebuilt a fresh ``scipy.optimize.linprog`` model
(dense ``N x (D+1)`` matrix) on every call — for ``D >= 7`` that
setup/teardown overhead made exact enumeration time out.

We now build the LP **once** and navigate the arrangement by swapping
variable *bounds* only (``context/shard_extraction/EXACT_EXTRACTION_OPTIMIZATION_SPEC.md``):

* Coordinate variables ``x in [-bound, bound]^D``.
* Auxiliary variables ``y_i`` (one per hyperplane).
* Static equality constraints ``A_i . x + c_i - y_i = 0`` — these never
  change, so the constraint matrix (and the simplex basis in memory) is
  reused across every neighbour check.
* A sign vector maps to ``y`` bounds: ``s_i = +1`` => ``y_i in [eps, inf)``;
  ``s_i = -1`` => ``y_i in (-inf, -eps]``.  Feasibility of those bounds
  proves the open cell has interior of slack ``>= eps``.

Because only bounds change between solves, the solver (CBC via
``python-mip``) hot-starts from the previous basis.  When ``python-mip``
is unavailable we fall back to the original per-call ``scipy`` LP so the
module never hard-fails on a mip-less environment.

Phase 2 (future): for ``11 <= D <= 15`` the BFS ``seen`` set itself
becomes the bottleneck under multiprocessing.  The documented next step
is memoryless Avis–Fukuda reverse search, which assigns each cell a
unique parent and so parallelises across cores with no shared state.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog

try:  # Optional fast path — see module docstring.
    import mip

    _HAS_MIP = True
except ImportError:  # pragma: no cover - depends on the environment
    mip = None  # type: ignore[assignment]
    _HAS_MIP = False


SignTuple = Tuple[int, ...]

# Feasibility predicate: maps a candidate sign vector to "does this open
# cell have non-empty interior?".
FeasibilityChecker = Callable[[np.ndarray], bool]


# ---------------------------------------------------------------------------
# Phase 1: stateful python-mip feasibility solver (bound swapping)
# ---------------------------------------------------------------------------

class _StatefulFeasibilitySolver:
    """
    Hot-started LP feasibility oracle for the cells of an arrangement.

    Builds the LP once; ``feasible(sign_vector)`` only rewrites the
    ``y`` bounds and re-optimises, reusing the simplex basis in memory.
    """

    def __init__(
        self,
        A: np.ndarray,
        c: np.ndarray,
        *,
        bound: float = 1e6,
        epsilon: float = 1e-9,
    ):
        if not _HAS_MIP:  # pragma: no cover - guarded by caller
            raise RuntimeError("python-mip is not available")
        self.n, self.d = A.shape
        # CBC's primal feasibility tolerance is ~1e-7: an epsilon below
        # it would let CBC treat an empty cell's contradictory bounds
        # (y_i >= eps AND y_i <= -eps) as satisfiable at y_i ~ 0, wrongly
        # reporting the cell feasible.  Floor the slack comfortably above
        # that tolerance.  Safe for integer arrangements — a genuinely
        # full-dimensional cell has O(1) slack, so this never rejects a
        # real cell.
        self.epsilon = max(float(epsilon), 1e-6)

        model = mip.Model(sense=mip.MINIMIZE, solver_name=mip.CBC)
        model.verbose = 0  # suppress solver chatter for speed

        # Coordinate variables x, bounded by a large box to keep the LP
        # bounded; auxiliary y_i = A_i . x + c_i carry the per-hyperplane
        # signed distance and are the only bounds we ever touch.
        self._x = [model.add_var(lb=-bound, ub=bound) for _ in range(self.d)]
        self._y = [model.add_var(lb=-mip.INF, ub=mip.INF) for _ in range(self.n)]

        # Static constraints — added once, never modified.
        for i in range(self.n):
            model += (
                mip.xsum(float(A[i, j]) * self._x[j] for j in range(self.d))
                + float(c[i])
                - self._y[i]
                == 0
            )
        model.objective = 0  # feasibility only — any feasible point is optimal

        self._model = model

    def feasible(self, sign_vector: np.ndarray) -> bool:
        """
        Return ``True`` iff the open cell labelled by ``sign_vector``
        has interior of slack ``>= epsilon``.

        Only ``y`` bounds change between calls, so the solver re-uses
        its previous basis (hot start).
        """
        eps = self.epsilon
        for i in range(self.n):
            if sign_vector[i] > 0:
                self._y[i].lb = eps
                self._y[i].ub = mip.INF
            else:
                self._y[i].lb = -mip.INF
                self._y[i].ub = -eps
        status = self._model.optimize()
        return status in (
            mip.OptimizationStatus.OPTIMAL,
            mip.OptimizationStatus.FEASIBLE,
        )


# ---------------------------------------------------------------------------
# Fallback: per-call scipy LP (used when python-mip is unavailable)
# ---------------------------------------------------------------------------

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
    has non-empty interior.  Kept as the fallback feasibility backend
    for environments without ``python-mip``.
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


def _make_feasibility_checker(
    A: np.ndarray,
    c: np.ndarray,
    *,
    epsilon: float,
    bound: float = 1e6,
) -> FeasibilityChecker:
    """
    Build the per-neighbour feasibility predicate.

    Prefers the hot-started ``python-mip`` solver; transparently falls
    back to the per-call scipy LP when mip is unavailable.
    """
    if _HAS_MIP:
        solver = _StatefulFeasibilitySolver(A, c, bound=bound, epsilon=epsilon)
        return solver.feasible
    return lambda sign_vector: _interior_slack(A, c, sign_vector, bound=bound) > epsilon


# ---------------------------------------------------------------------------
# Starting cell
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BFS enumeration
# ---------------------------------------------------------------------------

def enumerate_cells(
    A: np.ndarray,
    c: np.ndarray,
    *,
    max_cells: int = 100_000,
    seed: Optional[int] = 0,
    epsilon: float = 1e-6,
) -> List[SignTuple]:
    """
    Enumerate every non-empty cell of the arrangement.

    :param A: Hyperplane coefficient matrix, shape ``(N, D)``.
    :param c: Hyperplane constants, shape ``(N,)``.
    :param max_cells: Safety cap.  Raises if exceeded -- prevents
        runaway BFS on pathological arrangements.
    :param seed: RNG seed used to find the starting cell.
    :param epsilon: Minimum interior slack for a cell to count as
        non-empty (the ``y`` bound magnitude in the stateful solver).
        Must stay above the LP solver's feasibility tolerance (~1e-7
        for CBC); the stateful solver floors it at 1e-6 for safety.
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

    is_feasible = _make_feasibility_checker(A, c, epsilon=epsilon)

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
            if is_feasible(flipped):
                seen.add(key)
                queue.append(flipped)
                if len(seen) > max_cells:
                    raise RuntimeError(
                        f"Cell enumeration exceeded max_cells={max_cells}"
                    )
    return sorted(seen)
