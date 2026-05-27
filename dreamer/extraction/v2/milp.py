"""
Integer feasibility helper for v2 extractors.

A cell of a hyperplane arrangement is described by a sign vector
``s in {-1, +1}^N`` and a coefficient matrix ``(A, c)`` so that the
cell is

    { x in R^D : s_i * (A[i] . x + c[i]) > 0  for all i }.

This module exposes :func:`find_integer_point`, which decides whether
the cell contains an integer point and, if so, returns one.  All
hyperplane coefficients are assumed integral (as guaranteed by
:class:`dreamer.extraction.hyperplanes.Hyperplane`); together with
integer ``x`` this lets us tighten the strict inequalities ``> 0`` to
``>= 1``, which makes the problem a standard MILP feasibility instance.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import LinearConstraint, milp, Bounds


def find_integer_point(
    A: np.ndarray,
    c: np.ndarray,
    sign_vector: Sequence[int],
    *,
    bound: int = 10**6,
) -> Optional[np.ndarray]:
    """
    Return one integer point strictly inside the cell defined by
    ``sign_vector``, or :data:`None` if no such point exists.

    For every hyperplane ``i`` the constraint is

        s_i * (A[i] . x + c[i]) >= 1

    which is equivalent to strict inequality ``> 0`` because both sides
    are integers.

    :param A: Hyperplane coefficient matrix, shape ``(N, D)``.
    :param c: Hyperplane constants, shape ``(N,)``.
    :param sign_vector: Length-``N`` sequence of ``+1`` / ``-1``.
    :param bound: Symmetric box ``|x_d| <= bound`` to keep the MILP
        bounded.  The MILP is feasible iff some point in this box
        satisfies the strict cell constraints; for genuinely unbounded
        cells a tiny ``bound`` (e.g. ``10``) is enough -- larger only
        helps when the cell's interior is shifted far from the origin.
    :return: Integer point of shape ``(D,)`` or :data:`None`.
    :raises ValueError: If shapes are inconsistent or
        ``sign_vector`` contains a non ``+/-1`` entry.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    s = np.asarray(sign_vector, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")
    if s.shape != (A.shape[0],):
        raise ValueError(f"sign_vector shape {s.shape} incompatible with A shape {A.shape}")
    if not np.all(np.isin(s, [-1, 1])):
        raise ValueError("sign_vector entries must be +1 or -1")

    D = A.shape[1]
    # s_i * (A_i . x + c_i) >= 1  <=>  (s_i * A_i) . x >= 1 - s_i * c_i
    A_signed = (s[:, None] * A).astype(np.float64)
    lower = (1 - s * c).astype(np.float64)
    upper = np.full(A.shape[0], np.inf)

    constraints = LinearConstraint(A_signed, lower, upper)
    bounds = Bounds(lb=-bound, ub=bound)
    integrality = np.ones(D, dtype=int)
    objective = np.zeros(D, dtype=np.float64)

    result = milp(
        c=objective,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
    )
    if not result.success or result.x is None:
        return None
    pt = np.rint(result.x).astype(np.int64)
    # Final sanity check: round-tripping the MILP solution should still
    # satisfy the strict integer-tight constraint.  If rounding pushed
    # us onto a hyperplane, treat the cell as infeasible.
    if np.any(s * (A @ pt + c) < 1):
        return None
    return pt
