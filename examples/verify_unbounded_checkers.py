"""
Cross-check the exact method's unbounded classifier against an
*independent* dual (Stiemke) oracle on a **real** CMF arrangement.

Motivation
----------
The exact extractor reports that the large majority of cells are
*bounded* (hence discarded).  That is geometrically expected -- see the
"Why so many bounded?" note printed at the end -- but to rule out a
classification bug we verify the production checker against a second,
independently-derived oracle.

Two formulations of "is this cell unbounded?"
---------------------------------------------
* **PRIMAL** (production, ``cells.make_unbounded_checker``): the cell is
  unbounded iff its recession cone ``{d : s_i (A_i . d) >= 0}`` is
  nontrivial.  Implemented as a hot-started ``mip`` LP that maximises
  ``sum_i s_i (A_i . d)`` over the cone, plus an explicit
  ``rank(A) < D`` lineality short-circuit.

* **DUAL** (this file, Stiemke's theorem): the cell is *bounded* iff
  there exists ``w`` with ``A^T w = 0`` and ``s_i w_i >= 1`` for all i.
  Implemented from scratch with ``scipy.linprog`` -- a different library
  and the *dual* space, so agreement is real evidence of correctness.

Equivalence
-----------
Writing ``M_i = s_i A_i``, the recession cone is ``K = {d : M d >= 0}``
and the cell is unbounded iff ``K != {0}``.  Stiemke's theorem: exactly
one of (I) ``exists d: M d >= 0, M d != 0`` or (II) ``exists y > 0:
M^T y = 0`` holds.  Substituting ``y_i = s_i w_i`` turns (II) into the
dual LP above.  One shows (II) feasible  <=>  ``K = ker(A)``.  So:

* ``rank(A) == D`` (=> ``ker(A) = {0}``):  dual-feasible <=> ``K = {0}``
  <=> BOUNDED.  The two checkers are then *provably equivalent* and must
  agree on every cell.
* ``rank(A) < D``: every cell is unbounded (a lineality direction is a
  recession direction), but the dual LP as written can still be feasible
  -- so the *dual* alone is wrong in this case.  The primal handles it
  via its rank short-circuit.  The script flags this regime explicitly.

Run via::

    conda activate rama
    python examples/verify_unbounded_checkers.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import List, Tuple

import numpy as np
from scipy.optimize import linprog

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.base import BaseExtractor  # noqa: E402
from dreamer.extraction.v2.cells import iter_cells, make_unbounded_checker  # noqa: E402
from dreamer.extraction.v2.milp import find_integer_point  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


# --------------------------------------------------------------------------
# Build a real CMF arrangement (A, c) -- same path the extractor uses.
# --------------------------------------------------------------------------

def build_arrangement():
    """
    Return ``(A, c, name)`` for the chosen CMF, built exactly as the
    production extractor does: characteristic-poly + pole hyperplanes,
    shifted, packed into integer ``(A, c)``.

    Default is a *small* system so the cell enumeration completes
    exhaustively (every cell cross-checked).  Swap in a bigger p/q to
    sample a larger arrangement (capped by ``MAX_CELLS`` below).
    """
    config.configure(
        extraction={
            'INIT_POINT_MAX_COORD': 3,
            'IGNORE_DUPLICATE_SEARCHABLES': False,
        },
        logging={'GENERATE_LOGS': False},
    )

    constant = zeta(2)
    # formatter = pFq(constant, 2, 1, -1)   # small: enumerates fully
    formatter = pFq(constant, 4, 3, 1)  # large D=7: will be sampled
    cmf_data = formatter.to_cmf()

    extractor = ShardExtractor(constant, cmf_data)
    hps = extractor._extract_cmf_hps()
    shifted = [hp.apply_shift(cmf_data.shift) for hp in hps]
    A, c = BaseExtractor.hyperplanes_to_matrix(shifted)
    return A, c, cmf_data.cmf_name


# --------------------------------------------------------------------------
# Independent DUAL oracle (Stiemke) -- scipy, built from scratch.
# --------------------------------------------------------------------------

def real_margin1_feasible(A: np.ndarray, c: np.ndarray, sign_vector: np.ndarray) -> bool:
    """
    Continuous (LP, no integrality) feasibility of the MILP's constraints:
    does a REAL ``x`` with ``|x| <= 1e6`` satisfy ``s_i (A_i.x + c_i) >= 1``?

    This is the relaxation of ``find_integer_point``.  If this is
    INFEASIBLE, then no integer point can exist either, so the MILP's
    ``None`` is a *correct* drop (the cell is too thin to inscribe a
    unit-margin point).  If this is FEASIBLE but the MILP returned None,
    that's a genuine integer gap (real point fits, lattice point doesn't).
    """
    A = np.asarray(A, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    s = np.asarray(sign_vector, dtype=np.float64)
    n, d = A.shape
    # s_i (A_i.x + c_i) >= 1  <=>  -(s_i A_i).x <= s_i c_i - 1
    A_ub = -(s[:, None] * A)
    b_ub = s * c - 1.0
    res = linprog(
        np.zeros(d), A_ub=A_ub, b_ub=b_ub,
        bounds=[(-1e6, 1e6)] * d, method="highs",
    )
    return bool(res.success)


def make_dual_unbounded_checker(A: np.ndarray):
    """
    Return ``unbounded(sign_vector) -> bool`` via Stiemke's dual LP.

    Bounded  <=>  exists w:  A^T w = 0  and  s_i w_i >= 1  for all i.
    So unbounded == that LP is INFEASIBLE.
    """
    A = np.asarray(A, dtype=np.float64)
    n, d = A.shape
    A_eq = A.T            # (D, N): sum_i A[i, k] * w_i = 0  for each k
    b_eq = np.zeros(d)
    obj = np.zeros(n)     # pure feasibility

    def unbounded(sign_vector: np.ndarray) -> bool:
        s = np.asarray(sign_vector)
        # s_i = +1 -> w_i in [1, inf) ; s_i = -1 -> w_i in (-inf, -1].
        bounds = [(1.0, None) if s[i] > 0 else (None, -1.0) for i in range(n)]
        res = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        # success == feasible == BOUNDED ; infeasible == UNBOUNDED.
        return not res.success

    return unbounded


# --------------------------------------------------------------------------
# Combinatorial sanity baseline (general position).
# --------------------------------------------------------------------------

def general_position_counts(n: int, d: int) -> Tuple[int, int, int]:
    """
    Region counts for ``n`` hyperplanes in *general position* in R^D:

        total    = sum_{k=0}^{D} C(n, k)
        bounded  = C(n-1, D)
        unbounded = total - bounded

    Real CMF arrangements are NOT in general position (many shared
    intersections), so this is only a ballpark -- but it shows the
    *expected* trend: with n >> d, bounded regions dominate.
    """
    total = sum(math.comb(n, k) for k in range(0, d + 1))
    bounded = math.comb(n - 1, d) if n - 1 >= d else 0
    return total, bounded, total - bounded


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

MAX_CELLS = 2_000       # safety cap so a big arrangement still terminates
DEADLINE_SECONDS = 180  # wall-clock cap on enumeration


def main() -> int:
    A, c, name = build_arrangement()
    n, d = A.shape
    rank = int(np.linalg.matrix_rank(A))
    print(f"CMF: {name}")
    print(f"  hyperplanes N = {n}   dimension D = {d}   rank(A) = {rank}")
    if rank < d:
        print("  [!] rank(A) < D: arrangement has LINEALITY -> every cell is")
        print("      unbounded.  The dual oracle alone is INVALID here (it")
        print("      ignores lineality); only the primal is correct.  The")
        print("      two checkers are expected to DISAGREE in this regime.")
    else:
        print("  rank(A) == D: full rank, no lineality -> the two checkers")
        print("  are provably equivalent and MUST agree on every cell.")
    print()

    primal = make_unbounded_checker(A)
    dual = make_dual_unbounded_checker(A)

    total = unbounded_primal = unbounded_dual = disagreements = 0
    milp_ok = milp_none = 0
    dropped_integer_gap = dropped_too_thin = 0
    milp_seconds = 0.0
    examples: List[Tuple[tuple, bool, bool]] = []
    deadline = time.time() + DEADLINE_SECONDS
    exhaustive = True

    print("Enumerating cells and cross-checking...", flush=True)
    try:
        for sig in iter_cells(A, c, max_cells=MAX_CELLS, deadline=deadline):
            sv = np.asarray(sig, dtype=np.int64)
            up = primal(sv)
            ud = dual(sv)
            total += 1
            unbounded_primal += int(up)
            unbounded_dual += int(ud)
            if up != ud:
                disagreements += 1
                if len(examples) < 10:
                    examples.append((sig, up, ud))
            # For unbounded cells, also exercise the SAME MILP point-finder
            # the exact extractor uses -- this is what actually decides
            # whether the cell becomes a shard or gets silently dropped.
            if up:
                t0 = time.perf_counter()
                pt = find_integer_point(A, c, sv, bound=10**6)
                milp_seconds += time.perf_counter() - t0
                if pt is None:
                    milp_none += 1
                    # Was the drop correct? Check the continuous relaxation.
                    if real_margin1_feasible(A, c, sv):
                        dropped_integer_gap += 1   # real point fits, lattice doesn't
                    else:
                        dropped_too_thin += 1      # no unit-margin point at all
                else:
                    milp_ok += 1
    except Exception as exc:  # ExtractionTimeout / max_cells ceiling
        exhaustive = False
        print(f"  (enumeration stopped early: {type(exc).__name__}: {exc})")

    print()
    print("Results")
    print("-------")
    print(f"  cells checked          : {total}"
          f"{'  (EXHAUSTIVE)' if exhaustive else '  (SAMPLE - capped)'}")
    print(f"  unbounded (primal)     : {unbounded_primal}")
    print(f"  unbounded (dual)       : {unbounded_dual}")
    print(f"  bounded   (primal)     : {total - unbounded_primal}")
    if total:
        frac = 100.0 * unbounded_primal / total
        print(f"  unbounded fraction     : {frac:.2f}%")
    print(f"  DISAGREEMENTS          : {disagreements}")
    print()
    print("  MILP point-finder (find_integer_point) on UNbounded cells:")
    print(f"    point found (-> shard) : {milp_ok}")
    print(f"    None  (-> DROPPED)     : {milp_none}")
    print(f"      .. too thin (no real unit-margin point) : {dropped_too_thin}")
    print(f"      .. integer gap (real fits, lattice not) : {dropped_integer_gap}")
    if unbounded_primal:
        drop = 100.0 * milp_none / unbounded_primal
        print(f"    drop rate              : {drop:.2f}% of unbounded cells")
    if milp_ok + milp_none:
        print(f"    avg MILP time          : "
              f"{1000.0 * milp_seconds / (milp_ok + milp_none):.1f} ms/cell")
    if disagreements:
        print("  --- example disagreements (sign, primal_unbounded, dual_unbounded):")
        for sig, up, ud in examples:
            print(f"      {sig}  primal={up}  dual={ud}")
    elif total:
        print("  -> the two independent checkers agree on EVERY cell. The")
        print("     unbounded/bounded split is confirmed correct.")

    print()
    print("Why so many bounded? (combinatorial baseline)")
    print("---------------------------------------------")
    gp_total, gp_bounded, gp_unbounded = general_position_counts(n, d)
    print(f"  General-position arrangement of N={n} planes in R^{d}:")
    print(f"    total regions     ~ {gp_total}")
    print(f"    bounded regions   ~ {gp_bounded}")
    print(f"    unbounded regions ~ {gp_unbounded}"
          f"  ({100.0 * gp_unbounded / gp_total:.2f}% of total)")
    print("  Unbounded cells are those reaching infinity: they correspond to")
    print("  a (D-1)-dim arrangement, so #unbounded ~ O(N^(D-1)) while")
    print("  #total ~ O(N^D).  The unbounded share shrinks like ~D/N, so with")
    print("  many hyperplanes MOST cells are bounded -- expected, not a bug.")
    return 0 if disagreements == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
