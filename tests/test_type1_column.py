"""Walk-type-1 column without sympy's ``inv().T`` (R30).

For walk type 1 RD reads a column of S = W^{-T}, W = M(0)···M(N−1).  The handler
used to convert the FLINT walk product to sympy and invert it there at every
depth it read.  It now uses FLINT on the stored checkpoint: the inverse once to
choose the projection column, then one solve Wᵀx = e_j per further depth, and
the FLINT inverse for ``Limit.current``.  These tests pin that:

  - the column, its index and ``Limit.current`` are exactly what sympy's
    ``inv().T`` with the previous column rule gives (2F1, 3F2, 4F3, several
    depths, column chosen fresh and reused);
  - the numeric type-1 path never calls sympy's matrix inverse;
  - a singular walk product still gives ``(None, None)`` (no exception);
  - walk type 2 is unchanged.
"""

import pytest
import sympy as sp
from sympy.abc import n

from ramanujantools import Matrix, Position
from ramanujantools.cmf import pFq

from dreamer.utils.constants import ready_made_consts as R
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

# (cmf args, start, direction) — stored trajectories of the δ investigations (non-singular to depth 2000)
CASES = {
    "2f1": ((2, 1, -1), {"x0": 1, "x1": 1, "y0": 3}, {"x0": 1, "x1": 2, "y0": 3}),
    "3f2": ((3, 2, sp.Rational(1, 2)), {"x0": 1, "x1": -1, "x2": -1, "y0": 3, "y1": 3},
            {"x0": 0, "x1": -1, "x2": -2, "y0": 0, "y1": 0}),
    "3f2b": ((3, 2, sp.Rational(1, 2)), {"x0": 1, "x1": 1, "x2": 1, "y0": 3, "y1": 3},
             {"x0": 1, "x1": 1, "x2": 2, "y0": 3, "y1": 3}),
    "4f3": ((4, 3, sp.Rational(1, 2)), {"x0": 1, "x1": 1, "x2": 1, "x3": 1, "y0": 3, "y1": 3, "y2": 3},
            {"x0": 1, "x1": 1, "x2": 1, "x3": 2, "y0": 3, "y1": 3, "y2": 4}),
}
#: Cases where LIReC identifies log 2 with walk type 1 (only those build ``Limit`` objects).
IDENTIFIED_TYPE1 = {"2f1", "3f2b", "4f3"}


def _traj_matrix(case):
    args, start, direction = CASES[case]
    cmf = pFq(*args)
    P = lambda d: Position({sp.Symbol(k): sp.Integer(v) for k, v in d.items()})
    return cmf.trajectory_matrix(P(direction), P(start))


def _reference(m, depth, column=None):
    """Previous behaviour: sympy W.inv().T, first column with non-zero top and no zero entry (else first
    normalisable), normalised by its top entry."""
    S = m.walk({n: 1}, depth, {n: 0}).inv().T
    if column is None:
        first = None
        for j in range(S.cols):
            if S[0, j].is_zero:
                continue
            col = (S / S[0, j]).col(j)
            first = j if first is None else first
            if all(not v.is_zero for v in col):
                first = j
                break
        column = first
    return list((S / S[0, column]).col(column)), column, S


@pytest.mark.parametrize("case", sorted(CASES))
def test_column_and_limits_equal_sympy_inverse(case):
    m = _traj_matrix(case)
    h = TrajectoryAttributesHandler(m, R.log(2).value_sympy, walk_depth=300, walk_type=1)
    col, j = h._effective_walk_values(300)
    ref, jref, _ = _reference(m, 300)
    assert (col, j) == (ref, jref)
    for depth in (150, 299, 600):                     # column index now known: solve path
        col, jj = h._effective_walk_values(depth)
        assert (col, jj) == (_reference(m, depth, jref)[0], jref)
    limits = h._limits([300, 600])
    if case in IDENTIFIED_TYPE1:
        assert [l.current for l in limits] == [_reference(m, d)[2] for d in (300, 600)]
    else:
        assert limits == []                           # not identified: no Limit objects (unchanged rule)


def test_numeric_type1_path_never_uses_sympy_inverse(monkeypatch):
    m = _traj_matrix("3f2b")
    h = TrajectoryAttributesHandler(m, R.log(2).value_sympy, walk_depth=200, walk_type=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("sympy matrix inverse called on the numeric type-1 path")
    monkeypatch.setattr(sp.MatrixBase, "inv", forbidden)
    monkeypatch.setattr(Matrix, "inverse", forbidden)
    col, j = h._effective_walk_values(200)
    assert col is not None and j is not None
    assert h._effective_walk_values(100)[0] is not None
    assert len(h._limits([200])) == 1


def test_singular_walk_gives_none():
    m = Matrix([[n, 1], [0, 1]])                       # M(0) is singular, so W(N) is singular for every N ≥ 1
    h = TrajectoryAttributesHandler(m, R.log(2).value_sympy, walk_depth=10, walk_type=1)
    assert h._effective_walk_values(10) == (None, None)
    assert h._limits([10]) == []


def test_walk_type_2_unchanged():
    m = _traj_matrix("3f2b")
    h = TrajectoryAttributesHandler(m, R.log(2).value_sympy, walk_depth=200, walk_type=2)
    W = m.walk({n: 1}, 200, {n: 0})
    col, j = h._effective_walk_values(200)
    assert col == list((W / W[0, j]).col(j))
    assert [l.current for l in h._limits([200])] == [W]
