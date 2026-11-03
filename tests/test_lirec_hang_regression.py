"""Regression: LIReC identification must terminate on a degenerate walk column.

Found 2026-11-03: a 5-D 3F2(1/2) / log 2 analysis hung for 40 min on one trajectory —
start (-4,-1,-1,-2,-2), direction (-6,-2,-2,-3,-2).  Its walk column at the
identification depth is v = (1, X, 7/10 - X + e) with X ~ 1e339 and |e| ~ 1e-1188
(c1 grows like 10^(0.34 n) while c1 + c2 -> 7/10 like 10^(-1.19 n)): its only limits are
rational, so it carries no information about log 2.  LIReC's ``poly_check`` runs PSLQ
with ``maxcoeff = maxsteps = ∞``, and on this input PSLQ's certified norm bound plateaus
(precision exhausted) while no exit condition ever fires.

Deliberately imports nothing from the fix, so it can be run on unfixed code (it then
fails with the suite's per-test timeout).
"""

import pytest
import sympy as sp
from ramanujantools import Position

from dreamer.configs import config


def _hanging_column(monkeypatch):
    from dreamer import log
    from dreamer.loading import pFq
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
    l2 = log(2)
    cmf = pFq(l2, 3, 2, sp.Rational(1, 2)).to_cmf().cmf
    syms = list(cmf.matrices.keys())
    start = Position({s: sp.Integer(v) for s, v in zip(syms, [-4, -1, -1, -2, -2])})
    direction = Position({s: sp.Integer(v) for s, v in zip(syms, [-6, -2, -2, -3, -2])})
    handler = TrajectoryAttributesHandler.from_cmf(cmf, direction, start, constant=l2.value_sympy)
    col, _ = handler._effective_walk_values(int(config.search.IDENTIFY_DEPTH))
    return l2, col


@pytest.mark.timeout(240)
def test_rd_identification_terminates_and_rejects(monkeypatch):
    """End to end through RD's own path: identification ends and reports not identified."""
    l2, col = _hanging_column(monkeypatch)
    from dreamer.loading import pFq
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

    gap = col[1] + col[2] - sp.Rational(7, 10)
    assert col[1] > 10 ** 300 and abs(gap) < sp.Rational(1, 10 ** 1000), \
        "not the degenerate column — test setup changed"
    cmf = pFq(l2, 3, 2, sp.Rational(1, 2)).to_cmf().cmf
    syms = list(cmf.matrices.keys())
    handler = TrajectoryAttributesHandler.from_cmf(
        cmf, Position({s: sp.Integer(v) for s, v in zip(syms, [-6, -2, -2, -3, -2])}),
        Position({s: sp.Integer(v) for s, v in zip(syms, [-4, -1, -1, -2, -2])}),
        constant=l2.value_sympy)
    assert handler.identified() is False


@pytest.mark.timeout(240)
def test_lirec_terminates_without_a_relation_for_the_constant(monkeypatch):
    """LIReC as RD calls it returns; any relation it reports is among the column
    entries only (here c1 + c2 = 7/10) — none involves the constant (c0)."""
    from LIReC.db.access import db
    import dreamer.utils.storage.trajectory_attributes  # noqa: F401  (RD's LIReC set-up)

    l2, col = _hanging_column(monkeypatch)
    digits = int(config.search.CONSTANT_NO_DIGITS_LOW_RES)
    res = db.identify([l2.value_sympy.evalf(digits)] + [v.evalf(digits) for v in col[1:]])
    assert all("c0" not in str(rel) for rel in res)
