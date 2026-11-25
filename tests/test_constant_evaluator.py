"""The target constant is always evaluated by the faster engine (R32).

Benchmark (``context/investigations/big_walk_memory`` §10): arb is faster than
mpmath (sympy ``evalf``) for every supported constant from 100 digits on
(median 3–10×; equal at 15 digits), except arb's own ``polylog``, which is
17 s at 10 000 digits and > 30 min at 30 000 for Li₃(½) (mpmath 0.4 s / 2.8 s).
RD therefore evaluates Li_s(z) as z·ₛ₊₁Fₛ(1,…,1; 2,…,2; z) with arb (0.05 s /
0.19 s), falling back to arb's ``polylog`` when that ball is not accurate
enough (|z| near 1).  These tests pin that:

  - the polylog route agrees with sympy, also near |z| = 1, and is fast at
    30 000 digits;
  - identification, δ and the form checks never evaluate the target constant
    with ``evalf`` (mpmath) at ≥ 100 digits — they use ``high_precision``.
"""

import time

import pytest
import sympy as sp
from sympy.core.evalf import EvalfMixin

from ramanujantools import Position
from ramanujantools.cmf import pFq

from dreamer.utils.constants.high_precision import arb_value
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler


def _agree(a, b):
    d = abs(a - b)
    return float("inf") if d == 0 else float(-sp.log(d / abs(b), 10).evalf(20))


@pytest.mark.parametrize("s, z", [(1, sp.Rational(1, 3)), (2, sp.Rational(1, 2)), (3, sp.Rational(-1, 2)),
                                  (3, sp.Rational(9, 10)), (4, sp.Rational(99, 100)), (5, sp.Rational(1, 10))])
def test_polylog_agrees_with_sympy(s, z):
    expr = sp.polylog(s, z)
    assert _agree(arb_value(expr, 1000), expr.evalf(1030)) >= 1000


@pytest.mark.timeout(120)
def test_polylog_is_fast_at_high_precision():
    t = time.perf_counter()
    arb_value(sp.polylog(3, sp.Rational(1, 2)), 30_000)
    assert time.perf_counter() - t < 10          # hypergeometric route ≈ 0.2 s; arb's polylog > 30 min


@pytest.fixture
def no_mpmath_constant(monkeypatch):
    """Fail if an expression containing ``log`` (the target log 2) is evaluated by evalf at ≥ 100 digits."""
    original = EvalfMixin.evalf

    def guarded(self, n=15, *args, **kwargs):
        if n >= 100 and getattr(self, "has", None) and self.has(sp.log):
            raise AssertionError(f"evalf({n}) of {self} — the constant must go through high_precision")
        return original(self, n, *args, **kwargs)
    monkeypatch.setattr(EvalfMixin, "evalf", guarded)
    monkeypatch.setattr(EvalfMixin, "n", guarded)


def _log2_handler(walk_type):
    cmf = pFq(2, 1, -1)
    P = lambda d: Position({sp.Symbol(k): sp.Integer(v) for k, v in d.items()})
    m = cmf.trajectory_matrix(P({"x0": 1, "x1": 2, "y0": 3}), P({"x0": 1, "x1": 1, "y0": 3}))
    return TrajectoryAttributesHandler(m, sp.log(2), walk_depth=500, walk_type=walk_type)


@pytest.mark.parametrize("walk_type", [1, 2])
def test_identification_and_delta_never_use_evalf_for_the_constant(walk_type, no_mpmath_constant):
    h = _log2_handler(walk_type)
    assert h.identified()
    assert -1 < float(h.delta()) < 1
    values, _ = h._effective_walk_values(500)
    syms = [1] + list(sp.symbols(f"c:{h.traj_size()}")[1:])
    p, q = h.p_vector(), h.q_vector()
    form = sum(a * c for a, c in zip(p, syms)) / sum(b * c for b, c in zip(q, syms))
    assert h._form_correct_digits(form, values) > 50
