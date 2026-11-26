"""Fast ψ₁ / ζ(2, ·) at arguments with denominator 3 (R33 part a).

arb's ``polygamma``, Hurwitz ``zeta`` and ``dirichlet_l`` all use Euler–Maclaurin
for ψ₁(1/3): 1.4 s / 35 MB at 10 000 digits, 62 s / ~1 GB at 50 000
(``context/investigations/big_walk_memory`` §11).  RD evaluates them instead from

    ψ₁(1/3) − ψ₁(2/3) = 18/(π√3) · (S + 4ζ(3)/3),   S = Σ 1/(n³ C(2n,n)) = ½·₄F₃(1,1,1,1; 2,2,3/2; ¼),
    ψ₁(1/3) + ψ₁(2/3) = 4π²/3,

plus the recurrence ψ₁(x + 1) = ψ₁(x) − 1/x² for other residues p/3 + n
(0.06 s / 4 MB at 50 000 digits).  These tests pin the values (against mpmath and
arb's generic routine, at several precisions) and the speed.
"""

import time

import pytest
import sympy as sp
from flint import acb, arb, ctx, fmpq

from dreamer.utils.constants.high_precision import arb_value
from dreamer.utils.constants.rational_polygamma import trigamma_third

R = sp.Rational
ARGS = [R(1, 3), R(2, 3), R(4, 3), R(5, 3), R(13, 3), R(-1, 3), R(-5, 3), R(-22, 3)]


def _agree(a, b):
    d = abs(a - b)
    return float("inf") if d == 0 else float(-sp.log(d / abs(b), 10).evalf(20))


@pytest.mark.parametrize("a", ARGS)
@pytest.mark.parametrize("node", ["polygamma", "zeta"])
def test_agrees_with_sympy(a, node):
    expr = sp.polygamma(1, a) if node == "polygamma" else sp.zeta(2, a)
    if node == "zeta" and a < 0:
        pytest.skip("Hurwitz ζ(2, a) is used only for a > 0")
    assert _agree(arb_value(expr, 1000), expr.evalf(1030)) >= 1000


@pytest.mark.parametrize("digits", [30, 300, 3000])
@pytest.mark.parametrize("a", [R(1, 3), R(2, 3), R(7, 3), R(-4, 3)])
def test_agrees_with_arb_generic_route(a, digits):
    old = ctx.prec
    try:
        ctx.prec = int(digits * 3.33) + 64
        fast = trigamma_third(int(a.p), int(a.q))
        generic = acb(arb(fmpq(int(a.p), int(a.q)))).polygamma(acb(1)).real
        assert fast.overlaps(generic)
        assert fast.rel_accuracy_bits() >= ctx.prec - 32
    finally:
        ctx.prec = old


def test_only_denominator_three():
    with pytest.raises(ValueError):
        trigamma_third(1, 4)
    with pytest.raises(ValueError):
        trigamma_third(3, 3)                       # an integer is not p/3


def test_combination_used_for_dirichlet_l():
    """L(2, χ₋₃) = (ψ₁(1/3) − ψ₁(2/3))/9 — RD writes such targets as polygamma combinations."""
    expr = (sp.polygamma(1, R(1, 3)) - sp.polygamma(1, R(2, 3))) / 9
    assert _agree(arb_value(expr, 2000), expr.evalf(2030)) >= 2000


@pytest.mark.timeout(120)
def test_fast_at_high_precision():
    t = time.perf_counter()
    arb_value(sp.polygamma(1, R(1, 3)), 30_000)
    arb_value(sp.zeta(2, R(2, 3)), 30_000)
    assert time.perf_counter() - t < 2           # fast route ≈ 0.05 s; Euler–Maclaurin ≈ 15 s each
