"""
Fast trigamma ψ₁(p/3) with arb (R33, part a: denominator 3).

arb evaluates ψ_m(x), the Hurwitz ζ(s, x) and Dirichlet L-functions by
Euler–Maclaurin summation, whose cost and memory grow quickly with the precision:
ψ₁(1/3) takes 1.4 s / 35 MB at 10 000 digits and 62 s / ~1 GB at 50 000
(``context/investigations/big_walk_memory/README.md`` §11).  For the
denominator 3 there is a hypergeometric route, which arb sums by binary splitting:

    S = Σ_{n≥1} 1 / (n³ C(2n, n)) = ½ · ₄F₃(1, 1, 1, 1; 2, 2, 3/2; ¼),
    ψ₁(1/3) − ψ₁(2/3) = 18/(π√3) · (S + 4ζ(3)/3)        (= 9·L(2, χ₋₃)),
    ψ₁(1/3) + ψ₁(2/3) = 4π²/3                              (reflection),

0.008 s / 2 MB at 10 000 digits, 0.06 s / 4 MB at 50 000.  Other arguments
p/3 = r + n (r ∈ {1/3, 2/3}, integer n) use the exact recurrence
ψ₁(x + 1) = ψ₁(x) − 1/x²; for x > 0, ζ(2, x) = ψ₁(x).

The general case ψ_m(p/q) (any order and denominator, via Dirichlet-L
decomposition) is R33 part b and not implemented here.
"""

from __future__ import annotations

#: Largest |n| for which the recurrence's exact rational sum is used.
MAX_SHIFT = 1000


def _base_pair():
    """(ψ₁(1/3), ψ₁(2/3)) as arb balls at the current ``ctx.prec``."""
    from flint import arb

    quarter = arb(1) / 4
    s = quarter.hypgeom([1, 1, 1, 1], [2, 2, arb(3) / 2]) / 2          # Σ_{n≥1} 1/(n³ C(2n,n))
    pi = arb.pi()
    diff = 18 / (pi * arb(3).sqrt()) * (s + 4 * arb(3).zeta() / 3)
    total = 4 * pi ** 2 / 3
    return (total + diff) / 2, (total - diff) / 2


def trigamma_third(p: int, q: int):
    """ψ₁(p/q) for q = 3 and p not divisible by 3, as an arb ball at the current ``ctx.prec``.

    :param p: Numerator (any sign).
    :param q: Denominator; must be 3 after reducing p/q.
    :raises ValueError: if p/q does not reduce to denominator 3, or the integer shift exceeds :data:`MAX_SHIFT`.
    """
    from flint import arb, fmpq

    x = fmpq(int(p), int(q))
    if int(x.q) != 3:
        raise ValueError(f"trigamma_third needs denominator 3, got {p}/{q}")
    n, residue = divmod(int(x.p), 3)                   # x = residue/3 + n, residue ∈ {1, 2}
    if abs(n) > MAX_SHIFT:
        raise ValueError(f"shift {n} exceeds MAX_SHIFT = {MAX_SHIFT}")
    r = fmpq(residue, 3)
    correction = fmpq(0)
    for k in range(n):                                 # ψ₁(r + n) = ψ₁(r) − Σ_{k<n} 1/(r + k)²
        correction -= 1 / (r + k) ** 2
    for k in range(1, -n + 1):                         # ψ₁(r − m) = ψ₁(r) + Σ_{k≤m} 1/(r − k)²
        correction += 1 / (r - k) ** 2
    third, two_thirds = _base_pair()
    return (third if residue == 1 else two_thirds) + arb(correction)
