"""
Form search: the fastest-converging integer form (p, q) of a walk column.

A walk column v (n entries) gives approximants p·v / q·v.  When n ≥ 3 several integer forms converge to the
constant L, at different rates, so δ depends on the form.  LIReC returns the form with the smallest coefficients
that holds to its input precision, and the shard cache returns the first matching form — both often weak.

LLL on the lattice spanned by the rows ``e_i | round(10^N · x_i)``, ``x = [v, L·v]`` (scale-free), makes a form with
coefficients of size ‖c‖ and relative residual ε a vector of length about ``max(‖c‖, 10^N · ε)``: it is short only
if it holds to about N digits.  So LLL at ``N = digits(current form) + margin`` returns only forms that converge
*faster* than the current one, or none.  Iterating ends at the fastest form within the coefficient bound — a
function of the walk column alone, not of the start form.  A final step picks, among the forms at that rate, the
one with the best δ (the forms differ only in their reduced denominators).

All values are exact rationals of the walk; L is evaluated by the caller at a precision above the column's size.
"""
import itertools
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import mpmath as mp
import sympy as sp

Form = Tuple[Tuple[sp.Rational, ...], Tuple[sp.Rational, ...]]


@dataclass(frozen=True)
class FormSearchSettings:
    """Knobs of the form search (see ``search.FORM_SEARCH_*`` in the config)."""
    margin_digits: int = 30
    coef_bound: int = 10 ** 12
    max_steps: int = 10
    significance_digits: int = 100
    equal_rate_top: int = 5
    limit_diff_bound: sp.Rational = sp.Rational(1, 10 ** 10)
    min_denominator: int = 10 ** 6


@dataclass
class FormSearchResult:
    """Outcome of :func:`search`: the final form, its δ, and how it was reached."""
    form: Form
    delta: Optional[float]
    lll_calls: int
    improved: bool


def to_mpf(x) -> mp.mpf:
    """
    Exact sympy rational (or integer) as an mpf at the current precision.
    :param x: The number.
    :return: ``mp.mpf``.
    """
    x = sp.Rational(x)
    return mp.mpf(int(x.p)) / int(x.q)


def column_digits(col: Sequence[sp.Rational]) -> int:
    """
    Decimal size of the largest numerator or denominator in a column (sets the working precision).
    :param col: Walk column.
    :return: Number of digits.
    """
    return max(max(len(str(abs(sp.Rational(c).p))), len(str(sp.Rational(c).q))) for c in col if c != 0)


def working_dps(col: Sequence[sp.Rational]) -> int:
    """
    mpmath precision for scoring forms on a column: well above the column's own size.
    :param col: Walk column at the deepest depth.
    :return: Decimal digits of precision.
    """
    return int(2.2 * column_digits(col)) + 300


def approximant(col: Sequence[sp.Rational], form: Form) -> Optional[sp.Rational]:
    """
    The rational p·v / q·v (signed, reduced); None when q·v = 0.
    :param col: Walk column v.
    :param form: (p, q).
    :return: Exact rational.
    """
    p, q = form
    num = sum(sp.Rational(a) * c for a, c in zip(p, col))
    den = sum(sp.Rational(b) * c for b, c in zip(q, col))
    return None if den == 0 else sp.Rational(num, den)


def form_digits(col, form: Form, L: mp.mpf) -> Optional[float]:
    """
    Correct digits −log10|p·v/q·v − L| of a form on a column.
    :param col: Walk column.
    :param form: (p, q).
    :param L: Target at the working precision.
    :return: Digits, or None when q·v = 0 or the error is below the precision.
    """
    r = approximant(col, form)
    if r is None:
        return None
    err = abs(to_mpf(r) - L)
    return None if err == 0 else float(-mp.log10(err))


def form_delta(col, form: Form, L: mp.mpf) -> Optional[float]:
    """
    δ = −1 − log|p·v/q·v − L| / log Q, with Q the reduced denominator (RD's definition).
    :param col: Walk column.
    :param form: (p, q).
    :param L: Target at the working precision.
    :return: δ, or None when undefined (q·v = 0, Q = 1, error below precision).
    """
    r = approximant(col, form)
    if r is None or r.q == 1:
        return None
    err = abs(to_mpf(r) - L)
    return None if err == 0 else float(-1 - mp.log(err) / mp.log(int(r.q)))


def check_form(cols: Sequence[Sequence[sp.Rational]], form: Form, L: mp.mpf,
               settings: FormSearchSettings) -> Tuple[bool, str]:
    """
    Whether a form is a genuine identification on this trajectory (RD's checks plus a significance bound).
    :param cols: The same column at depths 0.9d, 0.95d, d.
    :param form: (p, q).
    :param L: Target at the working precision.
    :param settings: Search settings.
    :return: (valid, reason).
    """
    p, q = form
    coefs = list(p) + list(q)
    if max(abs(c) for c in coefs) > settings.coef_bound:
        return False, "coefficients too large"
    if sp.Matrix([list(p), list(q)]).rank() < 2:
        return False, "p ∥ q"
    ests = [approximant(c, form) for c in cols]
    if any(e is None for e in ests):
        return False, "q·v = 0"
    if all(e == ests[0] for e in ests[1:]):
        return False, "frozen walk"
    if any(abs(ests[i] - ests[i - 1]) >= settings.limit_diff_bound for i in range(1, len(ests))):
        return False, "not converging"
    if ests[-1].q <= settings.min_denominator:
        return False, "small denominator"
    errs = [abs(to_mpf(e) - L) for e in (ests[0], ests[-1])]
    if errs[1] == 0:
        return False, "below precision"
    if not errs[1] < errs[0]:
        return False, "error not shrinking"
    # A relation among 2n numbers with coefficients ≤ C holds by chance to about (2n − 1)·log10 C digits.
    chance = (2 * len(cols[-1]) - 1) * math.log10(max(abs(c) for c in coefs) + 1)
    if float(-mp.log10(errs[1])) < chance + settings.significance_digits:
        return False, "not significant"
    return True, "ok"


def lll_relations(col: Sequence[sp.Rational], L: mp.mpf, weight_digits: int, coef_bound: int) -> List[Form]:
    """
    Short relations a·v + b·(L v) ≈ 0 from one LLL reduction at weight 10^weight_digits, as forms (p, q) = (−a, b).
    :param col: Walk column (deepest depth).
    :param L: Target at a precision above ``weight_digits``.
    :param weight_digits: N in the weight 10^N.
    :param coef_bound: Largest coefficient kept.
    :return: Candidate forms (unchecked), gcd-normalised.
    """
    from fpylll import IntegerMatrix, LLL
    v = [to_mpf(c) for c in col]
    scale = max(abs(t) for t in v)
    x = [t / scale for t in v] + [L * t / scale for t in v]
    n = len(x)
    B = IntegerMatrix(n, n + 1)
    W = mp.mpf(10) ** weight_digits
    for i in range(n):
        B[i, i] = 1
        B[i, n] = int(mp.nint(W * x[i]))
    LLL.reduction(B)
    half = n // 2
    out = []
    for k in range(n):
        rel = [int(B[k, j]) for j in range(n)]
        g = math.gcd(*rel)
        if g == 0 or max(abs(c) for c in rel) > coef_bound or not any(rel[half:]):
            continue
        rel = [c // g for c in rel]
        out.append((tuple(sp.Integer(-c) for c in rel[:half]), tuple(sp.Integer(c) for c in rel[half:])))
    return out


def ladder_start(cols, L: mp.mpf, settings: FormSearchSettings, levels: int = 12) -> Optional[Form]:
    """
    A first valid form when nothing identified the trajectory: LLL on a geometric ladder of weights.
    :param cols: Column at 0.9d, 0.95d, d.
    :param L: Target at the working precision.
    :param settings: Search settings.
    :param levels: Ladder levels.
    :return: The valid candidate holding to the most digits, or None.
    """
    col = cols[-1]
    top = working_dps(col) - 300
    lo = 30
    ladder = sorted({int(lo * (top / lo) ** (k / (levels - 1))) for k in range(levels)}) if top > lo else [top]
    best = None
    for N in ladder:
        for f in lll_relations(col, L, N, settings.coef_bound):
            if check_form(cols, f, L, settings)[0]:
                d = form_digits(col, f, L)
                if d is not None and (best is None or d > best[0]):
                    best = (d, f)
    return best[1] if best else None


def refine(cols, L: mp.mpf, start: Form, settings: FormSearchSettings) -> Tuple[Form, int]:
    """
    Window steps: LLL at weight digits(current) + margin until no valid form holds to more digits.
    :param cols: Column at 0.9d, 0.95d, d.
    :param L: Target at the working precision.
    :param start: Starting form (any valid form of the trajectory).
    :param settings: Search settings.
    :return: (fastest form found, number of LLL calls).
    """
    col = cols[-1]
    cur, calls = start, 0
    for _ in range(settings.max_steps):
        dig = form_digits(col, cur, L)
        if dig is None:
            break
        calls += 1
        better = []
        for f in lll_relations(col, L, int(dig) + settings.margin_digits, settings.coef_bound):
            d = form_digits(col, f, L)
            if d is not None and d > dig + 1 and check_form(cols, f, L, settings)[0]:
                better.append((d, f))
        if not better:
            break
        cur = max(better, key=lambda t: t[0])[1]
    return cur, calls


def best_equal_rate(cols, L: mp.mpf, form: Form, settings: FormSearchSettings) -> Tuple[Form, int]:
    """
    Among the forms converging at the same rate as ``form``, the one with the best δ.

    Forms at one rate differ only in their reduced denominators.  The basis of those forms comes from one LLL call
    at weight digits − margin.  With a single basis vector there is nothing to choose.  Otherwise combinations with
    coefficients in {−1, 0, 1} are ranked by their exact reduced denominator (an integer gcd, cheap), and only the
    ``equal_rate_top`` smallest are scored.
    :param cols: Column at 0.9d, 0.95d, d.
    :param L: Target at the working precision.
    :param form: The fastest form found by :func:`refine`.
    :param settings: Search settings.
    :return: (best form, number of LLL calls).
    """
    col = cols[-1]
    dig = form_digits(col, form, L)
    if dig is None:
        return form, 0
    basis = lll_relations(col, L, max(int(dig) - settings.margin_digits, 10), settings.coef_bound)
    if len(basis) <= 1:
        return form, 1
    n = len(col)
    vecs = [[-c for c in p] + list(q) for p, q in basis]
    ranked = []
    for coeffs in itertools.product((-1, 0, 1), repeat=len(vecs)):
        if not any(coeffs):
            continue
        rel = [sum(c * v[i] for c, v in zip(coeffs, vecs)) for i in range(2 * n)]
        g = math.gcd(*(int(x) for x in rel))
        if not g:
            continue
        rel = [sp.Integer(int(x) // g) for x in rel]
        f = (tuple(-x for x in rel[:n]), tuple(rel[n:]))
        r = approximant(col, f)
        if r is not None and r.q > 1:
            ranked.append((int(r.q).bit_length(), f))
    ranked.sort(key=lambda t: t[0])
    best = (form_delta(col, form, L), form)
    checked = 0
    for _, f in ranked:
        if checked >= settings.equal_rate_top:
            break
        d = form_digits(col, f, L)
        if d is None or d < dig - 1:
            continue                                   # not at the final rate
        checked += 1
        if check_form(cols, f, L, settings)[0]:
            s = form_delta(col, f, L)
            if s is not None and (best[0] is None or s > best[0] + 1e-12):
                best = (s, f)
    return best[1], 1


def search(cols, L: mp.mpf, start: Form, settings: FormSearchSettings) -> FormSearchResult:
    """
    The fastest-converging form of the trajectory, then the best δ at that rate.
    :param cols: Column at 0.9d, 0.95d, d (exact rationals).
    :param L: Target at ``working_dps(cols[-1])`` digits.
    :param start: A valid form of the trajectory (LIReC's, a cached one, or :func:`ladder_start`'s).
    :param settings: Search settings.
    :return: The result.
    """
    fast, calls = refine(cols, L, start, settings)
    final, more = best_equal_rate(cols, L, fast, settings)
    return FormSearchResult(form=final, delta=form_delta(cols[-1], final, L), lll_calls=calls + more,
                            improved=final != start)
