"""Tests for the LIReC PSLQ bound (``dreamer.utils.constants.lirec_pslq_bound``).

LIReC's ``poly_check`` runs PSLQ with ``maxcoeff = maxsteps = ∞``; on a degenerate walk
column it never ends (found 2026-11-03: a 5-D 3F2(1/2) analysis hung for 40 min).  The
bound stops PSLQ once its certified lower bound on every relation's norm exceeds the
largest norm LIReC itself would accept (``roi >= min_roi``).

Coverage:
  (The regression on the real hanging trajectory is ``test_lirec_hang_regression.py``.)
  - Soundness of the bound: a relation whose norm exceeds it has ``roi < min_roi``, so
    LIReC rejects it anyway (the bound never cuts an acceptable relation).
  - Equivalence: identifications that LIReC finds are identical with the bound on and off.
  - PSLQ calls with an explicit ``maxcoeff`` and calls outside ``poly_check`` are untouched.
  - ``install`` / ``uninstall`` are idempotent and restore LIReC's functions.
"""

import math

import mpmath as mp
import pytest

import LIReC.lib.pslq_utils as pu
from LIReC.db.access import db
from LIReC.lib.pslq_utils import PolyPSLQRelation, PreciseConstant

from dreamer.utils.constants import lirec_pslq_bound as bound


@pytest.fixture
def unbounded():
    """LIReC exactly as shipped for the duration of a test, then the bound back."""
    bound.uninstall()
    yield
    bound.install()


# ---------------------------------------------------------------------------
# Soundness of the bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, digits, min_roi", [(2, 1000, 2), (7, 1000, 2), (4, 50, 2), (3, 15, 3)])
def test_relations_beyond_the_bound_are_rejected_by_lirec(n, digits, min_roi):
    """A relation with ‖m‖₂ just above the bound has roi < min_roi (LIReC rejects it)."""
    limit = bound.acceptance_norm_bound(n, digits, min_roi)
    with mp.workdps(digits + 20):
        consts = [PreciseConstant(mp.mpf(k + 2) ** 0.5 + mp.pi * k, digits, f"c{k}") for k in range(n)]
    # Smallest-substance vector with norm > limit: one large coefficient, the rest zero
    # but one (a relation needs >= 2 non-zero terms).
    big = limit + 1
    coeffs = [big, 1] + [0] * (n - 2)
    rel = PolyPSLQRelation(consts, 1, 1, coeffs)
    assert math.isqrt(sum(c * c for c in coeffs)) >= limit
    assert rel.roi < min_roi


def test_bound_formula():
    b = bound.acceptance_norm_bound(2, 1000, 2)
    bits = math.floor(1000 * math.log2(10))
    assert b >= math.isqrt(2 * 2 ** (2 * (bits // 2)))      # ≥ √n · 2^(B/min_roi)
    assert bound.acceptance_norm_bound(2, 10, 2) == bound.acceptance_norm_bound(2, 15, 2)  # LIReC floors at 15 digits


# ---------------------------------------------------------------------------
# Equivalence on identifications LIReC does find
# ---------------------------------------------------------------------------

def _known_inputs():
    """1000-digit inputs: six LIReC identifies (linear, Möbius, three constants) and one
    it does not (quadratic) — both outcomes must be unchanged by the bound."""
    with mp.workdps(1050):
        l2, z3, pi, e = mp.log(2), mp.zeta(3), mp.pi, mp.e
        cases = [
            [l2, 3 * l2 + mp.mpf(5) / 7],
            [pi, (2 * pi + 1) / (pi - 3)],
            [l2, (5 * l2 - 2) / (3 * l2 + 1)],
            [z3, (z3 - 1) / 4],
            [l2, z3, 2 * l2 - 3 * z3 + 1],
            [e, (e + 1) / (2 * e - 5)],
            [l2, l2 * l2 + l2],
        ]
        return [[mp.nstr(v, 1000) for v in vals] for vals in cases]


def _identify_text(values):
    return [str(r) for r in db.identify(values)]


def test_found_identifications_identical_with_and_without_bound():
    inputs = _known_inputs()
    with_bound = [_identify_text(v) for v in inputs]
    bound.uninstall()
    try:
        without = [_identify_text(v) for v in inputs]
    finally:
        bound.install()
    assert with_bound == without
    assert sum(1 for r in with_bound if r) == 6, "LIReC should identify six of the inputs"


# ---------------------------------------------------------------------------
# Scope of the patch
# ---------------------------------------------------------------------------

def test_explicit_maxcoeff_and_outside_poly_check_untouched(monkeypatch):
    seen = []
    monkeypatch.setattr(bound, "_REAL_PSLQ", lambda x, tol, maxcoeff, maxsteps, verbose: seen.append(maxcoeff))
    pu.pslq([1, 2], None, 1000, 100, False)                  # explicit maxcoeff
    pu.pslq([1, 2], None, mp.inf, mp.inf, False)            # ∞ but not inside poly_check
    assert seen == [1000, mp.inf]


def test_install_is_idempotent_and_uninstall_restores():
    bound.install()
    bound.install()
    assert pu.pslq is bound._bounded_pslq and pu.poly_check is bound._bounded_poly_check
    bound.uninstall()
    try:
        assert pu.pslq is bound._REAL_PSLQ and pu.poly_check is bound._REAL_POLY_CHECK
    finally:
        bound.install()
