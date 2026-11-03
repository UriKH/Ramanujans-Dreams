"""
Finite PSLQ inside LIReC: stop once no relation LIReC would accept can still be found.

Why
---
LIReC's ``poly_check`` runs PSLQ with ``maxcoeff = maxsteps = ∞``, so PSLQ ends only when
it returns a relation or its "bad error" failsafe fires.  On a degenerate walk column
neither happens: found 2026-11-03 on a 5-D 3F2(1/2) / log 2 trajectory whose column is
``(1, X, 7/10 − X + ε)`` with ``X ~ 1e339`` — PSLQ's certified bound plateaued after ~900
iterations and the analysis hung for 40 minutes on one core.

The bound (no new parameter)
----------------------------
* **PSLQ certificate** (Ferguson–Bailey–Arno): after every iteration, every integer
  relation ``m`` of the input satisfies ``‖m‖₂ ≥ 1 / max_j |H_jj|``.  mpmath's PSLQ (which
  LIReC copies) computes a conservative version of this bound as ``norm`` and stops when
  ``norm >= maxcoeff``.
* **LIReC's acceptance rule**: ``poly_check`` keeps a relation only if
  ``roi = precision_binary / substance >= min_roi``, where
  ``substance = #non-zero coefficients + Σ bit_length(c)  >  log₂ ‖m‖∞`` and
  ``precision_binary <= B = ⌊max(min input digits, 15) · log₂ 10⌋`` (``poly_eval`` caps it
  there).  Hence an accepted relation has ``‖m‖∞ < 2^(B / min_roi)`` and
  ``‖m‖₂ < √n · 2^(B / min_roi)``.

Setting ``maxcoeff = ⌈√n · 2^(B / min_roi)⌉ + 1`` never loses a relation LIReC accepts.
``maxcoeff`` acts twice in PSLQ: as the stop rule (``norm >= maxcoeff``) and as a filter —
each iteration returns the first column whose error is below ``tol`` *and* whose
coefficients are all below ``maxcoeff``.

* If unbounded PSLQ returns an acceptable relation ``m`` at iteration ``k``: before ``k`` no
  column passes the error test, so both runs are identical (the stop rule cannot fire,
  since ``norm <= ‖m‖₂ < maxcoeff``); at ``k`` the columns ahead of ``m`` fail the error
  test in both runs and ``m`` passes the filter — the bounded run returns the same ``m``.
* Otherwise unbounded PSLQ returns a relation LIReC rejects (``roi < min_roi``), or never
  returns; ``poly_check`` gives ``None``.  The bounded run gives ``None`` too, *or* — when the
  rejected relation was a huge chance fit found in the same iteration as a genuine small
  one — the small one, which LIReC then judges by its usual ``roi`` rule.  Seen on a
  frozen 3F2 column at 15 digits: ``[1, 1/32]`` → unbounded ``(-1.6e32, 5.2e33)`` (rejected,
  the true ``c2 = 1/32`` lost), bounded ``(1, -32)``.

So the bound only removes the infinite tail and relations LIReC would discard; on 1015
stored 2F1 / 3F2 trajectories LIReC's output was identical except for that one frozen
column (an extra, true relation; the first relation, which RD uses, unchanged).

It always triggers before precision runs out: the certificate can grow to about
``2^(working precision)``, while the stop sits at ``2^(B / min_roi)`` with ``min_roi > 1``.

Only PSLQ calls made *from* ``poly_check`` with ``maxcoeff = ∞`` are bounded (``B`` and
``min_roi`` come from that call); explicit ``maxcoeff`` values and other callers are
untouched.  :func:`install` runs at import of this module; RD imports it wherever it
calls LIReC (``trajectory_attributes``, legacy ``searchable``), so pool workers have it
too.  The proper home is LIReC itself (finite ``maxcoeff`` in ``poly_check``).
"""

from __future__ import annotations

import math
from contextvars import ContextVar
from typing import Optional, Tuple

import mpmath as mp
import LIReC.lib.pslq_utils as _pu

#: LIReC's own floor on a constant's precision (digits), see ``poly_get`` / ``poly_eval``.
_MIN_PSLQ_DPS = _pu.MIN_PSLQ_DPS

_REAL_PSLQ = _pu.pslq
_REAL_POLY_CHECK = _pu.poly_check

#: ``(precision digits B is derived from, min_roi)`` of the ``poly_check`` call in progress.
_ACTIVE: ContextVar[Optional[Tuple[int, float]]] = ContextVar("lirec_pslq_bound", default=None)


def acceptance_norm_bound(n: int, precision_digits: int, min_roi: float) -> int:
    """Smallest integer above the Euclidean norm of any relation LIReC can accept.

    ``⌈√n · 2^(B / min_roi)⌉ + 1`` with ``B = ⌊max(precision_digits, 15) · log₂ 10⌋``
    (see the module docstring for the derivation).

    :param n: Length of the PSLQ input vector.
    :param precision_digits: Smallest input precision (digits) of the ``poly_check`` call.
    :param min_roi: ``poly_check``'s acceptance threshold (> 0).
    :return: The ``maxcoeff`` to give PSLQ.
    """
    bits = math.floor(max(int(precision_digits), _MIN_PSLQ_DPS) * math.log2(10))
    exponent = bits / float(min_roi)
    whole = math.floor(exponent)
    # 2^exponent rounded up exactly enough: 2^whole · 2^frac with 2^frac < 2.
    infinity_norm = (1 << whole) * 2 if exponent > whole else (1 << whole)
    return math.isqrt(n * infinity_norm * infinity_norm) + 2


def _bounded_poly_check(consts, degree=None, order=None, exponents=None, test_prec=15,
                        min_roi=2, verbose=False):
    """``LIReC.poly_check`` with the acceptance bound active for its PSLQ call."""
    digits = min(c.precision for c in consts) if consts else _MIN_PSLQ_DPS
    token = _ACTIVE.set((int(digits), float(min_roi)))
    try:
        return _REAL_POLY_CHECK(consts, degree, order, exponents, test_prec, min_roi, verbose)
    finally:
        _ACTIVE.reset(token)


def _bounded_pslq(x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    """LIReC's ``pslq``; inside ``poly_check`` an infinite ``maxcoeff`` becomes the bound."""
    active = _ACTIVE.get()
    if active is not None and mp.isinf(maxcoeff):
        maxcoeff = acceptance_norm_bound(len(x), *active)
    return _REAL_PSLQ(x, tol, maxcoeff, maxsteps, verbose)


def install() -> None:
    """Route LIReC's ``poly_check`` / ``pslq`` through the bound (idempotent)."""
    _pu.pslq = _bounded_pslq
    _pu.poly_check = _bounded_poly_check
    try:  # pcf.py holds its own reference (PCF rationality checks)
        import LIReC.lib.pcf as _pcf
        _pcf.poly_check = _bounded_poly_check
    except ImportError:  # pragma: no cover — optional LIReC module
        pass


def uninstall() -> None:
    """Restore LIReC's own ``poly_check`` / ``pslq`` (for tests and comparisons)."""
    _pu.pslq = _REAL_PSLQ
    _pu.poly_check = _REAL_POLY_CHECK
    try:
        import LIReC.lib.pcf as _pcf
        _pcf.poly_check = _REAL_POLY_CHECK
    except ImportError:  # pragma: no cover
        pass


install()
