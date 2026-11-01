"""Identification of slow-converging trajectories (adaptive LIReC precision).

LIReC accepts only relations that hold to the precision of its inputs.  RD used
to give it a fixed ``CONSTANT_NO_DIGITS_LOW_RES`` (1000) digits, so a trajectory
with fewer correct digits at the identification depth never identified — e.g.
the paper's (arXiv:2610.27799, appendix table) Euler–Gompertz and Euler-γ dual
walks, δ = −0.979 / −0.940, have only ~54 / ~188 correct digits at depth 1000.

The fix keeps the 1000-digit attempt and, only when it fails, retries with
``IDENTIFY_LIREC_DIGITS_FRACTION`` of the walk's own correct digits (estimated
from the column agreement between depth 0.9N and N).  The held-out digits verify
the relation.  These tests pin:

  - the paper's seeds identify, with the paper's δ;
  - a trajectory that identifies at 1000 digits takes exactly the old path;
  - a target that the walk does not converge to — even one equal to the true
    limit to 35 digits — is not identified (the held-out digits reject it);
  - walks with too few correct digits never reach the low-precision retry.
"""

import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq
from ramanujantools.cmf.meijer_g import MeijerG

from dreamer.configs import config
from dreamer.utils.storage import trajectory_attributes as ta
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

GOMPERTZ = sp.E * sp.expint(1, 1)


def _handler(cmf, constant, start, direction, depth=1000, walk_type=1):
    syms = list(cmf.matrices.keys())
    return TrajectoryAttributesHandler.from_cmf(
        cmf, Position(dict(zip(syms, map(sp.Rational, direction)))),
        Position(dict(zip(syms, map(sp.Rational, start)))),
        constant=constant, walk_depth=depth, walk_type=walk_type,
    )


def _gompertz(constant=GOMPERTZ, depth=1000):
    """Paper seed: G^{0,2}_{2,1}(1), start (1,1,0), direction (1,1,0), dual walk."""
    return _handler(MeijerG(0, 2, 2, 1, 1), constant, (1, 1, 0), (1, 1, 0), depth)


def _gamma():
    """Paper seed: G^{3,2}_{2,3}(1), start 0, direction (0,0,1,1,1), dual walk."""
    return _handler(MeijerG(3, 2, 2, 3, 1), sp.EulerGamma, (0,) * 5, (0, 0, 1, 1, 1))


@pytest.fixture
def lirec_calls(monkeypatch):
    """Record the precision (significant digits of the target) of every LIReC call."""
    calls = []
    real = ta.db.identify

    def recording(values, *args, **kwargs):
        calls.append(len(str(values[0]).replace('.', '').replace('-', '').lstrip('0')))
        return real(values, *args, **kwargs)

    monkeypatch.setattr(ta.db, "identify", recording)
    return calls


@pytest.mark.parametrize("make, paper_delta", [(_gompertz, -0.979), (_gamma, -0.940)])
def test_paper_seeds_identify_with_paper_delta(make, paper_delta):
    h = make()
    assert h.identified()
    assert abs(h.delta() - paper_delta) < 5e-3


def test_gompertz_relation_is_the_paper_form():
    h = _gompertz()
    p, q = h.p_vector(), h.q_vector()
    # G = c0 / c1 on the normalised column (c0 = 1): p = e0, q = e1 up to a common scale.
    assert p[1:] == [0] * (len(p) - 1) and q[0] == 0 and q[2:] == [0] * (len(q) - 2)


def test_low_precision_retry_uses_part_of_the_walk_digits(lirec_calls):
    h = _gompertz()
    assert h.identified()
    full, retry = lirec_calls[0], lirec_calls[1]
    assert full >= config.search.CONSTANT_NO_DIGITS_LOW_RES
    walk_digits = h._walk_correct_digits(1000)
    assert walk_digits is not None and walk_digits < 60    # ~54 true correct digits
    assert retry <= config.search.IDENTIFY_LIREC_DIGITS_FRACTION * walk_digits + 1


def test_fast_converging_trajectory_takes_the_unchanged_path(lirec_calls):
    """log(2) at depth 300 identifies at 1000 digits: one LIReC call, no retry."""
    cmf = rt_pFq(2, 1, -1)
    h = _handler(cmf, sp.log(2), (-1, 2, 1), (-4, 8, 5), depth=300)
    assert h.identified()
    assert len(lirec_calls) == 1
    assert lirec_calls[0] >= config.search.CONSTANT_NO_DIGITS_LOW_RES


@pytest.mark.parametrize("wrong", [
    sp.EulerGamma,
    sp.Catalan,
    GOMPERTZ + sp.Rational(1, 10 ** 35),   # equals the true limit to 35 of its ~54 digits
])
def test_wrong_target_is_not_identified(wrong):
    assert not _gompertz(constant=wrong).identified()


@pytest.mark.timeout(60)
def test_digit_estimate_stays_numeric():
    """The digit estimate of a walk whose column changes by a huge exact rational must not go
    symbolic (sympy's exact floor(log(...)) factorises the integers and never returns)."""
    h = _handler(MeijerG(0, 2, 2, 1, 1), GOMPERTZ, (0, 0, 0), (-2, -1, 0), walk_type=2)
    digits = h._walk_correct_digits(1000)
    assert digits is None or isinstance(digits, int)
    assert not h.identified()


def test_too_few_digits_skip_the_retry(lirec_calls, monkeypatch):
    monkeypatch.setattr(config.search, "IDENTIFY_MIN_DIGITS", 10 ** 6)
    assert not _gompertz().identified()
    assert len(lirec_calls) == 1
