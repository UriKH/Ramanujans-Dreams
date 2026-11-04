"""Form search: the identified (p, q) is the fastest-converging form of the walk, whatever form identification
starts from (LIReC, a cached form, or an LLL ladder when LIReC finds nothing).

On a walk column with ≥ 3 entries several integer forms converge to the constant at different rates, and RD used
to keep the first one found (LIReC's smallest-coefficient form, or the first matching cached form), so δ depended
on evaluation order and good trajectories were missed.  With ``search.IDENTIFY_FORM_SEARCH`` the form is refined by
LLL at a weight just above its current accuracy until no faster form exists.  Cases are real 3F2(½)/log 2
trajectories (walk type 1); the numbers were checked with mpmath outside RD
(``context/investigations/form_search/``).
"""
import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer.configs import config
from dreamer.utils import form_search as fs
from dreamer.utils.storage.frequency_list import FrequencyList
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

LOG2 = sp.log(2)
START_5 = (1, -1, -1, -2, -2)
START_8 = (1, -1, -1, 3, 3)


class _Shard:
    """The part of a shard the handler uses: its p/q cache."""

    def __init__(self, forms=()):
        self.cache = FrequencyList(max_size=100)
        for f in forms:
            self.cache.append(f)


def _handler(start, direction, constant=LOG2, searchable=None, cmf=None):
    cmf = cmf or rt_pFq(3, 2, sp.Rational(1, 2))
    syms = list(cmf.matrices.keys())
    pos = lambda t: Position(dict(zip(syms, map(sp.Rational, t))))
    return TrajectoryAttributesHandler.from_cmf(cmf, pos(direction), pos(start), constant=constant,
                                                walk_type=1, searchable=searchable)


@pytest.fixture
def form_search_on(monkeypatch):
    monkeypatch.setattr(config.search, "IDENTIFY_FORM_SEARCH", True)


def test_flag_off_keeps_lirecs_form():
    h = _handler(START_5, (1, 0, -1, -3, -1))
    assert h.identified() and h.delta() == pytest.approx(-0.7717, abs=1e-3)


@pytest.mark.timeout(300)
def test_search_finds_the_faster_form(form_search_on):
    h = _handler(START_5, (1, 0, -1, -3, -1))
    assert h.identified() and h.delta() == pytest.approx(-0.2252, abs=2e-3)


@pytest.mark.timeout(300)
def test_result_does_not_depend_on_the_cache(form_search_on):
    weak = ((13, -2, 0), (19, -3, 0))                  # valid on this trajectory, δ ≈ −0.928
    empty = _handler(START_5, (4, 0, 0, -4, -1), searchable=_Shard())
    cached = _handler(START_5, (4, 0, 0, -4, -1), searchable=_Shard([weak]))
    assert empty.identified() and cached.identified()
    assert empty.delta() == pytest.approx(cached.delta(), abs=1e-9)
    assert empty.delta() == pytest.approx(-0.0835, abs=2e-3)


@pytest.mark.timeout(300)
def test_trajectory_lirec_misses_is_identified(form_search_on):
    h = _handler(START_8, (0, -1, -2, 0, 0))
    assert h.identified() and h.delta() > 0.03          # δ ≈ +0.039 at depth 1500


@pytest.mark.timeout(300)
def test_final_form_holds_along_the_walk(form_search_on):
    import mpmath as mp
    h = _handler(START_8, (0, -1, -2, 0, 0))
    p, q = h.p_vector(), h.q_vector()
    digits = []
    for depth in (h.walk_depth() // 2, h.walk_depth()):
        v, _ = h._effective_walk_values(depth)
        r = sp.Rational(sum(a * b for a, b in zip(p, v)), sum(a * b for a, b in zip(q, v)))
        with mp.workdps(8000):
            digits.append(float(-mp.log10(abs(mp.mpf(r.p) / r.q - mp.log(2)))))
    assert 1.8 < digits[1] / digits[0] < 2.2


@pytest.mark.timeout(300)
def test_wrong_constant_is_not_identified(form_search_on):
    assert not _handler(START_5, (4, 0, 0, -4, -1), constant=sp.log(3)).identified()


@pytest.mark.timeout(300)
def test_rational_constant_never_runs_lll(form_search_on, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("LLL must not run for a rational target")
    monkeypatch.setattr(fs, "lll_relations", forbidden)
    _handler(START_5, (4, 0, 0, -4, -1), constant=sp.Rational(1, 3)).identified()


@pytest.mark.timeout(300)
def test_two_entry_column_is_unchanged(form_search_on, monkeypatch):
    cmf = rt_pFq(2, 1, -1)
    on = _handler((1, 1, 3), (3, 1, 3), cmf=cmf)
    monkeypatch.setattr(config.search, "IDENTIFY_FORM_SEARCH", False)
    off = _handler((1, 1, 3), (3, 1, 3), cmf=cmf)
    assert on.identified() == off.identified()
    if on.identified():
        assert on.delta() == pytest.approx(off.delta(), abs=1e-12)
