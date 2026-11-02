"""δ of a negative target constant must equal δ of its absolute value.

LIReC identifies a *signed* form: p·v/q·v → C, checked against the signed constant.
The δ, correct-digits and shard-cache code compared ``|p·v/q·v|`` with the signed
constant instead, so for C < 0 the error was ≈ 2|C| and δ came out ≈ −1
(run-spec row #5, L(2,χ₋₃) − ⅔ζ(2) < 0).  The rational p·v/q·v for −C is the one for
C up to sign — same reduced denominator, same |error| — so every quantity below must
agree exactly between C and −C.
"""

import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf.meijer_g import MeijerG

from dreamer.utils.storage.frequency_list import FrequencyList
from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler

GOMPERTZ = sp.E * sp.expint(1, 1)


def _gompertz(constant, searchable=None):
    """Paper seed: G^{0,2}_{2,1}(1), start (1,1,0), direction (1,1,0), dual walk, depth 1000."""
    cmf = MeijerG(0, 2, 2, 1, 1)
    syms = list(cmf.matrices.keys())
    pos = Position(dict(zip(syms, map(sp.Rational, (1, 1, 0)))))
    return TrajectoryAttributesHandler.from_cmf(
        cmf, pos, pos, constant=constant, walk_depth=1000, walk_type=1, searchable=searchable,
    )


class _Shard:
    """The part of a shard the handler uses: one p/q cache shared by all its constants."""

    def __init__(self):
        self.cache = FrequencyList(max_size=100)


@pytest.mark.parametrize("positive", [GOMPERTZ, 1 - GOMPERTZ], ids=["G", "1-G"])
def test_negated_constant_has_the_same_delta(positive):
    pos, neg = _gompertz(positive), _gompertz(-positive)
    assert pos.identified() and neg.identified()
    assert neg.delta() == pos.delta()
    assert neg.digits_computed() == pos.digits_computed()
    assert neg.delta_sequence([500, 1000]) == pos.delta_sequence([500, 1000])
    assert abs(pos.delta() + 0.979) < 5e-3   # the paper's δ


def test_negated_constant_passes_the_convergence_check():
    assert _gompertz(-GOMPERTZ)._convergence_sanity_check()[0]


def test_shard_cache_does_not_match_the_opposite_sign():
    """A form converging to −C, cached by the shard for target −C, must not be reused for C."""
    shard = _Shard()
    neg = _gompertz(-GOMPERTZ, searchable=shard)
    assert neg.identified()
    pos = _gompertz(GOMPERTZ, searchable=shard)
    assert pos.identified()
    p, q = sp.Matrix(pos.p_vector()).T, sp.Matrix(pos.q_vector()).T
    walk = sp.Matrix(pos._effective_walk_values(1000)[0])
    assert sp.Rational(p.dot(walk), q.dot(walk)) > 0
    assert pos.delta() == neg.delta()
