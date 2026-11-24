"""The user's MG44 snippet (``context/snippets/type_conversion.py``) vs RD's plain map.

The snippet is copied here **verbatim** as a reference (it is not changed and not part
of RD).  It returns the formal-adjoint type-1 ray in a "canonical b0 = 0" form: on top
of RD's plain map (a' = 1 − a, b' = −b, d → −d, see
``dreamer/loading/funcs/walk_type_conversion.py``) it adds the type-2 b0 to every start
coordinate and db0 to every direction coordinate.  Adding one constant c to every
MeijerG parameter multiplies the solutions by z^c, i.e. it is the same CMF up to a
scalar gauge.  RD uses the plain map because it is an involution (exact round trip);
these tests pin the exact relation between the two (user, 2026-11-21: "Don't change
the snippet, if you want to test something … do it in another test file").
"""

import pytest
import sympy as sp
from ramanujantools import Position

from dreamer.loading.funcs.walk_type_conversion import meijer_g_adjoint

# ---- verbatim copy of context/snippets/type_conversion.py (2026-11-21) ------------------
a = sp.symbols("a0:4")
b = sp.symbols("b0:4")
AXES = (*a, *b)


def mg44_type2_to_type1(
    type2_initial: Position,
    type2_direction: Position,
) -> tuple[Position, Position]:
    """Return the canonical b0=0 MG44 formal-adjoint Type-1 ray."""

    def value(position, axis):
        # RT permits omitted initial coordinates; treat them as zero.
        return sp.sympify(position.get(axis, 0))

    source_b0 = value(type2_initial, b[0])
    source_db0 = value(type2_direction, b[0])

    type1_initial = Position({
        **{
            a[i]: 1 + source_b0 - value(type2_initial, a[i])
            for i in range(4)
        },
        **{
            b[j]: source_b0 - value(type2_initial, b[j])
            for j in range(4)
        },
    })

    type1_direction = Position({
        **{
            a[i]: source_db0 - value(type2_direction, a[i])
            for i in range(4)
        },
        **{
            b[j]: source_db0 - value(type2_direction, b[j])
            for j in range(4)
        },
    })

    return type1_initial, type1_direction
# ---- end of the verbatim copy ------------------------------------------------------------

R = sp.Rational

#: Row 5 of the run spec (type-1 seed) and its type-2 partner; the Zeus MG44 rays.
_TYPE2_START = (2, 1, -1, R(-5, 3), 0, 0, 0, R(-2, 3))
_RAYS = [
    (10, 9, -14, -6, 0, -1, -2, 2), (9, 10, -14, -6, 0, -1, -2, 2), (9, 9, -13, -6, 0, -1, -2, 2),
    (10, 9, -14, -6, 0, -1, -2, 3),
]
_GAUGED_START = (R(1, 2), 3, R(2, 3), -1, R(5, 2), 1, 0, 4)       # b0 ≠ 0
_GAUGED_RAY = (1, -2, 0, 4, 3, 1, 1, -1)                          # db0 ≠ 0


def _snippet(start, direction):
    s, d = mg44_type2_to_type1(Position(dict(zip(AXES, start))), Position(dict(zip(AXES, direction))))
    return tuple(s[x] for x in AXES), tuple(d[x] for x in AXES)


@pytest.mark.parametrize("direction", _RAYS)
def test_snippet_equals_plain_map_when_b0_is_zero(direction):
    z1, s1, d1 = meijer_g_adjoint(4, 4, 1, _TYPE2_START, direction)
    assert z1 == 1                                   # p + q = 8 even: same z
    assert _snippet(_TYPE2_START, direction) == (s1, d1)


def test_snippet_is_plain_map_plus_b0_gauge():
    _, s1, d1 = meijer_g_adjoint(4, 4, 1, _GAUGED_START, _GAUGED_RAY)
    b0, db0 = _GAUGED_START[4], _GAUGED_RAY[4]
    s_snip, d_snip = _snippet(_GAUGED_START, _GAUGED_RAY)
    assert s_snip == tuple(x + b0 for x in s1)
    assert d_snip == tuple(x + db0 for x in d1)
    assert s_snip[4] == 0 and d_snip[4] == 0         # the snippet's canonical b0 = 0
