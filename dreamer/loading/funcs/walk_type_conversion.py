"""Walk-type conversion: a type-2 ray is a type-1 ray of the formal-adjoint CMF.

RD's mathematics is built on **type-1** walks: the inverse transpose ``W^-T`` of the
walked matrix ``W = M(s) M(s+d) ... M(s+(N-1)d)``.  Some CMFs and rays in the literature
are given as **type-2** walks (``W`` itself).  This module converts such a request into
the equivalent type-1 request, so RD searches it with its type-1 machinery.

Why it works.  The dual system ``Z_{n+1} = M_n^{-T} Z_n`` keeps ``Z_n^T Y_n`` constant; it
is the discrete Lagrange adjoint.  So the type-1 walk of the CMF of an operator ``L`` is
the type-2 walk of the CMF of its formal adjoint ``L*`` and vice versa.  For MeijerG and
pFq, ``L*`` (conjugated by ``z``) is again an operator of the same family:

* **MeijerG(m, n, p, q, z)** — ``a' = 1 − a``, ``b' = −b``, ``z' = (−1)^(p+q) z``;
* **pFq(p, q, z)** — ``x' = 1 − x``, ``y' = 2 − y``, ``z' = (−1)^(p+q+1) z``
  (``z' = z`` for p = q + 1; the partner of an odd-p+q MeijerG, or of 1F1, lies at ``−z``);

and every direction flips sign (``d' = −d``), because a parameter that grows along the
ray grows the other way after ``a ↦ 1 − a``.  ``m, n, p, q`` do not change.  The pFq map
equals the route pFq → MeijerG ``G^{1,p}_{p,q+1}(−z | 1−a; 0, 1−b)`` → MeijerG adjoint →
back to pFq (:func:`pfq_adjoint_via_meijer_g`); the tests check both agree.

Both maps are **involutions**: applying the map to a type-1 ray of the adjoint CMF gives
back the type-2 ray, so the same functions also convert type 1 → type 2.

References: exact operator check and derivation in
``context/investigations/pfq_conversion_explainer/`` (``adjoint_check.py``,
``pfq_type_conversion.html``); depth study ``context/investigations/pfq_conversion_depth.md``.
The user's MG44 snippet (``context/snippets/type_conversion.py``) is the same map plus a
gauge shift that sets b0 = 0 (see ``tests/test_walk_type_snippet.py``).

Where it is used.  A ``pFq`` / ``MeijerG`` formatter whose resolved walk type is 2
(``use_inv_t=False``, or ``None`` with the global ``search.DEFAULT_USES_INV_T = False``)
returns from ``to_cmf()`` the **adjoint** CMF walked as type 1
(:func:`convert_type2_request`): converted z, shifts, selected start points and
trajectories, and a :class:`Type2Source` that maps every walked ray back.  Stored rows of
such a CMF carry their type-2 coordinates under ``"type2"``.  Other CMF families have no
map and are walked natively as type 2.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import sympy as sp

#: A point or direction in a CMF's axis order (MeijerG: a's then b's; pFq: x's then y's).
Coords = Tuple[sp.Expr, ...]

_n = sp.Symbol("n")


class WalkTypeConversionError(ValueError):
    """A converted type-1 ray cannot be walked (its start lies on a pole or a singular step)."""


# ---------------------------------------------------------------------------
# Parameter maps (pure, exact)
# ---------------------------------------------------------------------------

def _checked(p: int, q: int, start: Sequence, direction: Sequence) -> Tuple[Coords, Coords]:
    """Validate lengths and sympify both vectors."""
    if len(start) != p + q or len(direction) != p + q:
        raise ValueError(
            f"start and direction must have p + q = {p + q} coordinates, got {len(start)} and {len(direction)}"
        )
    return tuple(sp.sympify(v) for v in start), tuple(sp.sympify(v) for v in direction)


def meijer_g_adjoint(p: int, q: int, z, start: Sequence, direction: Sequence) -> Tuple[sp.Expr, Coords, Coords]:
    """Formal-adjoint ray of a MeijerG(m, n, p, q, z) ray (type 2 ↔ type 1).

    ``a' = 1 − a``, ``b' = −b``, ``d' = −d``, ``z' = (−1)^(p+q) z`` (``m, n`` unchanged).
    The map is an involution.

    :param p: Number of ``a`` parameters (axes ``a0 … a{p−1}``).
    :param q: Number of ``b`` parameters (axes ``b0 … b{q−1}``).
    :param z: The CMF's ``z``.
    :param start: Start point, ``p + q`` coordinates (a's then b's).
    :param direction: Direction, ``p + q`` coordinates.
    :return: ``(z', start', direction')``.
    :raises ValueError: If a vector does not have ``p + q`` coordinates.
    """
    start, direction = _checked(p, q, start, direction)
    new_start = tuple(1 - v for v in start[:p]) + tuple(-v for v in start[p:])
    return (-1) ** (p + q) * sp.sympify(z), new_start, tuple(-v for v in direction)


def pfq_adjoint(p: int, q: int, z, start: Sequence, direction: Sequence) -> Tuple[sp.Expr, Coords, Coords]:
    """Formal-adjoint ray of a pFq(p, q, z) ray (type 2 ↔ type 1), the direct map.

    ``x' = 1 − x``, ``y' = 2 − y``, ``d' = −d``, ``z' = (−1)^(p+q+1) z``.  The map is an
    involution.

    :param p: Number of numerator parameters (axes ``x0 … x{p−1}``).
    :param q: Number of denominator parameters (axes ``y0 … y{q−1}``).
    :param z: The CMF's ``z``.
    :param start: Start point, ``p + q`` coordinates (x's then y's).
    :param direction: Direction, ``p + q`` coordinates.
    :return: ``(z', start', direction')``.
    :raises ValueError: If a vector does not have ``p + q`` coordinates.
    """
    start, direction = _checked(p, q, start, direction)
    new_start = tuple(1 - v for v in start[:p]) + tuple(2 - v for v in start[p:])
    return (-1) ** (p + q + 1) * sp.sympify(z), new_start, tuple(-v for v in direction)


def pfq_to_meijer_g(p: int, q: int, z, start: Sequence, direction: Sequence):
    """pFq ray → the MeijerG ray of ``pFq(a; b; z) ∝ G^{1,p}_{p,q+1}(−z | 1 − a; 0, 1 − b)``.

    :return: ``((m, n, P, Q), z_G, start_G, direction_G)`` with ``m = 1, n = P = p, Q = q + 1``;
        the extra ``b0`` axis is fixed at 0 (start and direction).
    """
    start, direction = _checked(p, q, start, direction)
    start_g = tuple(1 - v for v in start[:p]) + (sp.Integer(0),) + tuple(1 - v for v in start[p:])
    direction_g = tuple(-v for v in direction[:p]) + (sp.Integer(0),) + tuple(-v for v in direction[p:])
    return (1, p, p, q + 1), -sp.sympify(z), start_g, direction_g


def meijer_g_to_pfq(params, z_g, start_g: Sequence, direction_g: Sequence):
    """Inverse of :func:`pfq_to_meijer_g`.

    :param params: ``(m, n, P, Q)``; must be ``(1, P, P, Q)``.
    :return: ``(p, q, z, start, direction)`` of the pFq ray.
    :raises ValueError: If the MeijerG ray is not of the ``G^{1,p}_{p,q+1}`` form with ``b0 = 0``.
    """
    m, n, big_p, big_q = params
    start_g, direction_g = _checked(big_p, big_q, start_g, direction_g)
    if (m, n) != (1, big_p) or start_g[big_p] != 0 or direction_g[big_p] != 0:
        raise ValueError("not a pFq ray: need G^{1,p}_{p,q+1} with b0 = 0 and db0 = 0")
    p, q = big_p, big_q - 1
    start = tuple(1 - v for v in start_g[:p]) + tuple(1 - v for v in start_g[p + 1:])
    direction = tuple(-v for v in direction_g[:p]) + tuple(-v for v in direction_g[p + 1:])
    return p, q, -sp.sympify(z_g), start, direction


def pfq_adjoint_via_meijer_g(p: int, q: int, z, start: Sequence, direction: Sequence):
    """The mathematician's route: pFq → MeijerG → MeijerG adjoint → back to pFq.

    Equal to :func:`pfq_adjoint` (checked in the tests); kept as the reference derivation.

    :return: ``(z', start', direction')``.
    """
    params, z_g, s_g, d_g = pfq_to_meijer_g(p, q, z, start, direction)
    z_g1, s_g1, d_g1 = meijer_g_adjoint(params[2], params[3], z_g, s_g, d_g)
    _, _, z1, s1, d1 = meijer_g_to_pfq(params, z_g1, s_g1, d_g1)
    return z1, s1, d1


# ---------------------------------------------------------------------------
# Walkability of a converted ray
# ---------------------------------------------------------------------------

def _nonneg_integer_roots(poly_expr) -> set:
    """Integer roots n ≥ 0 of a polynomial in ``n`` (exact, via sympy's integer root filter)."""
    from sympy.polys.polyroots import roots

    poly = sp.Poly(poly_expr, _n)
    if poly.is_zero or poly.degree() <= 0:
        return set()
    return {int(r) for r in roots(poly, filter="Z") if r >= 0}


def bad_type1_steps(cmf, start: Sequence, direction: Sequence) -> Tuple[List[int], bool]:
    """Steps ``n ≥ 0`` at which a type-1 walk of the ray breaks.

    A walk multiplies ``M(n) = M(start + n·direction)`` for ``n = 0, 1, …``; type 1 then
    inverts the product, so it needs every ``M(n)`` finite (no pole in an entry) and
    invertible (non-zero determinant).  Same rule as the Zeus check
    ``mg44_conversion_02-11-26/mg44_conv.py::regular_shift``.  The determinant is taken of
    the polynomial matrix ``L·M(n)`` (``L`` = lcm of the entry denominators, Berkowitz) —
    ramanujantools' rational-function ``det()`` takes minutes already for 3F2.

    :param cmf: A ramanujantools CMF.
    :param start: Start point in the CMF's axis order.
    :param direction: Direction in the CMF's axis order.
    :return: ``(steps, singular_everywhere)``: the sorted bad steps, and ``True`` when the
        determinant vanishes identically (no start along this ray is walkable).
    """
    from ramanujantools import Position

    axes = list(cmf.matrices.keys())
    m = cmf.trajectory_matrix(
        Position(dict(zip(axes, map(sp.sympify, direction)))),
        Position(dict(zip(axes, map(sp.sympify, start)))),
    )
    entries = [sp.cancel(sp.together(e)) for e in m]
    denominators = [sp.denom(e) for e in entries]
    bad = set()
    for den in denominators:
        bad |= _nonneg_integer_roots(den)
    lcm = sp.lcm_list(denominators)
    poly_matrix = sp.Matrix(m.rows, m.cols, [sp.cancel(e * lcm) for e in entries])
    det = sp.cancel(sp.expand(poly_matrix.det(method="berkowitz")) / lcm ** m.rows)
    det_num = sp.numer(det)
    if det_num == 0:
        return sorted(bad), True
    bad |= _nonneg_integer_roots(det_num)
    return sorted(bad), False


def check_type1_walkable(cmf, start: Sequence, direction: Sequence, *, context: str = "") -> None:
    """Raise :class:`WalkTypeConversionError` unless the type-1 walk of the ray is regular.

    RD does **not** move the start on its own: a converted seed on a pole is reported
    (with the smallest shift ``k`` such that ``start + k·direction`` is walkable) so the
    user decides.

    :param cmf: A ramanujantools CMF.
    :param start: Start point in the CMF's axis order.
    :param direction: Direction in the CMF's axis order.
    :param context: Text prepended to the error message (e.g. the type-2 seed).
    :raises WalkTypeConversionError: On a pole or a singular step at some ``n ≥ 0``.
    """
    steps, singular = bad_type1_steps(cmf, start, direction)
    where = f"{context}: " if context else ""
    ray = f"type-1 start {tuple(start)}, direction {tuple(direction)} in {cmf!r}"
    if singular:
        raise WalkTypeConversionError(
            f"{where}the converted {ray} is singular at every step (det M(n) ≡ 0); it cannot be walked as type 1."
        )
    if steps:
        raise WalkTypeConversionError(
            f"{where}the converted {ray} hits a pole or a singular matrix at step(s) n = {steps} "
            f"(n = 0 is the start). It becomes walkable from start + k·direction with k = {steps[-1] + 1}; "
            f"RD does not shift it on its own — move the seed if that is intended."
        )


# ---------------------------------------------------------------------------
# Provenance: map walked rays back to type 2
# ---------------------------------------------------------------------------

def _jsonable(v):
    v = sp.sympify(v)
    return int(v) if v.is_Integer else str(v)


@dataclass(frozen=True)
class Type2Source:
    """The type-2 request a converted (type-1) CMF came from.

    :ivar family: ``"pFq"`` or ``"MeijerG"``.
    :ivar params: ``(p, q)`` for pFq, ``(m, n, p, q)`` for MeijerG.
    :ivar z: The type-2 CMF's ``z`` (the walked CMF has the adjoint ``z'``).
    """

    family: str
    params: Tuple[int, ...]
    z: sp.Expr

    @property
    def _pq(self) -> Tuple[int, int]:
        return self.params[-2], self.params[-1]

    def _map(self, z, start, direction):
        p, q = self._pq
        fn = pfq_adjoint if self.family == "pFq" else meijer_g_adjoint
        return fn(p, q, z, start, direction)

    @property
    def adjoint_z(self) -> sp.Expr:
        """``z'`` of the walked (type-1) CMF."""
        p, q = self._pq
        return self._map(self.z, (0,) * (p + q), (0,) * (p + q))[0]

    def to_type1(self, start: Sequence, direction: Sequence) -> Tuple[Coords, Coords]:
        """Type-2 ray → the walked type-1 ray."""
        _, s, d = self._map(self.z, start, direction)
        return s, d

    def to_type2(self, start: Sequence, direction: Sequence) -> Tuple[Coords, Coords]:
        """Walked type-1 ray → the requested type-2 ray (the inverse; the map is an involution)."""
        _, s, d = self._map(self.adjoint_z, start, direction)
        return s, d

    def cmf_label(self) -> str:
        """The type-2 CMF, e.g. ``"pFq(2, 1, -1/2)"``."""
        return f"{self.family}({', '.join(str(v) for v in self.params)}, {self.z})"

    def row_entry(self, start: Sequence, direction: Sequence) -> dict:
        """JSON-safe type-2 coordinates of a walked ray, stored in a row under ``"type2"``."""
        s, d = self.to_type2(start, direction)
        return {"cmf": self.cmf_label(), "start_point": [_jsonable(v) for v in s],
                "direction": [_jsonable(v) for v in d]}


# ---------------------------------------------------------------------------
# Formatter → CMFData
# ---------------------------------------------------------------------------

def _source_of(formatter) -> Type2Source:
    from dreamer.loading.funcs.meijerG_fmt import MeijerG
    from dreamer.loading.funcs.pFq_fmt import pFq

    if isinstance(formatter, MeijerG):
        return Type2Source("MeijerG", (formatter.m, formatter.n, formatter.p, formatter.q), sp.sympify(formatter.z))
    if isinstance(formatter, pFq):
        return Type2Source("pFq", (formatter.p, formatter.q), sp.sympify(formatter.z))
    raise TypeError(f"no walk-type conversion for {type(formatter).__name__}")


def convert_type2_request(formatter):
    """``CMFData`` of the type-1 formal-adjoint CMF of a type-2 ``pFq`` / ``MeijerG`` formatter.

    Converts ``z``, the shifts (``s ↦ −s``: a start is an integer point plus the shift, and
    ``1 − x``, ``2 − y``, ``−b`` keep integer offsets integer), every selected start point,
    and every selected trajectory.  Each seed with a trajectory is checked with
    :func:`check_type1_walkable`.  The result is walked as type 1 (``use_inv_t=True``), has
    the adjoint CMF's ``cmf_name`` (so it shares shards and stored rows with type-1 runs of
    that CMF), and carries a :class:`Type2Source` in ``type2_source``.

    :param formatter: A ``pFq`` or ``MeijerG`` formatter whose walk type resolves to 2.
    :return: The converted ``CMFData``.
    :raises WalkTypeConversionError: If a converted seed starts on a pole / singular step.
    """
    from dreamer.loading.funcs.meijerG_fmt import MeijerG
    from dreamer.utils.logger import Logger

    src = _source_of(formatter)
    p, q = src._pq
    zero = (0,) * (p + q)

    shifts = formatter.shifts
    shift_values = list(shifts.values()) if hasattr(shifts, "values") else list(shifts)
    new_shifts = [-sp.sympify(s) for s in shift_values]

    points = formatter.selected_start_points
    trajectories = formatter.selected_trajectories
    new_points = None
    new_trajectories = None
    if points:
        pairs = list(zip(points, trajectories if trajectories is not None else [None] * len(points)))
        new_points = [src.to_type1(pt, zero)[0] for pt, _ in pairs]
        if trajectories is not None:
            new_trajectories = [None if t is None else src.to_type1(zero, t)[1] for _, t in pairs]

    common = dict(
        shifts=new_shifts, selected_start_points=new_points, only_selected=formatter.only_selected,
        use_inv_t=True, selected_trajectories=new_trajectories,
    )
    if isinstance(formatter, MeijerG):
        adjoint = MeijerG(formatter.consts, formatter.m, formatter.n, formatter.p, formatter.q, src.adjoint_z,
                          **common)
    else:
        adjoint = type(formatter)(formatter.consts, formatter.p, formatter.q, src.adjoint_z, **common)
    data = adjoint.to_cmf()

    if new_points and new_trajectories:
        for pt2, t2, pt1, t1 in zip(points, trajectories, new_points, new_trajectories):
            if t1 is not None:
                check_type1_walkable(data.cmf, pt1, t1,
                                     context=f"type-2 seed {tuple(pt2)} / {tuple(t2)} of {src.cmf_label()}")

    Logger(
        f"{src.cmf_label()} requested as walk type 2 → walked as type 1 on its formal adjoint "
        f"{type(formatter).__name__}(…, {src.adjoint_z}) ({adjoint.cmf_name}); seeds "
        f"{[tuple(x) for x in points or []]} → {[tuple(x) for x in new_points or []]}",
        Logger.Levels.info,
    ).log()
    return dataclasses.replace(data, type2_source=src)
