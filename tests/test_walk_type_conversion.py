"""Walk-type conversion: a type-2 request is walked as type 1 on the formal-adjoint CMF.

RD's mathematics is built on type-1 walks (``W^-T``).  Some CMFs and rays in the
literature are given as type-2 walks (``W``); the type-2 walk of the CMF of an
operator L is the type-1 walk of the CMF of its formal adjoint L*, which for
MeijerG and pFq is again a CMF of the same family (REQUESTS R1):

* MeijerG(m, n, p, q, z):  a' = 1 − a,  b' = −b,     z' = (−1)^(p+q) z;
* pFq(p, q, z):            x' = 1 − x,  y' = 2 − y,  z' = (−1)^(p+q+1) z;

and every direction flips sign.  Both maps are involutions (type 1 → type 2 is the
same map).  Derivation and exact operator check:
``context/investigations/pfq_conversion_explainer/``.

Categories:
  - Maps: exact values, z sign for odd/even p + q, involution, pFq direct ≡ via MeijerG.
  - Known answers (ramanujantools walks + RD's handler, depth 1000):
      * the 14 MeijerG rays verified on 2026-11-01 (𝒢 G^{0,2}_{2,1}, γ G^{3,2}_{2,3}, both odd
        p + q → partner at −z): same identification on both sides, equal δ where identified;
      * run-spec rows 9–16 (2F1, type F): converted D δ = F δ (≤ 7e−3; values of
        ``investigations/pfq_conversion_depth.md``), and the converted seeds equal the seeds
        converted by hand for the Zeus runs (``2F1-*_D_SA_02-11-26/run_rd.py``);
      * 1F1 (odd p + q): the type-2 limit and the converted limit are Möbius-related at −z,
        not at +z (control).
  - Poles: row 17's converted start lies on a pole → a clear error, not a silent walk.
  - Round trip type 2 → 1 → 2 is exact.
  - Wiring: a type-2 MeijerG / pFq formatter yields the adjoint CMF walked as type 1, with
    converted seeds and shifts; other families stay a native type-2 walk; stored rows carry
    the type-2 coordinates.
"""

import mpmath as mp
import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq
from ramanujantools.cmf.meijer_g import MeijerG as rt_MeijerG

from dreamer import log
from dreamer.configs import config
from dreamer.extraction.shard import Shard
from dreamer.loading import BaseCMF, MeijerG, pFq
from dreamer.loading.funcs.walk_type_conversion import (
    Type2Source,
    WalkTypeConversionError,
    check_type1_walkable,
    meijer_g_adjoint,
    pfq_adjoint,
    pfq_adjoint_via_meijer_g,
)
from dreamer.utils.stage_pool import StagePool, Tier1Task, run_tier1_task
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    derive_cmf_and_shard_ids,
)

R = sp.Rational
GOMPERTZ = sp.E * sp.expint(1, 1)


@pytest.fixture(autouse=True)
def _global_type1(monkeypatch):
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", True)


def _handler(cmf, start, direction, constant, depth, walk_type):
    keys = list(cmf.matrices.keys())
    return TrajectoryAttributesHandler.from_cmf(
        cmf, Position(dict(zip(keys, map(sp.sympify, direction)))),
        Position(dict(zip(keys, map(sp.sympify, start)))),
        constant=constant, walk_depth=depth, walk_type=walk_type,
    )


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------

class TestMaps:
    def test_meijer_g_values(self):
        z1, s1, d1 = meijer_g_adjoint(2, 2, 1, (R(1, 2), 3, R(2, 3), -1), (1, -2, 0, 4))
        assert z1 == 1
        assert s1 == (R(1, 2), -2, R(-2, 3), 1)
        assert d1 == (-1, 2, 0, -4)

    @pytest.mark.parametrize("p, q, sign", [(2, 1, -1), (2, 3, -1), (4, 4, 1), (2, 2, 1), (1, 2, -1)])
    def test_meijer_g_z_sign(self, p, q, sign):
        z1, _, _ = meijer_g_adjoint(p, q, R(1, 3), (0,) * (p + q), (1,) * (p + q))
        assert z1 == sign * R(1, 3)

    def test_pfq_values(self):
        z1, s1, d1 = pfq_adjoint(2, 1, R(-1, 2), (2, 2, 4), (1, 1, 2))
        assert (z1, s1, d1) == (R(-1, 2), (-1, -1, -2), (-1, -1, -2))

    @pytest.mark.parametrize("p, q, sign", [(2, 1, 1), (3, 2, 1), (1, 1, -1), (2, 2, -1), (3, 1, -1)])
    def test_pfq_z_sign(self, p, q, sign):
        z1, _, _ = pfq_adjoint(p, q, 5, (0,) * (p + q), (1,) * (p + q))
        assert z1 == sign * 5

    @pytest.mark.parametrize("p, q", [(1, 1), (2, 1), (3, 2), (2, 2), (3, 1), (4, 3)])
    def test_pfq_direct_equals_via_meijer_g(self, p, q):
        """pFq → MeijerG G^{1,p}_{p,q+1}(−z) → MeijerG adjoint → back to pFq gives the direct map."""
        start = sp.symbols(f"s0:{p + q}")
        direction = sp.symbols(f"d0:{p + q}")
        z = sp.Symbol("z")
        direct = pfq_adjoint(p, q, z, start, direction)
        via = pfq_adjoint_via_meijer_g(p, q, z, start, direction)
        assert tuple(sp.simplify(a - b) for a, b in zip(direct[1], via[1])) == (0,) * (p + q)
        assert tuple(sp.simplify(a - b) for a, b in zip(direct[2], via[2])) == (0,) * (p + q)
        assert sp.simplify(direct[0] - via[0]) == 0

    @pytest.mark.parametrize("family", ["meijer", "pfq"])
    def test_involution_round_trip(self, family):
        """Type 2 → 1 → 2 is the identity (start, direction and z)."""
        p, q = 2, 3
        start = (R(1, 2), 3, R(-2, 3), 0, 7)
        direction = (1, -2, 0, 4, 5)
        fn = meijer_g_adjoint if family == "meijer" else pfq_adjoint
        once = fn(p, q, R(-1, 3), start, direction)
        twice = fn(p, q, once[0], once[1], once[2])
        assert twice == (R(-1, 3), start, direction)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            pfq_adjoint(2, 1, 1, (1, 2), (1, 1, 1))


# ---------------------------------------------------------------------------
# Known answers: the 14 verified MeijerG rays (odd p + q → partner at −z)
# ---------------------------------------------------------------------------

def _rays(base, b0_index):
    """The base direction and every ±1 change of one axis except b0 (as on 2026-11-01)."""
    out = [list(base)]
    for i in range(len(base)):
        for sg in (-1, 1):
            d = [x + (sg if k == i else 0) for k, x in enumerate(base)]
            if i != b0_index and any(d):
                out.append(d)
    return out


# (label, m, n, p, q, constant, type-2 start, type-2 base direction). The type-2 rays are
# the partners of the paper's (arXiv:2610.27799) type-1 seeds at z = 1, so they lie at z = −1.
_MG_CASES = [
    ("gompertz", 0, 2, 2, 1, GOMPERTZ, (0, 0, 0), (-1, -1, 0)),
    ("gamma", 3, 2, 2, 3, sp.EulerGamma, (1, 1, 0, 0, 0), (0, 0, -1, -1, -1)),
]
_MG_RAYS = [(c, tuple(d)) for c in _MG_CASES for d in _rays(c[7], c[3])]


def test_fourteen_rays_are_listed():
    assert len(_MG_RAYS) == 14


@pytest.mark.parametrize("case, d2", _MG_RAYS, ids=[f"{c[0]}-{d}" for c, d in _MG_RAYS])
def test_meijer_g_ray_type2_equals_converted_type1(case, d2):
    _, m, n, p, q, constant, s2, _ = case
    z1, s1, d1 = meijer_g_adjoint(p, q, -1, s2, d2)
    assert z1 == 1                                   # odd p + q: the partner of z = −1 is z = 1
    h2 = _handler(rt_MeijerG(m, n, p, q, -1), s2, d2, constant, 1000, 2)
    h1 = _handler(rt_MeijerG(m, n, p, q, z1), s1, d1, constant, 1000, 1)
    assert h2.identified() == h1.identified()
    if h2.identified():
        assert abs(h2.delta() - h1.delta()) < 1e-3


def test_meijer_g_paper_seeds_identify():
    """The two seeds themselves identify on both sides (δ ≈ −0.9793 𝒢, −0.9397 γ)."""
    for (_, m, n, p, q, constant, s2, d2), expected in zip(_MG_CASES, (-0.97925, -0.93975)):
        z1, s1, d1 = meijer_g_adjoint(p, q, -1, s2, d2)
        h1 = _handler(rt_MeijerG(m, n, p, q, z1), s1, d1, constant, 1000, 1)
        assert h1.identified() and abs(h1.delta() - expected) < 2e-4


# ---------------------------------------------------------------------------
# Known answers: run-spec rows 9–16 (2F1, type F)
# ---------------------------------------------------------------------------

# row: (constant, z, F start, δ_F @1000, δ_D @1000 from investigations/pfq_conversion_depth.md)
_ROWS = {
    9: (sp.log(R(3, 2)), R(-1, 2), (2, 2, 4), 0.403392, 0.403418),
    10: (sp.log(R(4, 3)), R(-1, 3), (2, 2, 4), 0.460810, 0.461349),
    11: (sp.log(R(5, 4)), R(-1, 4), (1, 1, 2), 0.497181, 0.490734),
    12: (sp.log(R(6, 5)), R(-1, 5), (1, 1, 2), 0.517646, 0.519869),
    13: (sp.log(R(5, 3)), R(-2, 3), (1, 1, 2), 0.105641, 0.106881),
    14: (sp.log(R(7, 5)), R(-2, 5), (1, 1, 2), 0.194317, 0.194930),
    15: (sp.log(5), -4, (1, 1, 2), -0.420130, -0.419363),
    16: (sp.log(7), -6, (1, 1, 2), -0.553098, -0.553124),
}
_F_DIRECTION = (1, 1, 2)
#: Seeds converted by hand for the Zeus runs (run_rd.py ``f_to_d``), rows 9–10 and 11–16.
_HAND_CONVERTED = {(2, 2, 4): ((-1, -1, -2), (-1, -1, -2)), (1, 1, 2): ((0, 0, 0), (-1, -1, -2))}


@pytest.mark.parametrize("row", sorted(_ROWS))
def test_pfq_row_d_delta_equals_f_delta(row):
    constant, z, s2, delta_f, delta_d = _ROWS[row]
    z1, s1, d1 = pfq_adjoint(2, 1, z, s2, _F_DIRECTION)
    assert z1 == z                                    # p = q + 1: same z
    assert (s1, d1) == _HAND_CONVERTED[s2]
    hf = _handler(rt_pFq(2, 1, z), s2, _F_DIRECTION, constant, 1000, 2)
    hd = _handler(rt_pFq(2, 1, z1), s1, d1, constant, 1000, 1)
    assert hf.identified() and hd.identified()
    assert abs(hf.delta() - delta_f) < 1e-5 and abs(hd.delta() - delta_d) < 1e-5
    assert abs(hd.delta() - hf.delta()) < 7e-3


# ---------------------------------------------------------------------------
# Known answer: odd p + q in pFq (1F1 → partner at −z), Möbius relation of the limits
# ---------------------------------------------------------------------------

def _limit(cmf, start, direction, walk_type, depth=600):
    h = _handler(cmf, start, direction, sp.E, depth, walk_type)
    values, _ = h._effective_walk_values(depth)
    return sp.Rational(values[1])


def _mobius(l1, l2, digits=60):
    """Small-integer relation α + β L1 + γ L2 + ε L1 L2 = 0 (a Möbius map), or None."""
    with mp.workdps(digits):
        a, b = mp.mpf(l1.p) / l1.q, mp.mpf(l2.p) / l2.q
        return mp.pslq([1, a, b, a * b], maxcoeff=10 ** 4, maxsteps=10 ** 5)


@pytest.mark.parametrize("z", [-1, 2])
def test_pfq_odd_p_plus_q_partner_at_minus_z(z):
    s2, d2 = (R(1, 2), R(3, 2)), (1, 1)
    z1, s1, d1 = pfq_adjoint(1, 1, z, s2, d2)
    assert z1 == -z
    l2 = _limit(rt_pFq(1, 1, z), s2, d2, 2)
    assert _mobius(l2, _limit(rt_pFq(1, 1, z1), s1, d1, 1)) is not None
    assert _mobius(l2, _limit(rt_pFq(1, 1, -z1), s1, d1, 1)) is None      # control: wrong sign


# ---------------------------------------------------------------------------
# Poles
# ---------------------------------------------------------------------------

class TestPoles:
    def test_row17_converted_start_on_a_pole_raises(self):
        """Row 17 (3F2(1/2), F start (1,1,1;2,1), direction (7,5,6;2,1)): the D start is a pole."""
        z1, s1, d1 = pfq_adjoint(3, 2, R(1, 2), (1, 1, 1, 2, 1), (7, 5, 6, 2, 1))
        assert s1 == (0, 0, 0, 0, 1)
        with pytest.raises(WalkTypeConversionError, match="pole|singular"):
            check_type1_walkable(rt_pFq(3, 2, z1), s1, d1)

    @staticmethod
    def _capture_events(monkeypatch):
        from dreamer.utils.logger import Logger
        events = []
        monkeypatch.setattr(Logger, "event", classmethod(
            lambda cls, kind, msg, level=None, **fields: events.append((kind, msg, level, fields))))
        return events

    def test_formatter_seed_on_a_pole_is_dropped_with_event(self, monkeypatch):
        """R27: a seed on a pole is reported (warning event) and skipped; loading does not raise."""
        from dreamer.utils.logger import Logger
        events = self._capture_events(monkeypatch)
        f = pFq(log(2), 3, 2, R(1, 2), use_inv_t=False,
                selected_start_points=[(1, 1, 1, 2, 1)], selected_trajectories=[(7, 5, 6, 2, 1)])
        data = f.to_cmf()
        assert data.selected_points == [] and data.selected_trajectories == []
        (kind, msg, level, fields), = [e for e in events if e[0] == "type2_seed_unwalkable"]
        assert level == Logger.Levels.warning
        assert "(1, 1, 1, 2, 1)" in msg and "pole" in msg       # names the type-2 seed and the reason
        assert fields["type2_start"] == [1, 1, 1, 2, 1] and fields["bad_steps"] == [0]

    def test_only_the_bad_seed_is_dropped(self, monkeypatch):
        events = self._capture_events(monkeypatch)
        f = pFq(log(2), 3, 2, R(1, 2), use_inv_t=False,
                selected_start_points=[(1, 1, 1, 2, 1), (2, 2, 2, 3, 3)],
                selected_trajectories=[(7, 5, 6, 2, 1), (1, 1, 1, 1, 1)])
        data = f.to_cmf()
        assert data.selected_points == [(-1, -1, -1, -1, -1)]
        assert data.selected_trajectories == [(-1, -1, -1, -1, -1)]
        assert [e[0] for e in events].count("type2_seed_unwalkable") == 1

    def test_only_selected_without_walkable_seed_reports_no_shards(self, monkeypatch):
        events = self._capture_events(monkeypatch)
        f = pFq(log(2), 3, 2, R(1, 2), use_inv_t=False, only_selected=True,
                selected_start_points=[(1, 1, 1, 2, 1)], selected_trajectories=[(7, 5, 6, 2, 1)])
        data = f.to_cmf()
        assert data.only_selected and data.selected_points == []
        assert "type2_no_walkable_seed" in [e[0] for e in events]

    def test_pole_check_uses_absolute_coordinates(self, monkeypatch):
        """Selected points are offsets from the shift: the check walks point + shift."""
        from dreamer.loading.funcs import walk_type_conversion as wtc
        seen = []
        monkeypatch.setattr(wtc, "check_type1_walkable",
                            lambda cmf, start, direction, **kw: seen.append(tuple(start)))
        pFq(log(2), 2, 1, -1, shifts=[R(1, 2), 0, 0], use_inv_t=False,
            selected_start_points=[(1, 1, 3)], selected_trajectories=[(1, 1, 2)]).to_cmf()
        # absolute type-2 start (3/2, 1, 3) → absolute type-1 start (-1/2, 0, -1)
        assert seen == [(R(-1, 2), 0, -1)]

    def test_walkable_ray_passes(self):
        z1, s1, d1 = pfq_adjoint(2, 1, R(-1, 2), (2, 2, 4), (1, 1, 2))
        check_type1_walkable(rt_pFq(2, 1, z1), s1, d1)


# ---------------------------------------------------------------------------
# Wiring: formatters → CMFData
# ---------------------------------------------------------------------------

class TestFormatterWiring:
    def test_type2_pfq_becomes_type1_adjoint(self):
        f = pFq(log(R(3, 2)), 2, 1, R(-1, 2), use_inv_t=False,
                selected_start_points=[(2, 2, 4)], selected_trajectories=[(1, 1, 2)])
        data = f.to_cmf()
        assert data.use_inv_t is True
        assert data.cmf.z == R(-1, 2)
        assert data.selected_points == [(-1, -1, -2)]
        assert data.selected_trajectories == [(-1, -1, -2)]
        assert data.cmf_name == pFq(log(R(3, 2)), 2, 1, R(-1, 2), use_inv_t=True).cmf_name
        assert isinstance(data.type2_source, Type2Source)

    def test_type2_meijer_g_odd_moves_to_minus_z(self):
        f = MeijerG(log(2), 0, 2, 2, 1, -1, use_inv_t=False,
                    selected_start_points=[(0, 0, 0)], selected_trajectories=[(-1, -1, 0)])
        data = f.to_cmf()
        assert data.use_inv_t is True and data.cmf.z == 1
        assert (data.cmf.m, data.cmf.n, data.cmf.p, data.cmf.q) == (0, 2, 2, 1)
        assert data.selected_points == [(1, 1, 0)] and data.selected_trajectories == [(1, 1, 0)]

    def test_global_type2_converts_formatter_that_follows_it(self, monkeypatch):
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
        data = pFq(log(2), 1, 1, 2).to_cmf()
        assert data.use_inv_t is True and data.cmf.z == -2 and data.type2_source is not None

    def test_type1_is_unchanged(self):
        data = pFq(log(2), 2, 1, -1, selected_start_points=[(1, 1, 3)]).to_cmf()
        assert data.use_inv_t is None and data.type2_source is None
        assert data.selected_points == [(1, 1, 3)] and data.cmf.z == -1

    def test_shifts_and_seed_without_trajectory(self):
        f = pFq(log(2), 2, 1, -1, shifts=[R(1, 2), 0, R(1, 3)], use_inv_t=False,
                selected_start_points=[(R(3, 2), 1, R(7, 3))])
        data = f.to_cmf()
        assert list(data.shift.values()) == [R(-1, 2), 0, R(-1, 3)]
        assert data.selected_points == [(R(-1, 2), 0, R(-1, 3))]
        assert data.selected_trajectories is None

    def test_other_families_walk_type2_natively(self):
        cmf = rt_pFq(2, 1, -1)
        data = BaseCMF(log(2), "my2F1", cmf, use_inv_t=False).to_cmf()
        assert data.use_inv_t is False and data.type2_source is None

    def test_json_keeps_the_type2_request(self):
        """The exported formatter is the type-2 request (re-converted when reloaded)."""
        from dreamer.loading.funcs.formatter import Formatter
        f = pFq(log(2), 2, 1, -1, use_inv_t=False, selected_start_points=[(1, 1, 3)])
        back = Formatter.from_json_obj(f.to_json_obj())
        assert back.use_inv_t is False and back.selected_start_points == [(1, 1, 3)]
        assert back.to_cmf().selected_points == [(0, 0, -1)]


# ---------------------------------------------------------------------------
# Reporting: type-2 coordinates
# ---------------------------------------------------------------------------

class TestType2Reporting:
    def test_source_maps_back(self):
        src = pFq(log(2), 2, 1, -1, use_inv_t=False).to_cmf().type2_source
        assert src.to_type2((-1, -1, -2), (-1, -1, -2)) == ((2, 2, 4), (1, 1, 2))
        row = src.row_entry((-1, -1, -2), (-1, -1, -2))
        assert row == {"cmf": "pFq(2, 1, -1)", "start_point": [2, 2, 4], "direction": [1, 1, 2]}

    def test_stored_row_carries_type2_coordinates(self):
        data = pFq(log(R(3, 2)), 2, 1, R(-1, 2), use_inv_t=False).to_cmf()
        shard = Shard.from_cmf_data(data, log(R(3, 2)), [], [])
        assert shard.type2_source is data.type2_source
        keys = list(data.cmf.matrices.keys())
        start = Position(dict(zip(keys, map(sp.Integer, (-1, -1, -2)))))
        direction = Position(dict(zip(keys, map(sp.Integer, (-1, -1, -2)))))
        cmf_id, shard_id, enc = derive_cmf_and_shard_ids(shard)
        task = Tier1Task("k", 0, direction, start, (log(R(3, 2)),), cmf_id, shard_id, enc)
        with StagePool([shard], workers=1):
            res = run_tier1_task(task)
        (dto,) = res.dtos
        assert dto.walk_type == 1
        assert dto.extra["type2"] == {"cmf": "pFq(2, 1, -1/2)", "start_point": [2, 2, 4],
                                      "direction": [1, 1, 2]}
        assert "type2" not in dto.extra.get("fingerprints", {})
