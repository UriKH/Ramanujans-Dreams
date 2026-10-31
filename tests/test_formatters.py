"""Tests for the CMF formatter classes (pFq, MeijerG, BaseCMF).

Covers:
- JSON round-trip serialization for pFq and MeijerG
- CMF creation: correct dimension, symbol count
- Formatter registry completeness
- Input validation (bad p/q values, wrong shift length)
- MeijerG super().__init__ fix verification
"""
import pytest
import sympy as sp

from dreamer.loading.funcs.formatter import Formatter
from dreamer.loading.funcs.pFq_fmt import pFq
from dreamer.loading.funcs.meijerG_fmt import MeijerG
from dreamer.loading.funcs.base_cmf import BaseCMF
from dreamer.utils.constants.constant import Constant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def log2():
    """Return or create the log-2 constant."""
    from dreamer import log
    return log(2)


@pytest.fixture
def const_e():
    from dreamer import e
    return e


# ---------------------------------------------------------------------------
# 1. Formatter registry
# ---------------------------------------------------------------------------
class TestFormatterRegistry:

    def test_all_formatters_registered(self):
        """pFq, MeijerG, and BaseCMF must be registered on import."""
        for name in ("pFq", "MeijerG", "BaseCMF"):
            assert name in Formatter.registry, f"{name} missing from registry"

    def test_fetch_from_registry(self):
        cls = Formatter.fetch_from_registry("pFq")
        assert cls is pFq

    def test_fetch_unknown_raises(self):
        with pytest.raises(KeyError):
            Formatter.fetch_from_registry("UnknownFormatter")


# ---------------------------------------------------------------------------
# 2. pFq
# ---------------------------------------------------------------------------
class TestpFq:

    def test_creation_2f1(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        assert fmt.p == 2
        assert fmt.q == 1
        assert fmt.const == "log-2"
        assert len(fmt.shifts) == 3  # p + q

    def test_default_shifts(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        assert fmt.shifts == [0, 0, 0]

    def test_custom_shifts(self, log2):
        fmt = pFq(log2, 2, 1, -1, shifts=[0, sp.Rational(1, 2), 0])
        assert fmt.shifts[1] == sp.Rational(1, 2)

    def test_shifts_normalized_to_sympy_rationals(self, log2):
        """Integer/Rational shifts are coerced to exact sympy numbers."""
        fmt = pFq(log2, 2, 1, -1, shifts=[0, 1, sp.Rational(1, 2)])
        assert all(isinstance(s, sp.Rational) for s in fmt.shifts)  # Integer ⊂ Rational
        assert fmt.shifts == [0, 1, sp.Rational(1, 2)]

    def test_float_shift_raises(self, log2):
        """A float shift is rejected with a clear error (use sp.Rational instead)."""
        with pytest.raises(ValueError, match="rational"):
            pFq(log2, 2, 1, -1, shifts=[0, 0.5, 0])

    def test_irrational_shift_raises(self, log2):
        with pytest.raises(ValueError, match="rational"):
            pFq(log2, 2, 1, -1, shifts=[0, sp.sqrt(2), 0])

    def test_json_roundtrip(self, log2):
        original = pFq(log2, 2, 1, -1)
        json_obj = original.to_json_obj()
        restored = Formatter.from_json_obj(json_obj)
        assert isinstance(restored, pFq)
        assert restored.p == original.p
        assert restored.q == original.q
        assert restored.const == original.const

    def test_to_cmf_dimension(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        shift_cmf = fmt.to_cmf()
        assert shift_cmf.cmf.dim() == 3  # 2F1 → 3 symbols

    def test_to_cmf_has_matrices(self, log2):
        fmt = pFq(log2, 2, 1, -1)
        shift_cmf = fmt.to_cmf()
        assert len(shift_cmf.cmf.matrices) == 3

    def test_invalid_p_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 0, 1, -1)

    def test_invalid_q_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 2, 0, -1)

    def test_wrong_shift_length_raises(self, log2):
        with pytest.raises(ValueError):
            pFq(log2, 2, 1, -1, shifts=[0, 0])  # needs 3, given 2

    def test_selected_trajectories_paired_with_start_points(self, log2):
        fmt = pFq(
            log2, 2, 1, -1,
            selected_start_points=[(1, 2, 3)],
            selected_trajectories=[(0, 1, 0)],
            only_selected=True,
        )
        cmf_data = fmt.to_cmf()
        assert cmf_data.selected_trajectories == [(0, 1, 0)]

    def test_selected_trajectories_length_mismatch_raises(self, log2):
        with pytest.raises(ValueError, match="selected_trajectories length"):
            pFq(log2, 2, 1, -1,
                selected_start_points=[(1, 2, 3)],
                selected_trajectories=[(0, 1, 0), (1, 0, 0)])

    def test_selected_trajectories_without_start_points_raises(self, log2):
        with pytest.raises(ValueError, match="requires selected_start_points"):
            pFq(log2, 2, 1, -1, selected_trajectories=[(0, 1, 0)])

    def test_selected_trajectories_json_roundtrip(self, log2):
        original = pFq(
            log2, 2, 1, -1,
            selected_start_points=[(1, 2, 3), (4, 5, 6)],
            selected_trajectories=[(0, 1, 0), None],
            only_selected=True,
        )
        restored = Formatter.from_json_obj(original.to_json_obj())
        assert restored.selected_trajectories == [(0, 1, 0), None]
        assert restored.selected_start_points == [(1, 2, 3), (4, 5, 6)]


# ---------------------------------------------------------------------------
# 3. MeijerG
# ---------------------------------------------------------------------------
class TestMeijerG:

    def test_creation(self, const_e):
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        assert fmt.m == 1
        assert fmt.n == 1
        assert fmt.p == 1
        assert fmt.q == 2

    def test_default_shifts_length(self, const_e):
        """Shifts should default to [0]*(p+q)."""
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        assert len(fmt.shifts) == 3  # p + q = 1 + 2

    def test_super_init_passes_correct_args(self, const_e):
        """After the fix, Formatter.__init__ should receive shifts as a list, not a bool."""
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        # The parent class (Formatter) stores shifts — check it's a list
        assert isinstance(fmt.shifts, list), (
            f"shifts should be list, got {type(fmt.shifts)}"
        )

    def test_json_roundtrip(self, const_e):
        original = MeijerG(const_e, 1, 1, 1, 2, 1)
        json_obj = original.to_json_obj()
        restored = Formatter.from_json_obj(json_obj)
        assert isinstance(restored, MeijerG)
        assert restored.m == original.m
        assert restored.p == original.p
        assert restored.q == original.q

    def test_to_cmf(self, const_e):
        fmt = MeijerG(const_e, 1, 1, 1, 2, 1)
        shift_cmf = fmt.to_cmf()
        # MeijerG(1,1,1,2) has p+q = 3 symbols
        assert shift_cmf.cmf.dim() == 3

    def test_invalid_pq_raises(self, const_e):
        with pytest.raises(ValueError):
            MeijerG(const_e, 1, 1, 0, 2, 1)


# ---------------------------------------------------------------------------
# 4. BaseCMF
# ---------------------------------------------------------------------------
class TestBaseCMF:

    def test_creation_from_raw_cmf(self, log2):
        from ramanujantools.cmf import pFq as rt_pFq
        raw_cmf = rt_pFq(2, 1, -1)
        fmt = BaseCMF(log2, 'UnknownCMF', raw_cmf)
        assert fmt.cmf is raw_cmf

    def test_json_roundtrip(self, log2):
        """to_json_obj -> from_json_obj restores matrices, axis order, shifts and name."""
        from ramanujantools.cmf import pFq as rt_pFq
        raw_cmf = rt_pFq(2, 1, -1)
        fmt = BaseCMF(log2, 'UnknownCMF', raw_cmf, shifts=[0, sp.Rational(1, 2), 0],
                      selected_start_points=[(1, sp.Rational(3, 2), 2)], use_inv_t=False)
        back = Formatter.from_json_obj(fmt.to_json_obj())

        assert isinstance(back, BaseCMF)
        assert back.cmf_name == fmt.cmf_name
        assert back.consts == fmt.consts
        assert back.shifts == fmt.shifts
        assert back.use_inv_t is False
        assert back.selected_start_points == fmt.selected_start_points
        assert list(back.cmf.matrices.keys()) == list(raw_cmf.matrices.keys())
        for axis, matrix in raw_cmf.matrices.items():
            assert sp.simplify(sp.Matrix(back.cmf.matrices[axis]) - sp.Matrix(matrix)).is_zero_matrix

    def test_json_roundtrip_through_string(self, log2):
        """The JSON object survives json.dumps / json.loads (what exports do)."""
        import json
        from ramanujantools.cmf import pFq as rt_pFq
        fmt = BaseCMF(log2, 'UnknownCMF', rt_pFq(1, 1, -1))
        back = Formatter.from_json_obj(json.loads(json.dumps(fmt.to_json_obj())))
        assert back.cmf_name == fmt.cmf_name
        assert back.to_cmf().cmf.dim() == 2

    def test_legacy_layout_without_name_raises_clear_error(self, log2):
        """Old exports (matrices as top-level srepr keys, no cmf_name) cannot be
        reconstructed faithfully: fail loudly instead of guessing an id."""
        from ramanujantools.cmf import pFq as rt_pFq
        raw_cmf = rt_pFq(1, 1, -1)
        legacy = {
            'type': 'BaseCMF',
            'data': {
                'consts': ['log-2'], 'use_inv_t': True, 'shifts': [0, 0],
                'selected_start_points': None, 'selected_trajectories': None,
                'only_selected': False,
                **{sp.srepr(k): sp.srepr(m) for k, m in raw_cmf.matrices.items()},
            },
        }
        with pytest.raises(ValueError, match="cmf_name"):
            Formatter.from_json_obj(legacy)


# ---------------------------------------------------------------------------
# Defaults resolved at construction time, not import time
# ---------------------------------------------------------------------------
class TestUseInvTDefault:

    @pytest.mark.parametrize("make", [
        lambda: MeijerG("log-2", 1, 1, 2, 2, -1),
        lambda: pFq("log-2", 2, 1, -1),
    ])
    @pytest.mark.parametrize("value", [True, False])
    def test_follows_config_changed_after_import(self, monkeypatch, make, value):
        from dreamer.configs import config
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", value)
        assert make().use_inv_t is value
