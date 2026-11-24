"""The per-CMF walk type (``use_inv_t`` of a formatter / shard) reaches every walk.

Semantics (agreed with the user 2026-11-20, REQUESTS R24):

* ``use_inv_t=None`` (the default) follows the global ``search.DEFAULT_USES_INV_T``
  **at run time** — it is not frozen when the formatter is built, and it is
  saved to JSON as ``null``;
* an explicit ``True`` / ``False`` overrides the global for that CMF only.

The bug these tests pin: every stage built its ``TrajectoryAttributesHandler``
without a walk type, so only the global setting counted and a per-CMF
``use_inv_t`` was silently ignored.

Categories:
  - Formatter / CMFData / Shard keep ``None`` and explicit values (JSON round trip).
  - Handler: the searchable's walk type is used unless a walk type is passed.
  - Known answer: an overridden walk equals the explicitly typed walk.
  - Stage task: the stored row's ``walk_type`` and fingerprint follow the override.
  - Fingerprints: unchanged for rows that follow the global; distinct per walk type.
  - Rebuild from a stored row (Tier-3) uses the row's own walk type.
"""

import pytest
import sympy as sp
from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e, log
from dreamer.configs import config
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.loading import BaseCMF, MeijerG, pFq
from dreamer.loading.funcs.formatter import Formatter
from dreamer.utils.stage_pool import StagePool, Tier1Task, run_tier1_task
from dreamer.utils.storage.attribute_fingerprints import (
    attribute_fingerprint,
    stale_attributes,
    tier1_is_fresh,
)
from dreamer.utils.storage.trajectory_attributes import (
    TrajectoryAttributesHandler,
    derive_cmf_and_shard_ids,
    tier1_config_fingerprint,
)
from dreamer.utils.walk_type import resolve_walk_type, searchable_walk_type


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _global_type1(monkeypatch):
    """Every test starts with the global walk type 1 (the default)."""
    monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", True)


@pytest.fixture
def cmf():
    """1F1(z=1) — a minimal 2-symbol CMF."""
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(cmf):
    return list(cmf.matrices.keys())


def _shard(cmf, symbols, use_inv_t):
    zero = Position({s: sp.Integer(0) for s in symbols})
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(cmf, e, hps, [1, 1], zero, interior, use_inv_t=use_inv_t)


def _ray(symbols):
    """A ray of 1F1(1) that walks in both walk types (start (1,1), direction (1,1))."""
    traj = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    start = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return traj, start


# ---------------------------------------------------------------------------
# Resolution rule
# ---------------------------------------------------------------------------

class TestResolveWalkType:
    def test_none_follows_global(self, monkeypatch):
        assert resolve_walk_type(None) == 1
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
        assert resolve_walk_type(None) == 2

    @pytest.mark.parametrize("glob", [True, False])
    def test_explicit_overrides_global(self, monkeypatch, glob):
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", glob)
        assert resolve_walk_type(True) == 1
        assert resolve_walk_type(False) == 2

    def test_searchable_without_attribute_follows_global(self):
        class _Bare:
            pass
        assert searchable_walk_type(_Bare()) == 1
        assert searchable_walk_type(None) == 1


# ---------------------------------------------------------------------------
# Formatter / CMFData / Shard keep the per-CMF value
# ---------------------------------------------------------------------------

class TestFormatterKeepsValue:
    def test_default_is_none_not_frozen(self, monkeypatch):
        # BaseCMF: pFq / MeijerG type-2 requests are converted to type 1 (R1, test_walk_type_conversion).
        f = BaseCMF(log(2), "my2F1", rt_pFq(2, 1, -1))
        assert f.use_inv_t is None
        # The global changes after the formatter was built: the CMF follows it.
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
        assert f.to_cmf().use_inv_t is None
        assert resolve_walk_type(f.to_cmf().use_inv_t) == 2

    @pytest.mark.parametrize("value", [None, True, False])
    def test_json_round_trip(self, value):
        f = pFq(log(2), 2, 1, -1, use_inv_t=value)
        obj = f.to_json_obj()
        assert obj["data"]["use_inv_t"] is value
        assert Formatter.from_json_obj(obj).use_inv_t is value

    @pytest.mark.parametrize("value", [None, True, False])
    def test_base_cmf_passes_value_to_cmf_data(self, value):
        assert BaseCMF(log(2), "my2F1", rt_pFq(2, 1, -1), use_inv_t=value).to_cmf().use_inv_t is value

    @pytest.mark.parametrize("value", [None, True])
    def test_meijer_g_passes_type1_value_to_cmf_data(self, value):
        assert MeijerG(log(2), 1, 2, 2, 2, -1, use_inv_t=value).to_cmf().use_inv_t is value

    def test_shard_keeps_none_and_explicit(self, cmf, symbols):
        assert _shard(cmf, symbols, None).use_inv_t is None
        assert _shard(cmf, symbols, False).use_inv_t is False

    def test_shard_from_cmf_data_keeps_override(self):
        cmf_data = BaseCMF(log(2), "my2F1", rt_pFq(2, 1, -1), use_inv_t=False).to_cmf()
        sh = Shard.from_cmf_data(cmf_data, log(2), [], [])
        assert sh.use_inv_t is False and searchable_walk_type(sh) == 2


# ---------------------------------------------------------------------------
# Handler: searchable's walk type unless a walk type is passed
# ---------------------------------------------------------------------------

class TestHandlerWalkType:
    @pytest.mark.parametrize("use_inv_t, expected", [(None, 1), (True, 1), (False, 2)])
    def test_from_searchable(self, cmf, symbols, use_inv_t, expected):
        sh = _shard(cmf, symbols, use_inv_t)
        traj, start = _ray(symbols)
        h = TrajectoryAttributesHandler.from_cmf(cmf, traj, start, searchable=sh)
        assert h.walk_type() == expected

    def test_explicit_walk_type_wins(self, cmf, symbols):
        sh = _shard(cmf, symbols, False)
        traj, start = _ray(symbols)
        h = TrajectoryAttributesHandler.from_cmf(cmf, traj, start, searchable=sh, walk_type=1)
        assert h.walk_type() == 1

    def test_override_walk_equals_typed_walk(self, cmf, symbols):
        """Known answer: the overridden walk is exactly the explicitly typed type-2 walk."""
        sh = _shard(cmf, symbols, False)
        traj, start = _ray(symbols)
        h_override = TrajectoryAttributesHandler.from_cmf(cmf, traj, start, walk_depth=50, searchable=sh)
        h_typed = TrajectoryAttributesHandler.from_cmf(cmf, traj, start, walk_depth=50, walk_type=2)
        h_type1 = TrajectoryAttributesHandler.from_cmf(cmf, traj, start, walk_depth=50, walk_type=1)
        assert h_override._walked_matrix(50) is not None
        assert h_override._walked_matrix(50) == h_typed._walked_matrix(50)
        assert h_override._walked_matrix(50) != h_type1._walked_matrix(50)


# ---------------------------------------------------------------------------
# Stage task: stored row follows the override
# ---------------------------------------------------------------------------

class TestStageTask:
    @pytest.mark.parametrize("use_inv_t, expected", [(None, 1), (False, 2)])
    def test_row_walk_type_and_fingerprint(self, cmf, symbols, use_inv_t, expected):
        sh = _shard(cmf, symbols, use_inv_t)
        traj, start = _ray(symbols)
        cmf_id, shard_id, enc = derive_cmf_and_shard_ids(sh)
        task = Tier1Task("k", 0, traj, start, (e,), cmf_id, shard_id, enc)
        with StagePool([sh], workers=1):
            res = run_tier1_task(task)
        (dto,) = res.dtos
        assert dto.walk_type == expected
        assert dto.config_fingerprint == tier1_config_fingerprint(dto.walk_depth, walk_type=expected)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

class TestFingerprints:
    def test_following_global_is_unchanged(self):
        """Rows whose CMF follows the global keep their fingerprint (no stored row goes stale)."""
        assert tier1_config_fingerprint(100, walk_type=1) == tier1_config_fingerprint(100)
        assert attribute_fingerprint("eigenvalues", 100, walk_type=1) == attribute_fingerprint("eigenvalues", 100)

    def test_walk_type_equals_global_setting(self, monkeypatch):
        fp2 = tier1_config_fingerprint(100, walk_type=2)
        eig2 = attribute_fingerprint("eigenvalues", 100, walk_type=2)
        assert fp2 != tier1_config_fingerprint(100)
        monkeypatch.setattr(config.search, "DEFAULT_USES_INV_T", False)
        assert tier1_config_fingerprint(100) == fp2
        assert attribute_fingerprint("eigenvalues", 100) == eig2

    def test_freshness_uses_walk_type(self):
        row = {"delta": 0.1, "config_fingerprint": tier1_config_fingerprint(100, walk_type=2),
               "eigenvalues": "[]", "fingerprints": {"eigenvalues": attribute_fingerprint("eigenvalues", 100,
                                                                                         walk_type=2)}}
        assert tier1_is_fresh(row, 100, walk_type=2)
        assert not tier1_is_fresh(row, 100)
        assert stale_attributes(row, ["eigenvalues"], 100, walk_type=2) == []
        assert stale_attributes(row, ["eigenvalues"], 100) == ["eigenvalues"]


# ---------------------------------------------------------------------------
# Rebuild from a stored row (Tier-3) uses the row's walk type
# ---------------------------------------------------------------------------

class TestRebuildFromRow:
    def test_tier3_seed_fields_carry_walk_type(self):
        from dreamer.post_process.tier3_post_process_mod import _SEED_FIELDS
        assert "walk_type" in _SEED_FIELDS

    def test_tier3_task_walks_row_walk_type(self, cmf, symbols, monkeypatch):
        from dreamer.post_process import tier3_post_process_mod as t3

        seen = {}

        class _Capture:
            @staticmethod
            def from_cmf(*args, **kwargs):
                seen.update(kwargs)
                raise RuntimeError("captured")

        monkeypatch.setattr(t3, "TrajectoryAttributesHandler", _Capture)
        monkeypatch.setattr(t3, "stage_extra", lambda key, default=None: {"c": cmf})
        record = {"start_point": [1, 1], "direction": [1, 1], "walk_depth": 50, "walk_type": 2}
        t3._tier3_task(t3.Tier3Task("p", "c", record, None, (), None))
        assert seen.get("walk_type") == 2
