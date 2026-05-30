import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.configs import extraction_config
from dreamer.extraction.extractor import ShardExtractor
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.types import CMFData


def _shift_cmf(selected_points=None, only_selected=False):
    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    return CMFData(cmf=cmf, shift=shift, selected_points=selected_points, only_selected=only_selected)


def test_extract_returns_whole_space_when_no_hyperplanes(monkeypatch):
    extractor = ShardExtractor(e, _shift_cmf())
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: set())

    shards = extractor.extract(call_number=1)

    assert len(shards) == 1
    assert shards[0].is_whole_space


def test_extract_only_selected_requires_points():
    extractor = ShardExtractor(e, _shift_cmf(selected_points=None, only_selected=True))
    with pytest.raises(ValueError, match="No start points were provided for extraction"):
        extractor.extract(call_number=1)


def test_extract_uses_selected_points_to_create_shards(monkeypatch):
    cmf_data = _shift_cmf(selected_points=[(2, 2), (3, 3)], only_selected=True)
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())

    hyperplanes = {
        Hyperplane(symbols[0], symbols=symbols),
        Hyperplane(symbols[1], symbols=symbols),
    }
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hyperplanes)

    shards = extractor.extract(call_number=1)

    assert len(shards) >= 1
    for shard in shards:
        assert shard.get_interior_point() is not None


# ----------------------------------------------------------------------
# v2 integration tests
# ----------------------------------------------------------------------


@pytest.fixture
def _restore_strategy():
    """Restore the configured strategy after the test, even on failure."""
    previous = extraction_config.STRATEGY
    yield
    extraction_config.STRATEGY = previous


def test_default_strategy_is_auto():
    """Pipeline integration must default to v2 'auto' for timeout safety."""
    assert extraction_config.STRATEGY == "auto"


def test_extract_unknown_strategy_raises(monkeypatch, _restore_strategy):
    cmf_data = _shift_cmf()
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())
    monkeypatch.setattr(
        extractor,
        "_extract_cmf_hps",
        lambda: [Hyperplane(symbols[0], symbols=symbols)],
    )
    extraction_config.STRATEGY = "bogus"
    with pytest.raises(ValueError, match="Unknown extraction strategy"):
        extractor.extract(call_number=1)


def test_extract_via_v2_heuristic_builds_shards(monkeypatch, _restore_strategy):
    """End-to-end through the v2 ray-shooter (no lrs binary needed)."""
    cmf_data = _shift_cmf()
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())
    # Two coordinate axes -> four open quadrants, all unbounded.
    hps = [
        Hyperplane(symbols[0], symbols=symbols),
        Hyperplane(symbols[1], symbols=symbols),
    ]
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)
    extraction_config.STRATEGY = "heuristic"

    shards = extractor.extract(call_number=1)

    # The heuristic should find at least 2 of the 4 quadrants.
    assert len(shards) >= 2
    seen_encodings = {tuple(s.encoding) for s in shards if s.encoding}
    valid = {(1, 1), (1, -1), (-1, 1), (-1, -1)}
    assert seen_encodings.issubset(valid)
    # Every shard must come with an integer interior point.
    for shard in shards:
        pt = shard.get_interior_point()
        assert pt is not None
        assert shard.in_space(pt)


def test_extract_via_v2_routes_through_manager(monkeypatch, _restore_strategy):
    """Verify the extractor calls ExtractionManager with the configured strategy."""
    import numpy as np

    cmf_data = _shift_cmf()
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())
    hps = [Hyperplane(symbols[0], symbols=symbols)]
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)
    extraction_config.STRATEGY = "auto"

    captured = {}

    class _Spy:
        def __init__(self, strategy, timeout_seconds, exact_unbounded_check="lp",
                     exact_num_workers=1, heuristic_refine=False,
                     heuristic_refine_threshold=50.0, heuristic_refine_workers=1,
                     heuristic_num_rays=None, heuristic_max_seconds=None,
                     heuristic_missing_mass=5e-4, heuristic_face_aligned=False,
                     heuristic_face_subsets=200, heuristic_face_offsets=50,
                     symmetry=None):
            captured["strategy"] = strategy
            captured["timeout_seconds"] = timeout_seconds
            captured["exact_unbounded_check"] = exact_unbounded_check
            captured["exact_num_workers"] = exact_num_workers
            captured["heuristic_refine"] = heuristic_refine
            captured["heuristic_refine_threshold"] = heuristic_refine_threshold
            captured["heuristic_refine_workers"] = heuristic_refine_workers
            captured["heuristic_num_rays"] = heuristic_num_rays
            captured["heuristic_max_seconds"] = heuristic_max_seconds  # forwarded from TIMEOUT_SECONDS
            captured["heuristic_missing_mass"] = heuristic_missing_mass
            captured["heuristic_face_aligned"] = heuristic_face_aligned
            captured["heuristic_face_subsets"] = heuristic_face_subsets
            captured["heuristic_face_offsets"] = heuristic_face_offsets

        def extract(self, hyperplanes):
            captured["num_hps"] = len(hyperplanes)
            return {(1,): np.array([3, 0], dtype=np.int64)}

    monkeypatch.setattr(
        "dreamer.extraction.extractor.ExtractionManager", _Spy
    )

    shards = extractor.extract(call_number=1)

    assert captured["strategy"] == "auto"
    assert captured["timeout_seconds"] == extraction_config.TIMEOUT_SECONDS
    assert captured["heuristic_max_seconds"] == extraction_config.TIMEOUT_SECONDS
    assert captured["exact_unbounded_check"] == extraction_config.EXACT_UNBOUNDED_CHECK
    assert captured["heuristic_refine"] == extraction_config.HEURISTIC_REFINE_WITNESSES
    assert captured["heuristic_num_rays"] == extraction_config.HEURISTIC_NUM_RAYS
    assert captured["heuristic_missing_mass"] == extraction_config.HEURISTIC_MISSING_MASS
    assert captured["heuristic_face_aligned"] == extraction_config.HEURISTIC_FACE_ALIGNED
    assert captured["heuristic_face_subsets"] == extraction_config.HEURISTIC_FACE_SUBSETS
    assert captured["heuristic_face_offsets"] == extraction_config.HEURISTIC_FACE_OFFSETS
    assert captured["num_hps"] == 1
    assert len(shards) == 1
    assert tuple(shards[0].encoding) == (1,)


def test_extract_via_v2_passes_symmetry_for_pfq(monkeypatch, _restore_strategy):
    """pFq + IGNORE_DUPLICATE_SEARCHABLES must pass a SymmetryStrategy to the manager."""
    import numpy as np
    from dreamer.extraction.v2 import BlockSortSymmetry

    cmf_data = _shift_cmf()
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())
    hps = [Hyperplane(symbols[0], symbols=symbols)]
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)

    captured = {}

    monkeypatch.setattr(
        "dreamer.extraction.extractor.ExtractionManager",
        lambda **kw: (captured.update(kw), type(
            "_Stub", (), {"extract": lambda self, hps_: {(1,): np.array([1, 0])}}
        )())[1],
    )

    extraction_config.STRATEGY = "auto"
    extraction_config.IGNORE_DUPLICATE_SEARCHABLES = True
    extractor.extract(call_number=1)
    # pFq symmetry is now implemented as canonical teleportation: a
    # BlockSortSymmetry strategy must be forwarded to the manager.
    assert isinstance(captured.get("symmetry"), BlockSortSymmetry)


def test_apply_shift_hoist_matches_per_shard(monkeypatch):
    """Passing pre-shifted hyperplanes (hyperplanes_already_shifted=True)
    must yield identical A/b to letting Shard.__init__ shift internally."""
    import numpy as np
    from dreamer.extraction.shard import Shard

    cmf = rt_pFq(1, 1, sp.Integer(1))
    syms = list(cmf.matrices.keys())
    shift = Position({syms[0]: sp.Integer(1), syms[1]: sp.Integer(2)})  # nonzero
    cmf_data = CMFData(cmf=cmf, shift=shift)
    hps = [Hyperplane(syms[0], symbols=syms), Hyperplane(syms[1], symbols=syms)]
    enc = [1, -1]

    s_raw = Shard.from_cmf_data(cmf_data, e, hps, enc)
    shifted = [hp.apply_shift(shift) for hp in hps]
    s_hoist = Shard.from_cmf_data(
        cmf_data, e, shifted, enc, hyperplanes_already_shifted=True
    )
    assert np.array_equal(s_raw.A, s_hoist.A)
    assert np.array_equal(s_raw.b, s_hoist.b)
    assert s_raw.symbols == s_hoist.symbols


# ----------------------------------------------------------------------
# Task 2: shard-cache load/skip
# ----------------------------------------------------------------------


def _boom(*_a, **_k):
    raise AssertionError("discovery should have been skipped (cache hit)")


def _write_cache(tmp_path, cmf_data, hps, encodings):
    """Build shards for the given encodings and persist them as the
    <cmf>__shards.jsonl cache under tmp_path."""
    from dreamer.utils.storage.atlas_writer import write_shard_records
    from dreamer.extraction.shard import Shard

    symbols = list(hps)[0].symbols
    shards = []
    for enc in encodings:
        pt = Position({s: v for s, v in zip(symbols, enc)})  # any interior witness
        shards.append(Shard.from_cmf_data(cmf_data, e, list(hps), list(enc), pt))
    write_shard_records(str(tmp_path), e, cmf_data.cmf_name, shards)


def test_load_shard_cache_skips_extraction(monkeypatch, tmp_path):
    from dreamer.configs import sys_config

    cmf_data = _shift_cmf()
    symbols = list(cmf_data.cmf.matrices.keys())
    hps = [
        Hyperplane(symbols[0], symbols=symbols),
        Hyperplane(symbols[1], symbols=symbols),
    ]
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", str(tmp_path))
    _write_cache(tmp_path, cmf_data, hps, [(1, 1), (-1, -1)])

    extractor = ShardExtractor(e, cmf_data)
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)
    # Prove discovery is bypassed entirely.
    monkeypatch.setattr(extractor, "_discover_via_v2", _boom)
    monkeypatch.setattr(extractor, "_discover_via_legacy", _boom)
    monkeypatch.setattr(extraction_config, "LOAD_SHARD_CACHE", True)

    shards = extractor.extract(call_number=1)

    assert {tuple(s.encoding) for s in shards} == {(1, 1), (-1, -1)}
    for s in shards:
        assert s.get_interior_point() is not None


def test_cache_ignored_when_flag_off(monkeypatch, tmp_path):
    from dreamer.configs import sys_config

    cmf_data = _shift_cmf()
    symbols = list(cmf_data.cmf.matrices.keys())
    hps = [Hyperplane(symbols[0], symbols=symbols),
           Hyperplane(symbols[1], symbols=symbols)]
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", str(tmp_path))
    _write_cache(tmp_path, cmf_data, hps, [(1, 1)])

    extractor = ShardExtractor(e, cmf_data)
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)
    monkeypatch.setattr(extraction_config, "LOAD_SHARD_CACHE", False)
    # Cache exists but flag is off -> discovery must run.
    sentinel = {(-1, 1): Position({symbols[0]: -1, symbols[1]: 1})}
    monkeypatch.setattr(extractor, "_discover_via_v2", lambda *a, **k: dict(sentinel))
    extraction_config.STRATEGY = "auto"

    shards = extractor.extract(call_number=1)
    assert {tuple(s.encoding) for s in shards} == {(-1, 1)}


def test_stale_cache_falls_back_to_extraction(monkeypatch, tmp_path):
    """A cache whose encodings don't match the current hyperplane count
    must be ignored (forces fresh extraction), not mis-aligned."""
    from dreamer.configs import sys_config

    cmf_data = _shift_cmf()
    symbols = list(cmf_data.cmf.matrices.keys())
    # Cache written for TWO hyperplanes...
    hps2 = [Hyperplane(symbols[0], symbols=symbols),
            Hyperplane(symbols[1], symbols=symbols)]
    monkeypatch.setattr(sys_config, "EXPORT_CMFS", str(tmp_path))
    _write_cache(tmp_path, cmf_data, hps2, [(1, 1), (-1, -1)])

    # ...but this run sees only ONE hyperplane -> stale.
    hps1 = [Hyperplane(symbols[0], symbols=symbols)]
    extractor = ShardExtractor(e, cmf_data)
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps1)
    monkeypatch.setattr(extraction_config, "LOAD_SHARD_CACHE", True)
    extraction_config.STRATEGY = "auto"
    sentinel = {(1,): Position({symbols[0]: 1, symbols[1]: 0})}
    monkeypatch.setattr(extractor, "_discover_via_v2", lambda *a, **k: dict(sentinel))

    shards = extractor.extract(call_number=1)
    # Came from discovery, not the stale 2-bit cache.
    assert {tuple(s.encoding) for s in shards} == {(1,)}
