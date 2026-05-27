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
        def __init__(self, strategy, timeout_seconds):
            captured["strategy"] = strategy
            captured["timeout_seconds"] = timeout_seconds

        def extract(self, hyperplanes):
            captured["num_hps"] = len(hyperplanes)
            return {(1,): np.array([3, 0], dtype=np.int64)}

    monkeypatch.setattr(
        "dreamer.extraction.extractor.ExtractionManager", _Spy
    )

    shards = extractor.extract(call_number=1)

    assert captured["strategy"] == "auto"
    assert captured["timeout_seconds"] == extraction_config.STRATEGY_TIMEOUT_SECONDS
    assert captured["num_hps"] == 1
    assert len(shards) == 1
    assert tuple(shards[0].encoding) == (1,)


def test_extract_via_v2_warns_on_pfq_dedup(monkeypatch, capsys, _restore_strategy):
    """pFq + IGNORE_DUPLICATE_SEARCHABLES under v2 must warn (no dedup yet)."""
    import numpy as np

    cmf_data = _shift_cmf()
    extractor = ShardExtractor(e, cmf_data)
    symbols = list(cmf_data.cmf.matrices.keys())
    hps = [Hyperplane(symbols[0], symbols=symbols)]
    monkeypatch.setattr(extractor, "_extract_cmf_hps", lambda: hps)
    monkeypatch.setattr(
        "dreamer.extraction.extractor.ExtractionManager",
        lambda **kw: type(
            "_Stub", (), {"extract": lambda self, hps_: {(1,): np.array([1, 0])}}
        )(),
    )

    extraction_config.STRATEGY = "auto"
    extraction_config.IGNORE_DUPLICATE_SEARCHABLES = True
    extractor.extract(call_number=1)
    out = capsys.readouterr().out
    assert "IGNORE_DUPLICATE_SEARCHABLES" in out
    assert "WARNING" in out
