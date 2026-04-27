"""Tests for DataManager JSON payloads and JSONable contract behavior."""

import sympy as sp
import pytest

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.jsonable import JSONable
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector


class _BrokenJSONable(JSONable):
    """Intentionally incomplete implementation used to assert abstract contract enforcement."""

    pass


def _build_demo_shard() -> Shard:
    """Create a minimal shard used in DataManager JSON roundtrip tests."""
    cmf = rt_pFq(1, 1, sp.Integer(1))
    symbols = list(cmf.matrices.keys())
    shift = Position({symbols[0]: sp.Integer(0), symbols[1]: sp.Integer(0)})
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(cmf, e, hps, [1, 1], shift, interior)


def test_jsonable_requires_to_json_implementation():
    """Failure-path: abstract JSONable subclasses without to_json implementation must be non-instantiable."""
    with pytest.raises(TypeError):
        _BrokenJSONable()


def test_data_manager_json_roundtrip_preserves_searchable_space_and_entries():
    """Known-answer/invariant: DataManager JSON roundtrip should preserve searchable context and stored SearchData."""
    space = _build_demo_shard()
    dm = DataManager(use_LIReC=True, searchable_space=space)

    start = space.get_interior_point()
    traj = Position({sym: sp.Integer(0) for sym in space.symbols})
    sd = SearchData(SearchVector(start, traj), delta=1.25)
    dm[sd.sv] = sd

    restored = DataManager.from_json_obj(dm.to_json())

    assert isinstance(restored.searchable_space, Shard)
    assert restored.searchable_space.const.name == space.const.name
    assert len(restored) == 1
    assert next(iter(restored.values())).delta == 1.25

