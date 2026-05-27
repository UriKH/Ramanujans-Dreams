"""
Unit tests for ``dreamer.extraction.v2``.

These exercise the strategy-pattern shard extractor without depending
on the lrs binary being installed.  Where lrs *is* required, the test
is skipped automatically.
"""
from __future__ import annotations

import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import sympy as sp

from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.v2 import (
    BaseExtractor,
    ExtractionManager,
    LrslibExtractor,
    RayShootingExtractor,
)
from dreamer.extraction.v2 import cells, lrs_io, milp


x, y, z = sp.symbols("x y z")


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _hps_axes_2d():
    """Two coordinate axes in the plane -> four open quadrant cells, all unbounded."""
    return [
        Hyperplane(x, symbols=[x, y]),
        Hyperplane(y, symbols=[x, y]),
    ]


def _hps_strip_2d():
    """Two parallel lines making a bounded strip in y and two unbounded half-planes
    in x.  In R^2 the arrangement has 3 cells along y; together with x-coord
    sign that's 6 cells, but here we keep just the two parallel ones so the
    middle cell is a bounded strip when intersected with -- wait, actually two
    parallel hyperplanes alone in R^2 yield three unbounded cells (below both,
    between them, above both).  Useful as a sanity check that "between" is also
    flagged unbounded."""
    return [
        Hyperplane(y - 1, symbols=[x, y]),
        Hyperplane(y + 1, symbols=[x, y]),
    ]


def _hps_triangle_2d():
    """Three lines forming a (bounded) triangle plus six unbounded cells."""
    return [
        Hyperplane(y, symbols=[x, y]),                       # y = 0
        Hyperplane(x, symbols=[x, y]),                       # x = 0
        Hyperplane(x + y - 4, symbols=[x, y]),               # x+y = 4
    ]


# ----------------------------------------------------------------------
# BaseExtractor helpers
# ----------------------------------------------------------------------


class TestHyperplanesToMatrix:
    def test_packs_coefficients(self):
        hps = _hps_axes_2d()
        A, c = BaseExtractor.hyperplanes_to_matrix(hps)
        assert A.shape == (2, 2)
        assert c.shape == (2,)
        # x => coefficients [1, 0], constant 0
        # y => coefficients [0, 1], constant 0
        assert A.tolist() == [[1, 0], [0, 1]]
        assert c.tolist() == [0, 0]

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            BaseExtractor.hyperplanes_to_matrix([])

    def test_rejects_mismatched_symbols(self):
        hp1 = Hyperplane(x, symbols=[x, y])
        hp2 = Hyperplane(y, symbols=[y, x])  # different order
        with pytest.raises(ValueError, match="symbol order"):
            BaseExtractor.hyperplanes_to_matrix([hp1, hp2])


# ----------------------------------------------------------------------
# MILP feasibility
# ----------------------------------------------------------------------


class TestFindIntegerPoint:
    def test_quadrant_has_integer_point(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.int64)
        c = np.array([0, 0], dtype=np.int64)
        pt = milp.find_integer_point(A, c, [1, 1])
        assert pt is not None
        assert (A @ pt + c > 0).all()

    def test_empty_intersection_returns_none(self):
        # x > 0 AND x < 0 is empty
        A = np.array([[1], [1]], dtype=np.int64)
        c = np.array([0, 0], dtype=np.int64)
        assert milp.find_integer_point(A, c, [1, -1]) is None

    def test_no_integer_point_in_thin_slab(self):
        # 0 < x < 1 has no integer x; encode as x > 0 (sign +1) and x < 1 (sign -1
        # on x - 1).
        A = np.array([[1], [1]], dtype=np.int64)
        c = np.array([0, -1], dtype=np.int64)
        assert milp.find_integer_point(A, c, [1, -1]) is None

    def test_strict_inequalities_enforced(self):
        # x > 0 must not return x = 0
        A = np.array([[1]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        pt = milp.find_integer_point(A, c, [1])
        assert pt is not None
        assert pt[0] >= 1

    def test_validates_sign_vector(self):
        A = np.array([[1]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError, match="\\+1 or -1"):
            milp.find_integer_point(A, c, [0])


# ----------------------------------------------------------------------
# lrs format + parser (no binary needed)
# ----------------------------------------------------------------------


class TestLrsIo:
    def test_format_hrep_round_trip(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.int64)
        c = np.array([0, 0], dtype=np.int64)
        text = lrs_io.format_hrep(A, c, [1, -1], name="demo")
        lines = text.splitlines()
        assert lines[0] == "demo"
        assert lines[1] == "H-representation"
        assert lines[2] == "begin"
        assert lines[3] == "2 3 integer"
        # First row encodes x >= 0  (sign +1)  =>  0 1 0
        # Second row encodes y <= 0 i.e. -y >= 0  =>  0 0 -1
        assert lines[4].split() == ["0", "1", "0"]
        assert lines[5].split() == ["0", "0", "-1"]
        assert lines[6] == "end"

    def test_format_hrep_rejects_zero_sign(self):
        A = np.array([[1]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        with pytest.raises(ValueError, match="\\+1 or -1"):
            lrs_io.format_hrep(A, c, [0])

    def test_parse_vrep_detects_ray(self):
        vrep = """
        * comment
        V-representation
        begin
        2 3 rational
         1  0 0
         0  1 1
        end
        """
        assert lrs_io.parse_vrep_unbounded(vrep) is True

    def test_parse_vrep_bounded(self):
        vrep = """
        V-representation
        begin
        3 3 rational
         1 0 0
         1 1 0
         1 0 1
        end
        """
        assert lrs_io.parse_vrep_unbounded(vrep) is False


# ----------------------------------------------------------------------
# Cell enumeration
# ----------------------------------------------------------------------


class TestEnumerateCells:
    def test_two_axes_have_four_cells(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        found = cells.enumerate_cells(A, c, seed=0)
        assert sorted(found) == sorted([
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ])

    def test_parallel_lines_have_three_cells(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_strip_2d())
        found = cells.enumerate_cells(A, c, seed=0)
        # y < -1 -> (-1, -1), -1 < y < 1 -> (-1, +1), y > 1 -> (+1, +1)
        assert sorted(found) == sorted([
            (-1, -1), (-1, 1), (1, 1)
        ])

    def test_triangle_arrangement_has_seven_cells(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        found = cells.enumerate_cells(A, c, seed=0)
        # 3 lines in general position -> 7 cells (one bounded triangle + 6 unbounded).
        assert len(found) == 7

    def test_respects_max_cells(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        with pytest.raises(RuntimeError, match="max_cells"):
            cells.enumerate_cells(A, c, seed=0, max_cells=2)


# ----------------------------------------------------------------------
# RayShootingExtractor
# ----------------------------------------------------------------------


class TestRayShootingExtractor:
    def test_finds_some_unbounded_quadrants(self):
        extractor = RayShootingExtractor(num_rays=200, max_coord=3, seed=0)
        result = extractor.extract(_hps_axes_2d())
        assert len(result) >= 2  # should hit at least 2/4 quadrants
        for sig, point in result.items():
            assert all(s in (-1, 1) for s in sig)
            A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
            vals = A @ point + c
            assert tuple(np.where(vals > 0, 1, -1).tolist()) == sig

    def test_empty_hyperplanes(self):
        assert RayShootingExtractor().extract([]) == {}

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            RayShootingExtractor(num_rays=0)
        with pytest.raises(ValueError):
            RayShootingExtractor(max_coord=0)


# ----------------------------------------------------------------------
# LrslibExtractor (skip when binary missing)
# ----------------------------------------------------------------------


lrs_required = pytest.mark.skipif(
    not lrs_io.lrs_available(), reason="lrs binary not available"
)


class TestLrslibExtractor:
    def test_constructor_raises_without_binary(self):
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=False):
            with pytest.raises(FileNotFoundError, match="lrs"):
                LrslibExtractor()

    def test_is_unbounded_dispatch(self):
        """Even without the binary we can exercise the parse path with a fake stdout."""
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=True):
            extractor = LrslibExtractor()
        A = np.array([[1, 0]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        bounded_vrep = "V-representation\nbegin\n1 3 rational\n 1 0 0\nend\n"
        with patch("dreamer.extraction.v2.lrs_extractor.run_lrs", return_value=bounded_vrep):
            assert extractor._is_unbounded(A, c, np.array([1])) is False
        unbounded_vrep = "V-representation\nbegin\n1 3 rational\n 0 1 0\nend\n"
        with patch("dreamer.extraction.v2.lrs_extractor.run_lrs", return_value=unbounded_vrep):
            assert extractor._is_unbounded(A, c, np.array([1])) is True

    def test_timeout_is_wrapped(self):
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=True):
            extractor = LrslibExtractor(per_call_timeout=0.1)
        A = np.array([[1, 0]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        with patch(
            "dreamer.extraction.v2.lrs_extractor.run_lrs",
            side_effect=subprocess.TimeoutExpired(cmd="lrs", timeout=0.1),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                extractor._is_unbounded(A, c, np.array([1]))

    @lrs_required
    def test_end_to_end_axes(self):
        extractor = LrslibExtractor()
        result = extractor.extract(_hps_axes_2d())
        # All 4 quadrants of R^2 are unbounded.
        assert len(result) == 4
        for sig, pt in result.items():
            A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
            vals = A @ pt + c
            assert tuple(np.where(vals > 0, 1, -1).tolist()) == sig

    @lrs_required
    def test_end_to_end_triangle_drops_bounded(self):
        extractor = LrslibExtractor()
        result = extractor.extract(_hps_triangle_2d())
        # 7 cells total but the inner triangle is bounded, so 6 survive.
        assert len(result) == 6


# ----------------------------------------------------------------------
# ExtractionManager
# ----------------------------------------------------------------------


class _FakeExact(BaseExtractor):
    name = "exact"

    def __init__(self, returns=None, raises=None, sleep=0.0):
        self._returns = returns or {}
        self._raises = raises
        self._sleep = sleep

    def extract(self, hyperplanes):
        if self._sleep:
            import time as _time
            _time.sleep(self._sleep)
        if self._raises:
            raise self._raises
        return self._returns


class _FakeHeuristic(BaseExtractor):
    name = "heuristic"

    def __init__(self, returns=None):
        self._returns = returns or {}

    def extract(self, hyperplanes):
        return self._returns


class TestExtractionManager:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ExtractionManager(strategy="bogus")  # type: ignore[arg-type]

    def test_heuristic_only(self):
        heur = _FakeHeuristic(returns={(1,): np.array([1])})
        mgr = ExtractionManager(strategy="heuristic", heuristic=heur)  # type: ignore[arg-type]
        assert mgr.extract([]) == {(1,): heur._returns[(1,)]}

    def test_exact_only_propagates_error(self):
        exact = _FakeExact(raises=RuntimeError("nope"))
        mgr = ExtractionManager(strategy="exact", exact=exact)  # type: ignore[arg-type]
        with pytest.raises(RuntimeError, match="nope"):
            mgr.extract([])

    def test_auto_uses_exact_when_fast(self):
        exact = _FakeExact(returns={(1,): np.array([1])})
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=2.0
        )  # type: ignore[arg-type]
        out = mgr.extract([])
        assert (1,) in out

    def test_auto_falls_back_on_timeout(self):
        exact = _FakeExact(returns={(1,): np.array([1])}, sleep=1.0)
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=0.05
        )  # type: ignore[arg-type]
        out = mgr.extract([])
        assert (-1,) in out

    def test_auto_falls_back_on_exception(self):
        exact = _FakeExact(raises=RuntimeError("explode"))
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=2.0
        )  # type: ignore[arg-type]
        out = mgr.extract([])
        assert (-1,) in out

    def test_auto_falls_back_when_binary_missing(self):
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        # No exact passed -> lazy construction fails -> heuristic used.
        with patch(
            "dreamer.extraction.v2.manager.LrslibExtractor",
            side_effect=FileNotFoundError("lrs"),
        ):
            mgr = ExtractionManager(
                strategy="auto", heuristic=heur, timeout_seconds=1.0
            )  # type: ignore[arg-type]
            out = mgr.extract([])
        assert (-1,) in out
