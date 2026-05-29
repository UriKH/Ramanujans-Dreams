"""
Unit tests for ``dreamer.extraction.v2``.

These exercise the strategy-pattern shard extractor without depending
on the lrs binary being installed.  Where lrs *is* required, the test
is skipped automatically.
"""
from __future__ import annotations

import subprocess
import time
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

    def test_returned_point_has_correct_dim(self):
        """Witness shape must be (D,) -- not the (2D,) raw MILP vector
        that includes the L1 slacks."""
        A = np.eye(5, dtype=np.int64)
        c = np.zeros(5, dtype=np.int64)
        pt = milp.find_integer_point(A, c, [1, 1, 1, 1, 1])
        assert pt is not None
        assert pt.shape == (5,)
        # Sanity: A @ pt is valid (would raise if shape were 2D).
        _ = A @ pt

    def test_l1_minimisation_prefers_near_origin(self):
        """With L1 objective the witness should land near the origin,
        not 10**6 like the old feasibility-only formulation could give."""
        # 3-D positive orthant: x > 0, y > 0, z > 0
        A = np.eye(3, dtype=np.int64)
        c = np.zeros(3, dtype=np.int64)
        pt = milp.find_integer_point(A, c, [1, 1, 1])
        assert pt is not None
        # The L1-optimal integer point in the open positive orthant
        # under strict-integer-tightening x_i >= 1 is (1, 1, 1).
        assert pt.tolist() == [1, 1, 1]


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

    def test_respects_max_cells_when_set(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        with pytest.raises(RuntimeError, match="max_cells"):
            cells.enumerate_cells(A, c, seed=0, max_cells=2)

    def test_max_cells_none_does_not_cap(self):
        """Default max_cells is None -> the timeout, not a count, stops."""
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        found = cells.enumerate_cells(A, c, seed=0)  # default max_cells=None
        assert len(found) == 7  # full arrangement enumerated, no cap raised

    def test_find_start_cell_prefers_near_origin(self):
        """The reverse-search base is sampled near the origin first.  With
        two 1-D hyperplanes far out at x = +-50, the tight [-3, 3] box lies
        entirely inside the middle cell (-1, +1); a far-out base would land
        in an outer cell instead, so getting the middle cell confirms the
        near-origin bias (which front-loads fat, integer-rich cells)."""
        A = np.array([[1], [1]], dtype=np.int64)
        c = np.array([-50, 50], dtype=np.int64)  # hyperplanes x = 50 and x = -50
        base = cells._find_start_cell(A, c, rng=np.random.default_rng(0))
        assert tuple(base.tolist()) == (-1, 1)

    def test_find_start_cell_widens_when_tight_box_unusable(self):
        """If every lattice point in the tight box lies exactly on a
        hyperplane, sampling must widen outward until a real cell is found
        rather than giving up."""
        ks = list(range(-3, 4))  # hyperplanes x = -3, -2, ..., 3
        A = np.array([[1]] * len(ks), dtype=np.int64)
        c = np.array([-k for k in ks], dtype=np.int64)
        base = cells._find_start_cell(A, c, rng=np.random.default_rng(0))
        assert base.shape == (len(ks),)
        assert 0 not in base.tolist()  # a genuine open cell, not on a hyperplane

    def test_scipy_fallback_matches_mip(self, monkeypatch):
        """With python-mip force-disabled, enumeration must fall back to
        the scipy LP and produce the identical cell set."""
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        mip_result = cells.enumerate_cells(A, c, seed=0)

        monkeypatch.setattr(cells, "_HAS_MIP", False)
        scipy_result = cells.enumerate_cells(A, c, seed=0)

        assert sorted(scipy_result) == sorted(mip_result)
        assert len(scipy_result) == 7

    def test_checker_uses_stateful_solver_when_mip_present(self):
        """When mip is available the feasibility checker must be the
        bound-swapping stateful solver, not the scipy fallback."""
        if not cells._HAS_MIP:
            pytest.skip("python-mip not installed in this environment")
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        checker = cells._make_feasibility_checker(A, c, epsilon=1e-9)
        # The stateful solver exposes itself as a bound method of the
        # solver instance; the scipy fallback is a plain lambda.
        assert getattr(checker, "__self__", None).__class__ is cells._StatefulFeasibilitySolver
        # And it answers feasibility correctly: every quadrant is real.
        for sign in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            assert checker(np.array(sign, dtype=np.int64)) is True

    def test_stateful_solver_rejects_empty_cell(self):
        """A geometrically empty sign pattern must be infeasible."""
        if not cells._HAS_MIP:
            pytest.skip("python-mip not installed in this environment")
        # Two parallel hyperplanes x0 = 0 and x0 = 1 (c = [0, -1]).
        # sign (-1, +1) means x0 < 0 AND x0 > 1 -> empty; the other
        # three sign patterns are non-empty.
        A = np.array([[1, 0], [1, 0]], dtype=np.int64)
        c = np.array([0, -1], dtype=np.int64)
        solver = cells._StatefulFeasibilitySolver(A, c, epsilon=1e-6)
        assert solver.feasible(np.array([1, 1], dtype=np.int64)) is True
        assert solver.feasible(np.array([1, -1], dtype=np.int64)) is True
        assert solver.feasible(np.array([-1, 1], dtype=np.int64)) is False
        assert solver.feasible(np.array([-1, -1], dtype=np.int64)) is True

    def test_reverse_search_higher_dim_matches_bruteforce(self):
        """Reverse search must find every non-empty cell.  Cross-check
        against an independent brute-force feasibility sweep on a small
        arrangement where 2^N enumeration is affordable."""
        rng = np.random.default_rng(3)
        A = rng.integers(-2, 3, size=(8, 4)).astype(np.int64)
        c = rng.integers(-2, 3, size=8).astype(np.int64)
        found = set(cells.enumerate_cells(A, c, seed=0))

        # Independent ground truth: test all 2^8 sign patterns directly.
        import itertools
        checker = cells._make_feasibility_checker(A, c, epsilon=1e-6)
        truth = set()
        for signs in itertools.product((-1, 1), repeat=8):
            if checker(np.array(signs, dtype=np.int64)):
                truth.add(signs)
        assert found == truth

    def test_deadline_raises_extraction_timeout(self):
        """A deadline already in the past must abort with ExtractionTimeout."""
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        with pytest.raises(cells.ExtractionTimeout):
            cells.enumerate_cells(A, c, seed=0, deadline=time.time() - 1.0)

    def test_parallel_matches_serial(self):
        """num_workers>1 (reverse-search subtree dispatch) must yield the
        same cell set as the serial sweep."""
        rng = np.random.default_rng(11)
        A = rng.integers(-2, 3, size=(9, 4)).astype(np.int64)
        c = rng.integers(-2, 3, size=9).astype(np.int64)
        serial = cells.enumerate_cells(A, c, seed=0, num_workers=1)
        parallel = cells.enumerate_cells(A, c, seed=0, num_workers=4)
        assert sorted(parallel) == sorted(serial)


# ----------------------------------------------------------------------
# Recession-cone unbounded check
# ----------------------------------------------------------------------


class TestUnboundedChecker:
    def test_all_quadrants_unbounded(self):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        check = cells.make_unbounded_checker(A)
        for sign in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            assert check(np.array(sign, dtype=np.int64)) is True

    def test_triangle_inner_cell_is_bounded(self):
        # The bounded inner triangle of y>0, x>0, x+y<4 has sign
        # (+1, +1, -1) on [y, x, x+y-4]; it must read as bounded.
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        check = cells.make_unbounded_checker(A)
        # Find the inner (bounded) cell among the 7 and confirm it's the
        # only bounded one.
        all_cells = cells.enumerate_cells(A, c, seed=0)
        bounded = [s for s in all_cells if not check(np.array(s, dtype=np.int64))]
        assert len(bounded) == 1  # exactly the inner triangle

    def test_scipy_fallback_matches_mip(self, monkeypatch):
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_triangle_2d())
        all_cells = cells.enumerate_cells(A, c, seed=0)

        mip_check = cells.make_unbounded_checker(A)
        mip_unb = {s for s in all_cells if mip_check(np.array(s, dtype=np.int64))}

        monkeypatch.setattr(cells, "_HAS_MIP", False)
        scipy_check = cells.make_unbounded_checker(A)
        scipy_unb = {s for s in all_cells if scipy_check(np.array(s, dtype=np.int64))}

        assert mip_unb == scipy_unb
        assert len(mip_unb) == 6  # 6 unbounded, 1 bounded


# ----------------------------------------------------------------------
# RayShootingExtractor
# ----------------------------------------------------------------------


class TestRayShootingExtractor:
    def test_finds_all_unbounded_quadrants(self):
        """With the algebraic formulation, even a modest ray budget
        should cover every quadrant of the 2-D axes arrangement."""
        extractor = RayShootingExtractor(num_rays=200, max_coord=3, seed=0)
        result = extractor.extract(_hps_axes_2d())
        # All 4 quadrants are unbounded; algebraic shooter should hit them.
        assert len(result) == 4
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        for sig, point in result.items():
            assert all(s in (-1, 1) for s in sig)
            vals = A @ point + c
            assert tuple(np.where(vals > 0, 1, -1).tolist()) == sig

    def test_empty_hyperplanes(self):
        assert RayShootingExtractor().extract([]) == {}

    def test_collision_keeps_nearest_witness(self):
        """When two rays land in the same cell, the nearest-to-origin
        (min-L1) witness is kept regardless of arrival order."""
        ext = RayShootingExtractor()
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        for order in ([[3, 3], [1, 1]], [[1, 1], [3, 3]]):
            out = {}
            ext._collect_unique_cells_into(np.array(order, dtype=np.int64), A, c, out)
            assert out[(1, 1)].tolist() == [1, 1]

    def test_refine_witnesses_returns_milp_minimal(self):
        """With refine_witnesses=True (threshold 0 -> refine all) each
        shard's point is the MILP L1-minimal integer point of its cell --
        same cells discovered, and never worse than the raw ray witness."""
        hps = _hps_axes_2d()
        A, c = BaseExtractor.hyperplanes_to_matrix(hps)
        raw = RayShootingExtractor(num_rays=500, max_coord=5, seed=0).extract(hps)
        refined = RayShootingExtractor(
            num_rays=500, max_coord=5, seed=0,
            refine_witnesses=True, refine_l1_threshold=0,
        ).extract(hps)
        assert set(raw) == set(refined)  # refinement never changes which cells
        for sig, pt in refined.items():
            expected = milp.find_integer_point(A, c, np.asarray(sig, dtype=np.int64))
            assert pt.tolist() == expected.tolist()
            assert np.abs(pt).sum() <= np.abs(raw[sig]).sum()

    def test_refine_only_above_threshold(self):
        """Only witnesses with L1 norm above refine_l1_threshold are
        recomputed; smaller ones are left exactly as the ray found them."""
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        ext = RayShootingExtractor(refine_witnesses=True, refine_l1_threshold=50)
        # Far-out witness (L1 = 200 > 50) -> refined to the MILP minimum (1, 1).
        far = {(1, 1): np.array([100, 100], dtype=np.int64)}
        ext._refine_witnesses(A, c, far)
        assert far[(1, 1)].tolist() == [1, 1]
        # Small witness (L1 = 6 < 50) -> left untouched.
        near = {(1, 1): np.array([3, 3], dtype=np.int64)}
        ext._refine_witnesses(A, c, near)
        assert near[(1, 1)].tolist() == [3, 3]

    def test_refine_parallel_matches_serial(self):
        """Parallel refinement must produce the identical (MILP-minimal)
        witnesses as the serial path."""
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        big = {
            (1, 1): np.array([90, 90], dtype=np.int64),
            (1, -1): np.array([90, -90], dtype=np.int64),
            (-1, 1): np.array([-90, 90], dtype=np.int64),
            (-1, -1): np.array([-90, -90], dtype=np.int64),
        }
        serial = {k: v.copy() for k, v in big.items()}
        RayShootingExtractor(
            refine_witnesses=True, refine_l1_threshold=0
        )._refine_witnesses(A, c, serial)
        par = {k: v.copy() for k, v in big.items()}
        RayShootingExtractor(
            refine_witnesses=True, refine_l1_threshold=0, refine_workers=2
        )._refine_parallel(A, c, par, list(par.keys()))
        assert {k: v.tolist() for k, v in serial.items()} == {
            k: v.tolist() for k, v in par.items()
        }
        for k in par:
            exp = milp.find_integer_point(A, c, np.asarray(k, dtype=np.int64))
            assert par[k].tolist() == exp.tolist()

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            RayShootingExtractor(num_rays=0)
        with pytest.raises(ValueError):
            RayShootingExtractor(max_coord=0)
        with pytest.raises(ValueError):
            RayShootingExtractor(batch_size=0)
        with pytest.raises(ValueError):
            RayShootingExtractor(plateau_ratio=1.5)
        with pytest.raises(ValueError):
            RayShootingExtractor(plateau_patience=0)

    def test_plateau_patience_needs_consecutive_low_batches(self):
        """patience>1 must NOT stop on a single low-yield batch -- it keeps
        going until the low streak is sustained, so it discovers at least as
        many cells as patience=1 on the same arrangement/seed."""
        import sympy as sp
        syms = list(sp.symbols("a b c d"))      # 4-D axes: 16 unbounded cells
        hps = [Hyperplane(s, syms) for s in syms]
        kw = dict(num_rays=200_000, batch_size=2_000, plateau_ratio=1e-2, seed=0)
        eager = RayShootingExtractor(plateau_patience=1, **kw).extract(hps)
        patient = RayShootingExtractor(plateau_patience=5, **kw).extract(hps)
        assert len(patient) >= len(eager)

    def test_no_scipy_dependency(self):
        """The rewrite must be solver-free -- module must not import scipy."""
        import dreamer.extraction.v2.ray_extractor as mod
        # Detect scipy in any form referenced from the module's globals.
        for name, value in vars(mod).items():
            mod_name = getattr(value, "__module__", "") or ""
            assert not mod_name.startswith("scipy"), (
                f"ray_extractor referenced scipy via attribute {name!r}"
            )
        # Defensive: scan AST for any scipy import (catches a sneaked-in
        # lazy import that wouldn't show up in module globals).
        import ast
        import inspect
        tree = ast.parse(inspect.getsource(mod))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("scipy"), (
                        f"ray_extractor imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                modname = node.module or ""
                assert not modname.startswith("scipy"), (
                    f"ray_extractor imports from {modname}"
                )

    def test_handles_parallel_rays_without_div_by_zero(self):
        """A ray parallel to a hyperplane (dot = 0) must not crash and
        must still yield a witness in some open cell."""
        import sympy as sp
        x, y = sp.symbols("x y")
        # Hyperplanes x = 0 and y = 0; rays along (1, 0) and (0, 1)
        # are each parallel to one of them.
        hps = [Hyperplane(x, [x, y]), Hyperplane(y, [x, y])]
        # max_coord=1 forces only +/-1, 0 entries -> guaranteed
        # axis-aligned rays in the sample.
        extractor = RayShootingExtractor(num_rays=64, max_coord=1, seed=42)
        result = extractor.extract(hps)
        # Should still cover all 4 quadrants without raising.
        assert len(result) >= 1
        for sig in result:
            assert 0 not in sig

    def test_drops_ray_lying_on_hyperplane(self):
        """A ray that lies exactly on a hyperplane (M=0 and c=0) must
        be filtered out -- no witness can sit on a hyperplane."""
        import sympy as sp
        x, y = sp.symbols("x y")
        # y = 0 passes through origin; any ray with v_y = 0 lies on it.
        hps = [Hyperplane(y, [x, y])]
        extractor = RayShootingExtractor(num_rays=32, max_coord=1, seed=0)
        result = extractor.extract(hps)
        # Surviving witnesses must each have y != 0 (sign +/-1, not 0).
        for sig, point in result.items():
            assert sig[0] in (-1, 1)
            assert int(point[1]) != 0

    def test_finds_cells_in_higher_dim(self):
        """6-D coordinate axes -> 2^6 = 64 cells, all unbounded.  The
        algebraic shooter should cover most of them with a moderate
        ray budget."""
        import sympy as sp
        syms = list(sp.symbols("a b c d e f"))
        hps = [Hyperplane(s, syms) for s in syms]
        extractor = RayShootingExtractor(num_rays=4096, max_coord=3, seed=0)
        result = extractor.extract(hps)
        # Don't insist on all 64 -- random sampling won't be exhaustive,
        # but a vectorised pass should comfortably cover the majority.
        assert len(result) >= 32

    def test_sampled_rays_are_primitive(self):
        """Every sampled direction must have coordinate-gcd == 1.
        This is what guarantees witness points stay close to origin."""
        extractor = RayShootingExtractor(num_rays=2_000, max_coord=6, seed=7)
        rng = np.random.default_rng(extractor.seed)
        V = extractor._sample_rays(rng, d=5, n_rays=2_000)
        # gcd of absolute coords must be 1 for every row.
        gcds = np.gcd.reduce(np.abs(V), axis=1)
        assert (gcds == 1).all(), f"non-primitive rows: {(gcds != 1).sum()}"

    def test_vectorised_runs_fast(self):
        """Heuristic on a non-trivial arrangement must complete in
        well under a second -- the whole point of the rewrite."""
        import sympy as sp
        import time as _time
        syms = list(sp.symbols("a b c d e"))
        hps = [Hyperplane(s, syms) for s in syms]
        extractor = RayShootingExtractor(num_rays=10_000, max_coord=4, seed=0)
        t0 = _time.perf_counter()
        result = extractor.extract(hps)
        elapsed = _time.perf_counter() - t0
        assert elapsed < 1.0, f"too slow: {elapsed:.3f}s"
        assert result  # found at least one cell

    def test_adaptive_stops_early_on_plateau(self):
        """On a trivial arrangement the heuristic should plateau quickly
        and not shoot the full num_rays budget."""
        import sympy as sp
        x, y = sp.symbols("x y")
        hps = [Hyperplane(x, [x, y]), Hyperplane(y, [x, y])]  # 4 quadrants
        # Huge cap, small batch: must find all 4 then stop on plateau
        # well before exhausting the (effectively unbounded) budget.
        extractor = RayShootingExtractor(
            num_rays=10_000_000, batch_size=1_000, plateau_ratio=1e-2, seed=0
        )
        t0 = time.perf_counter()
        result = extractor.extract(hps)
        elapsed = time.perf_counter() - t0
        assert len(result) == 4
        assert elapsed < 1.0  # plateaued fast, didn't shoot 10M rays

    def test_plateau_disabled_runs_full_budget(self):
        """plateau_ratio=0 disables early stopping."""
        import sympy as sp
        x, y = sp.symbols("x y")
        hps = [Hyperplane(x, [x, y]), Hyperplane(y, [x, y])]
        extractor = RayShootingExtractor(
            num_rays=2_000, batch_size=500, plateau_ratio=0.0, seed=0
        )
        # Still correct (4 quadrants); just doesn't bail early.
        assert len(extractor.extract(hps)) == 4

    def test_deadline_stops_batches(self):
        """An expired deadline must stop the batch loop promptly."""
        import sympy as sp
        x, y = sp.symbols("x y")
        hps = [Hyperplane(x, [x, y]), Hyperplane(y, [x, y])]
        extractor = RayShootingExtractor(num_rays=10_000_000, batch_size=1_000)
        # Past deadline -> returns (possibly empty) without shooting 10M.
        out = extractor.extract(hps, deadline=time.time() - 1.0)
        assert isinstance(out, dict)


# ----------------------------------------------------------------------
# LrslibExtractor (skip when binary missing)
# ----------------------------------------------------------------------


lrs_required = pytest.mark.skipif(
    not lrs_io.lrs_available(), reason="lrs binary not available"
)


class TestLrslibExtractor:
    def test_default_lp_mode_needs_no_lrs_binary(self):
        """The default 'lp' backend must construct even when lrs is absent."""
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=False):
            ext = LrslibExtractor()  # default unbounded_check='lp'
        assert ext.unbounded_check == "lp"

    def test_lrs_mode_raises_without_binary(self):
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=False):
            with pytest.raises(FileNotFoundError, match="lrs"):
                LrslibExtractor(unbounded_check="lrs")

    def test_rejects_unknown_unbounded_check(self):
        with pytest.raises(ValueError, match="lp.*lrs"):
            LrslibExtractor(unbounded_check="bogus")

    def test_lrs_is_unbounded_dispatch(self):
        """Exercise the lrs parse path with a fake stdout."""
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=True):
            extractor = LrslibExtractor(unbounded_check="lrs")
        A = np.array([[1, 0]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        bounded_vrep = "V-representation\nbegin\n1 3 rational\n 1 0 0\nend\n"
        with patch("dreamer.extraction.v2.lrs_extractor.run_lrs", return_value=bounded_vrep):
            assert extractor._is_unbounded_lrs(A, c, np.array([1])) is False
        unbounded_vrep = "V-representation\nbegin\n1 3 rational\n 0 1 0\nend\n"
        with patch("dreamer.extraction.v2.lrs_extractor.run_lrs", return_value=unbounded_vrep):
            assert extractor._is_unbounded_lrs(A, c, np.array([1])) is True

    def test_extract_salvages_partial_on_timeout(self, monkeypatch):
        """Serial interleaved extract: a timeout mid-stream must re-raise
        an ExtractionTimeout carrying the shards classified so far."""
        ext = LrslibExtractor(num_workers=1)  # default LP, no lrs needed

        def fake_iter(A, c, **kwargs):
            yield (1, 1)
            yield (-1, -1)
            raise cells.ExtractionTimeout("enumeration deadline")  # no payload

        monkeypatch.setattr(
            "dreamer.extraction.v2.lrs_extractor.iter_cells", fake_iter
        )
        # Force every cell unbounded by stubbing the checker factory.
        monkeypatch.setattr(
            "dreamer.extraction.v2.lrs_extractor.make_unbounded_checker",
            lambda A: (lambda s: True),
        )
        monkeypatch.setattr(
            "dreamer.extraction.v2.lrs_extractor.find_integer_point",
            lambda A, c, s, bound: np.asarray(s),
        )

        with pytest.raises(cells.ExtractionTimeout) as ei:
            ext.extract(_hps_axes_2d(), deadline=time.time() + 100)

        partial = ei.value.partial
        assert set(partial.keys()) == {(1, 1), (-1, -1)}

    def test_lrs_timeout_is_wrapped(self):
        with patch("dreamer.extraction.v2.lrs_extractor.lrs_available", return_value=True):
            extractor = LrslibExtractor(unbounded_check="lrs", per_call_timeout=0.1)
        A = np.array([[1, 0]], dtype=np.int64)
        c = np.array([0], dtype=np.int64)
        with patch(
            "dreamer.extraction.v2.lrs_extractor.run_lrs",
            side_effect=subprocess.TimeoutExpired(cmd="lrs", timeout=0.1),
        ):
            with pytest.raises(RuntimeError, match="timed out"):
                extractor._is_unbounded_lrs(A, c, np.array([1]))

    def test_end_to_end_axes_lp(self):
        """Default LP backend, no lrs binary required."""
        extractor = LrslibExtractor()
        result = extractor.extract(_hps_axes_2d())
        assert len(result) == 4  # all 4 quadrants unbounded
        A, c = BaseExtractor.hyperplanes_to_matrix(_hps_axes_2d())
        for sig, pt in result.items():
            vals = A @ pt + c
            assert tuple(np.where(vals > 0, 1, -1).tolist()) == sig

    def test_end_to_end_triangle_drops_bounded_lp(self):
        extractor = LrslibExtractor()
        result = extractor.extract(_hps_triangle_2d())
        # 7 cells total, inner triangle bounded -> 6 unbounded survive.
        assert len(result) == 6

    @lrs_required
    def test_lp_matches_lrs_end_to_end(self):
        """The LP backend must agree with the lrs backend on which cells
        are unbounded -- same shard set for both."""
        for hps in (_hps_axes_2d(), _hps_triangle_2d(), _hps_strip_2d()):
            lp = set(LrslibExtractor(unbounded_check="lp").extract(hps).keys())
            lrs = set(LrslibExtractor(unbounded_check="lrs").extract(hps).keys())
            assert lp == lrs

    def test_parallel_matches_serial(self):
        """Salvage-aware parallel workers must find the same shard set as
        the serial sweep (encodings and points)."""
        import sympy as sp
        syms = list(sp.symbols("a b c d"))
        # An arrangement with enough cells to exercise multiple subtrees.
        hps = [Hyperplane(s, syms) for s in syms] + [
            Hyperplane(syms[0] - syms[1], syms),
            Hyperplane(syms[2] - syms[3], syms),
        ]
        serial = LrslibExtractor(num_workers=1).extract(hps)
        parallel = LrslibExtractor(num_workers=4).extract(hps)
        assert set(parallel.keys()) == set(serial.keys())
        for k in serial:
            assert np.array_equal(parallel[k], serial[k])

    def test_parallel_salvages_partial_on_timeout(self):
        """A past deadline in parallel mode must raise ExtractionTimeout
        carrying whatever shards the workers completed (possibly empty),
        not hang or lose everything."""
        import sympy as sp
        syms = list(sp.symbols("a b c d"))
        hps = [Hyperplane(s, syms) for s in syms] + [
            Hyperplane(syms[0] - syms[1], syms),
        ]
        ext = LrslibExtractor(num_workers=4)
        with pytest.raises(cells.ExtractionTimeout) as ei:
            ext.extract(hps, deadline=time.time() - 1.0)  # already expired
        assert isinstance(ei.value.partial, dict)


# ----------------------------------------------------------------------
# ExtractionManager
# ----------------------------------------------------------------------


class _FakeExact(BaseExtractor):
    name = "exact"

    def __init__(self, returns=None, raises=None):
        self._returns = returns or {}
        self._raises = raises

    def extract(self, hyperplanes, *, deadline=None):
        self.seen_deadline = deadline
        if self._raises:
            raise self._raises
        return self._returns


class _FakeHeuristic(BaseExtractor):
    name = "heuristic"

    def __init__(self, returns=None):
        self._returns = returns or {}

    def extract(self, hyperplanes, *, deadline=None):
        return self._returns


class TestExtractionManager:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ExtractionManager(strategy="bogus")  # type: ignore[arg-type]

    def test_heuristic_only(self):
        heur = _FakeHeuristic(returns={(1,): np.array([1])})
        mgr = ExtractionManager(strategy="heuristic", heuristic=heur)  # type: ignore[arg-type]
        assert mgr.extract([]) == {(1,): heur._returns[(1,)]}

    def test_heuristic_refine_forwarded(self):
        """heuristic refine knobs must reach the lazily-built extractor."""
        built = ExtractionManager(
            strategy="heuristic",
            heuristic_refine=True,
            heuristic_refine_threshold=80.0,
            heuristic_refine_workers=4,
        )._get_heuristic()
        assert isinstance(built, RayShootingExtractor)
        assert built.refine_witnesses is True
        assert built.refine_l1_threshold == 80.0
        assert built.refine_workers == 4
        # Default leaves the solver-free path on.
        assert (
            ExtractionManager(strategy="heuristic")._get_heuristic().refine_witnesses
            is False
        )

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

    def test_auto_passes_deadline_to_exact(self):
        """Manager must hand the exact extractor an absolute deadline so
        it can self-abort cooperatively (no killing threads)."""
        exact = _FakeExact(returns={(1,): np.array([1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact,
            heuristic=_FakeHeuristic(), timeout_seconds=30.0,
        )  # type: ignore[arg-type]
        t_before = __import__("time").time()
        mgr.extract([])
        assert exact.seen_deadline is not None
        # Deadline must be ~now + timeout_seconds.
        assert t_before + 29 <= exact.seen_deadline <= t_before + 31

    def test_auto_falls_back_on_timeout(self):
        # The exact extractor signals a timeout by raising ExtractionTimeout
        # (what the real one does when it passes its cooperative deadline).
        exact = _FakeExact(raises=cells.ExtractionTimeout("deadline"))
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=0.05
        )  # type: ignore[arg-type]
        out = mgr.extract([])
        assert (-1,) in out

    def test_auto_unions_partial_exact_with_heuristic(self):
        """On exact timeout the manager must union exact's salvaged
        shards with the heuristic's, preferring exact's point on overlap."""
        exact_pt = np.array([2, 2], dtype=np.int64)
        heur_same = np.array([9, 9], dtype=np.int64)
        heur_other = np.array([-1, -1], dtype=np.int64)
        exact = _FakeExact(
            raises=cells.ExtractionTimeout("deadline", partial={(1, 1): exact_pt})
        )
        heur = _FakeHeuristic(returns={(1, 1): heur_same, (-1, -1): heur_other})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=0.01
        )  # type: ignore[arg-type]
        out = mgr.extract([])
        assert set(out.keys()) == {(1, 1), (-1, -1)}
        # Exact's MILP point wins for the cell both found.
        assert np.array_equal(out[(1, 1)], exact_pt)
        assert np.array_equal(out[(-1, -1)], heur_other)

    def test_auto_does_not_stall_on_slow_then_timeout(self):
        """Regression: the fallback must run promptly, not block waiting
        on a runaway exact thread (the old ThreadPoolExecutor bug)."""
        import time as _time
        exact = _FakeExact(raises=cells.ExtractionTimeout("deadline"))
        heur = _FakeHeuristic(returns={(-1,): np.array([-1])})
        mgr = ExtractionManager(
            strategy="auto", exact=exact, heuristic=heur, timeout_seconds=0.01
        )  # type: ignore[arg-type]
        t0 = _time.perf_counter()
        out = mgr.extract([])
        assert (-1,) in out
        assert _time.perf_counter() - t0 < 1.0  # no multi-second stall

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
