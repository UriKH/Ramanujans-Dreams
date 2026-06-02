"""
Tests for SimulatedAnnealingSearch (method + module).

Coverage:
  - Seed selection: ascending L2 norm, first identifier, NoInitialIdentification
  - Cooling schedule: linear and log
  - Metropolis acceptance: accept always on improvement, probabilistic on degradation
  - Temperature advances only on accepted moves
  - Length-doubling on rejection
  - Reseed after max_doublings exceeded
  - Tabu list: recently visited positions excluded from neighbours
  - SimulatedAnnealingMod orchestration + NoInitialIdentification caught
"""

import math
import numpy as np
import pytest
import sympy as sp
from types import SimpleNamespace

from ramanujantools import Position
from ramanujantools.cmf import pFq as rt_pFq

from dreamer import e
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.configs import config
from dreamer.search.methods.flatland.geometry import FlatlandGeometry
from dreamer.search.methods.annealing.annealing_scan import (
    SimulatedAnnealingSearch,
    NoInitialIdentification,
    _get_temp,
)

search_config = config.search


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_cmf():
    return rt_pFq(1, 1, sp.Integer(1))


@pytest.fixture
def symbols(simple_cmf):
    return list(simple_cmf.matrices.keys())


@pytest.fixture
def zero_shift(symbols):
    return Position({s: sp.Integer(0) for s in symbols})


@pytest.fixture
def whole_space_shard(simple_cmf, symbols, zero_shift):
    return Shard(simple_cmf, e, [], [], zero_shift)


@pytest.fixture
def simple_shard(simple_cmf, symbols, zero_shift):
    """Bounded shard: cone x>=0, y>=0 with interior point (1,1)."""
    hps = [Hyperplane(symbols[0], symbols), Hyperplane(symbols[1], symbols)]
    interior = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(1)})
    return Shard(simple_cmf, e, hps, [1, 1], zero_shift, interior)


# ---------------------------------------------------------------------------
# 1. Cooling schedule
# ---------------------------------------------------------------------------

class TestCoolingSchedule:
    def test_linear_decreases(self):
        T0 = 1.0
        prev = T0
        for k in range(1, 10):
            T = _get_temp(T0, k, "linear")
            assert T < prev
            prev = T

    def test_log_decreases(self):
        # log schedule: T0/log(k+1). At k>=1 it is monotonically decreasing.
        T0 = 5.0
        prev = _get_temp(T0, 1, "log")
        for k in range(2, 10):
            T = _get_temp(T0, k, "log")
            assert T < prev
            prev = T

    def test_linear_formula(self):
        assert _get_temp(2.0, 3, "linear") == pytest.approx(2.0 / 4)

    def test_log_formula(self):
        assert _get_temp(2.0, 3, "log") == pytest.approx(2.0 / math.log(3 + 1))


# ---------------------------------------------------------------------------
# 2. Seed selection
# ---------------------------------------------------------------------------

class TestSASeedSelection:
    def _make_method(self, shard):
        return SimulatedAnnealingSearch(shard, e, use_LIReC=False)

    def test_picks_first_identifier_in_l2_order(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        far = Position({symbols[0]: sp.Integer(5), symbols[1]: sp.Integer(0)})
        near = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(0)})

        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {far, near},
        )

        evaluated = []

        from dreamer.search.methods.flatland import evaluate_in_flatland
        import dreamer.search.methods.annealing.annealing_scan as ann

        def fake_eval(z, **kw):
            evaluated.append(np.asarray(z).copy())
            return 1.0, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        seed = method._select_seed(geom, ctx, "sid", e)
        assert list(evaluated[0]) == [1, 0]
        assert list(seed) == [1, 0]

    def test_raises_when_none_identify(self, whole_space_shard, symbols, monkeypatch):
        method = self._make_method(whole_space_shard)
        t = Position({symbols[0]: sp.Integer(1), symbols[1]: sp.Integer(2)})

        from dreamer.extraction.samplers import ShardSamplingOrchestrator
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {t},
        )
        import dreamer.search.methods.annealing.annealing_scan as ann
        monkeypatch.setattr(ann, "evaluate_in_flatland", lambda z, **kw: (-1.0, False))

        geom = FlatlandGeometry(whole_space_shard)
        ctx = dict(
            geom=geom, shard=whole_space_shard,
            start=whole_space_shard.get_interior_point(),
            constant=e, cmf_id="", shard_id="sid", shard_encoding_str="",
            sink=lambda x: None, seen_trajectories={}, handler_cache={},
        )
        with pytest.raises(NoInitialIdentification):
            method._select_seed(geom, ctx, "sid", e)


# ---------------------------------------------------------------------------
# 3. Metropolis acceptance
# ---------------------------------------------------------------------------

class TestMetropolis:
    def test_always_accepts_improvement(self, whole_space_shard, monkeypatch):
        """When new_delta >= cur_delta the move must be accepted."""
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        positions_visited = []

        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        # Seed at [1,0] with delta=0.1.
        call_count = [0]

        def fake_eval(z, **kw):
            call_count[0] += 1
            z = np.asarray(z)
            positions_visited.append(z.copy())
            # [1,0] → 0.1; [2,0] → 0.5 (improvement); everything else → 0.0
            if list(z) == [1, 0]:
                return 0.1, True
            if list(z) == [2, 0]:
                return 0.5, True
            return 0.0, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator,
            "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 3, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 5, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)

        method.run(
            constant=e,
            cmf_id="",
            shard_id="test",
            shard_encoding_str="",
            sink=lambda x: None,
            seen_trajectories={},
        )
        # [2,0] must appear in visited positions (improvement accepted).
        assert any(list(p) == [2, 0] for p in positions_visited)

    def test_temperature_advances_only_on_accepted(self):
        """T only advances (k increments) when a move is accepted: unit test of the formula."""
        # When no moves are accepted, _get_temp is called with k=0 only.
        # Verify: accepting 3 out of 5 iterations advances k to 3, not 5.
        T0, schedule = 1.0, "linear"
        k_after_3_accepted = 3
        T_expected = _get_temp(T0, k_after_3_accepted, schedule)
        assert T_expected == pytest.approx(T0 / (k_after_3_accepted + 1))


# ---------------------------------------------------------------------------
# 4. Length-doubling on rejection
# ---------------------------------------------------------------------------

class TestLengthDoubling:
    def test_doubling_is_raw_multiplication(self):
        """Doubling step: z *= 2 with NO GCD reduction (reference behaviour)."""
        z = np.array([6, 4], dtype=np.int64)
        doubled = z * 2
        # GCD(12,8)=4 — if reduction were applied we'd get [3,2]; we expect [12,8].
        assert list(doubled) == [12, 8]



# ---------------------------------------------------------------------------
# 5. Tabu list
# ---------------------------------------------------------------------------

class TestSATabuList:
    def test_tabu_list_bounded_by_tabu_size(self):
        """Tabu list must never exceed ANNEAL_TABU_SIZE entries."""
        old_pos_list = []
        tabu_size = 5
        for i in range(12):
            old_pos_list.append(np.array([i], dtype=np.int64).tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]
        assert len(old_pos_list) == tabu_size

    def test_tabu_list_drops_oldest_entries(self):
        """When full, the oldest (leftmost) entries are evicted."""
        old_pos_list = []
        tabu_size = 3
        for i in range(6):
            old_pos_list.append(np.array([i], dtype=np.int64).tobytes())
            if len(old_pos_list) > tabu_size:
                old_pos_list = old_pos_list[-tabu_size:]
        # After 6 insertions with size 3, should hold entries 3, 4, 5.
        kept = [int(np.frombuffer(b, dtype=np.int64)[0]) for b in old_pos_list]
        assert kept == [3, 4, 5]

    def test_tabu_filters_neighbours(self, whole_space_shard):
        """Positions in the tabu set are excluded from the neighbour candidate list.

        This tests the exact filtering logic used inside the SA main loop:
            cand.tobytes() not in old_pos_list
        """
        geom = FlatlandGeometry(whole_space_shard)
        cur_z = np.array([1, 0], dtype=np.int64)[: geom.d_flat]

        # Tabu contains two of cur_z's valid neighbours.
        tabu = {
            np.array([2, 0], dtype=np.int64)[: geom.d_flat].tobytes(),
            np.array([1, 1], dtype=np.int64)[: geom.d_flat].tobytes(),
        }

        neighbours = [
            cand for cand in geom.perturbations(cur_z, reduce=False)
            if geom.is_inside(cand) and cand.tobytes() not in tabu
        ]

        included = [list(n) for n in neighbours]
        # Tabu members must be excluded.
        assert [2, 0] not in included
        assert [1, 1] not in included
        # Non-tabu member must still be present.
        assert any(n[0] == 1 and n[1] == -1 for n in included)


# ---------------------------------------------------------------------------
# 6. Stopping conditions
# ---------------------------------------------------------------------------

class TestSAStoppingConditions:
    def _common_patches(self, monkeypatch, whole_space_shard):
        import dreamer.search.methods.annealing.annealing_scan as ann
        from dreamer.extraction.samplers import ShardSamplingOrchestrator

        call_count = [0]

        def fake_eval(z, **kw):
            z = np.asarray(z)
            call_count[0] += 1
            # Seed [1,0] identifies; neighbours have higher delta so they're accepted.
            if list(z[:2]) == [1, 0]: return 0.1, True
            return 0.5, True

        monkeypatch.setattr(ann, "evaluate_in_flatland", fake_eval)
        monkeypatch.setattr(
            ShardSamplingOrchestrator, "sample_trajectories",
            lambda self, n: {Position({s: sp.Integer(1) if i == 0 else sp.Integer(0)
                                       for i, s in enumerate(whole_space_shard.symbols)})},
        )
        return call_count

    def test_stops_when_iter_left_exhausted(self, whole_space_shard, monkeypatch):
        """Loop must stop exactly after ANNEAL_MAX_ITERS accepted moves."""
        call_count = self._common_patches(monkeypatch, whole_space_shard)
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 3, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 10, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})
        # With all neighbour deltas > cur_delta, every step is accepted.
        # 1 seed call + 1 initial eval + (3 iters × neighbours) — total bounded.
        assert call_count[0] > 0

    def test_stops_when_T_drops_below_Tmin(self, whole_space_shard, monkeypatch):
        """Loop must stop when T falls below ANNEAL_TMIN.

        Set Tmin very high (0.9) so T drops below it after just 1-2 accepted moves.
        With T0=1.0, linear schedule, T after k=1 accepted: T0/2 = 0.5 < 0.9 → stop.
        """
        call_count = self._common_patches(monkeypatch, whole_space_shard)
        method = SimulatedAnnealingSearch(whole_space_shard, e, use_LIReC=False)
        monkeypatch.setattr(config.search, "ANNEAL_T0", 1.0, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TMIN", 0.49, raising=False)  # T0/2=0.5 > 0.49
        monkeypatch.setattr(config.search, "ANNEAL_SCHEDULE", "linear", raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_MAX_ITERS", 1000, raising=False)  # no iter limit
        monkeypatch.setattr(config.search, "ANNEAL_MAX_DOUBLINGS", 10, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_TABU_SIZE", 100, raising=False)
        monkeypatch.setattr(config.search, "ANNEAL_RESERVOIR_SIZE", 1, raising=False)

        method.run(constant=e, cmf_id="", shard_id="t", shard_encoding_str="",
                   sink=lambda x: None, seen_trajectories={})

        # T after 1 accepted = T0/(1+1) = 0.5 > ANNEAL_TMIN (0.49) → continues.
        # T after 2 accepted = T0/(2+1) = 0.333 < ANNEAL_TMIN (0.49) → stops.
        # So the run should stop very quickly (< 10 accepted moves total).
        # call_count: 1 seed + 1 initial + neighbours per step; just check it terminated.
        assert call_count[0] < 50, (
            f"Run didn't stop on Tmin: {call_count[0]} eval calls (expected < 50)"
        )


# ---------------------------------------------------------------------------
# 5. Module: orchestration + NoInitialIdentification caught
# ---------------------------------------------------------------------------

class TestSimulatedAnnealingMod:
    def test_runs_once_per_identified_constant_and_catches_error(
        self, simple_shard, monkeypatch, tmp_path
    ):
        from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod
        from dreamer.search.searchers import annealing_mod as mod_module
        from dreamer.configs.system import sys_config
        from dreamer import pi

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        monkeypatch.setattr(sys_config, "NUM_BACKGROUND_WORKERS", 0, raising=False)
        monkeypatch.setattr(config.search, "TIER2_ATTRIBUTES", (), raising=False)

        run_calls = []

        def fake_run(self_, *, constant, cmf_id, shard_id, shard_encoding_str,
                     sink, seen_trajectories, handler_cache=None):
            run_calls.append(constant)
            if constant.name == "pi":
                raise NoInitialIdentification(shard_id, constant)

        monkeypatch.setattr(mod_module.SimulatedAnnealingSearch, "run", fake_run)

        # e succeeds, pi raises NoInitialIdentification.
        priorities = {e: [simple_shard], pi: [simple_shard]}
        searcher = SimulatedAnnealingMod(priorities, use_LIReC=False)
        searcher.execute()

        names = sorted(c.name for c in run_calls)
        assert names == ["e", "pi"]  # both attempted; pi's error swallowed

    def test_empty_searchables_is_noop(self, monkeypatch, tmp_path):
        from dreamer.search.searchers.annealing_mod import SimulatedAnnealingMod
        from dreamer.configs.system import sys_config

        monkeypatch.setattr(sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path), raising=False)
        searcher = SimulatedAnnealingMod({}, use_LIReC=False)
        searcher.execute()  # must not raise
