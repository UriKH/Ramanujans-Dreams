from contextlib import contextmanager
from types import SimpleNamespace
from typing import cast

import pytest
import sympy as sp
from ramanujantools import Position

import dreamer.search.methods.genetic as genetic_method_mod
import dreamer.search.searchers.genetic_mod as genetic_searcher_mod
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.search.searchers.genetic_mod import GeneticSearchMod
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.storage_objects import DataManager, SearchData, SearchVector


class DummyCMF:
    def __init__(self):
        self.x, self.y = sp.symbols("x y")
        self.symbols = [self.x, self.y]

    def dim(self):
        return 2


class DummySpace:
    def __init__(self):
        self.cmf = DummyCMF()
        self.symbols = list(self.cmf.symbols)
        self.const = SimpleNamespace(name="dummy-const")

    def get_interior_point(self):
        x, y = self.cmf.symbols
        return Position({x: 0, y: 0})

    def in_space(self, _point):
        return True

    def is_valid_trajectory(self, _trajectory):
        return True

    def compute_trajectory_data(self, traj, start, **_kwargs):
        x, y = self.cmf.symbols
        score = float(100 - abs(int(traj[x]) - 2) - abs(int(traj[y]) + 1))
        sd = SearchData(SearchVector(start, traj))
        sd.delta = score
        return sd


class NoStartSpace(DummySpace):
    def get_interior_point(self):
        return None


class ConstrainedDummySpace(DummySpace):
    def is_valid_trajectory(self, trajectory):
        x, y = self.cmf.symbols
        return int(trajectory[x]) <= 0 and int(trajectory[y]) <= 0

    def compute_trajectory_data(self, traj, start, **_kwargs):
        assert self.is_valid_trajectory(traj), "GA evaluated a trajectory outside A v <= 0"
        x, y = self.cmf.symbols
        score = float(100 - abs(int(traj[x]) + 2) - abs(int(traj[y]) + 1))
        sd = SearchData(SearchVector(start, traj))
        sd.delta = score
        return sd


class ImpossibleConstrainedSpace(DummySpace):
    def is_valid_trajectory(self, _trajectory):
        return False


def test_genetic_search_known_answer_finds_expected_best(monkeypatch):
    space = DummySpace()
    x, y = space.cmf.symbols
    target = Position({x: 2, y: -1})
    seq = [
        target,
        Position({x: 1, y: -1}),
        Position({x: 3, y: -1}),
        Position({x: 2, y: -2}),
    ]

    def _fake_random_position_like(_template, _max_coord):
        return seq.pop(0) if seq else target

    monkeypatch.setattr(genetic_method_mod, "_random_position_like", _fake_random_position_like)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=2,
        pop_size=4,
        mutation_prob=0.0,
        crossover_prob=0.0,
        elite_fraction=0.5,
        parallel_eval=False,
        max_retries=0,
    )

    result = method.search(template_trajectory=Position({x: 1, y: -1}))
    best_delta, best_sv = result.best_delta

    assert best_delta == pytest.approx(100.0)
    assert best_sv is not None
    assert best_sv.trajectory == target


def test_genetic_search_uses_parallel_pool_when_enabled(monkeypatch):
    space = DummySpace()
    called = {"pool_used": False}

    class _DummyPool:
        def map(self, func, trajectories, starts, chunksize=1):
            return [func(t, s) for t, s in zip(trajectories, starts)]

    @contextmanager
    def _dummy_pool_context():
        called["pool_used"] = True
        yield _DummyPool()

    monkeypatch.setattr(genetic_method_mod, "create_pool", _dummy_pool_context)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=4,
        parallel_eval=True,
    )
    method.search(template_trajectory=Position({space.cmf.x: 1, space.cmf.y: -1}))

    assert called["pool_used"]


def test_genetic_search_requires_valid_start_point():
    method = GeneticSearchMethod(cast(Searchable, cast(object, NoStartSpace())), constant=None, parallel_eval=False)
    with pytest.raises(ValueError, match="requires a valid start point"):
        method.search()


def test_genetic_search_rejects_invalid_population_size():
    with pytest.raises(ValueError, match="pop_size"):
        GeneticSearchMethod(cast(Searchable, cast(object, DummySpace())), constant=None, pop_size=1)


def test_genetic_module_execute_exports_data(monkeypatch, tmp_path):
    exported = []
    space = DummySpace()

    def _identity_tqdm(iterable, *args, **kwargs):
        return iterable

    @contextmanager
    def _fake_export_stream(root, **_kwargs):
        assert str(tmp_path) in root

        def _writer(chunk, filename):
            exported.append((chunk, filename))

        yield _writer

    monkeypatch.setattr(genetic_searcher_mod, "SmartTQDM", _identity_tqdm)
    monkeypatch.setattr(genetic_searcher_mod.sys_config, "EXPORT_SEARCH_RESULTS", str(tmp_path))
    monkeypatch.setattr(genetic_searcher_mod.Exporter, "export_stream", _fake_export_stream)

    mod = GeneticSearchMod(
        [cast(Searchable, cast(object, space))],
        use_LIReC=True,
        generations=1,
        pop_size=4,
        parallel_eval=False,
    )
    mod.execute()

    assert len(exported) == 1
    assert isinstance(exported[0][0], DataManager)
    assert exported[0][1]


def test_genetic_search_repairs_invalid_mutations_with_constrained_sampling(monkeypatch):
    space = ConstrainedDummySpace()
    x, y = space.cmf.symbols
    valid_pool = [
        Position({x: -1, y: -1}),
        Position({x: -2, y: -1}),
        Position({x: -1, y: -2}),
    ]

    class _FakeSampler:
        def __init__(self, _space):
            pass

        def sample_trajectories(self, _compute_n_samples, exact=False):
            assert exact is True
            return set(valid_pool)

    def _always_invalid_mutate(_pos, **_kwargs):
        return Position({x: 4, y: 4})

    monkeypatch.setattr(genetic_method_mod, "ShardSamplingOrchestrator", _FakeSampler)
    monkeypatch.setattr(genetic_method_mod, "_mutate_position", _always_invalid_mutate)

    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=2,
        pop_size=4,
        mutation_prob=1.0,
        crossover_prob=0.0,
        parallel_eval=False,
    )

    result = method.search(template_trajectory=Position({x: -1, y: -1}))
    assert result
    assert result.best_delta[0] is not None
    assert all(space.is_valid_trajectory(sd.sv.trajectory) for sd in result.values())


def test_genetic_search_raises_when_constraints_have_no_valid_trajectories():
    space = ImpossibleConstrainedSpace()
    x, y = space.cmf.symbols
    method = GeneticSearchMethod(
        cast(Searchable, cast(object, space)),
        constant=None,
        generations=1,
        pop_size=2,
        max_retries=1,
        parallel_eval=False,
    )

    with pytest.raises(ValueError, match="could not sample enough valid trajectories"):
        method.search(template_trajectory=Position({x: 1, y: 1}))
