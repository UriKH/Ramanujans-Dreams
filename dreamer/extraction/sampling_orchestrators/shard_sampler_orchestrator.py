from __future__ import annotations

from typing import Callable, Set
import sympy as sp

from dreamer.extraction.shard import Shard
from dreamer.utils.caching import cached_property
from dreamer.utils.rand import np
from ramanujantools import Position

from dreamer.extraction.samplers.raycast_sampler import RaycastPipelineSampler
from dreamer.extraction.sampling_orchestrators.sampling_orchestrator import SamplingOrchestrator
from dreamer.extraction.samplers.sphere_sampler import PrimitiveSphereSampler


class ShardSamplingOrchestrator(SamplingOrchestrator):
    """Trajectory sampler for shards using the extraction sampling pipeline."""
    def __init__(self, searchable: Shard):
        super().__init__(searchable)
        if not isinstance(self.searchable, Shard):
            raise ValueError(f"{self.__class__.__name__} can only be used with {Shard.__name__} objects.")

        a_matrix = self.searchable.A
        if a_matrix is None:
            self.sampler = PrimitiveSphereSampler(len(self.searchable.symbols))
        else:
            self.sampler = RaycastPipelineSampler(np.asarray(a_matrix, dtype=np.float64))

    def sample_trajectories(self, compute_n_samples: Callable[[int], int] | int, *, exact: bool = False) -> Set[Position]:
        if isinstance(self.sampler, PrimitiveSphereSampler):
            samples = self.sampler.harvest(compute_n_samples)
        else:
            samples = self.sampler.harvest(compute_n_samples, exact=exact)

        return {
            Position({sym: sp.sympify(int(v)) for v, sym in zip(p, self.searchable.symbols)})
            for p in samples
        }

    @cached_property
    def search_space_dim(self):
        return self.sampler.d
