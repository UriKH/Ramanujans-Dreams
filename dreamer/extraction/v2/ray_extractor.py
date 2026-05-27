"""
Heuristic ray-shooting shard extractor.

Strategy: shoot many integer-direction rays outward from the origin
and inspect the sign vector of points along each ray.

A ray ``r`` (integer direction) starts at the origin -- which itself
lies in some cell -- and as ``t`` grows it crosses one hyperplane at a
time, entering a new cell at each crossing.  Once ``t`` is large enough
to lie *past* every hyperplane the ray will ever cross, the sign
vector stops changing: that final sign vector labels an unbounded
cell, and the current integer point is a valid representative.

This explores cells reached by some ray from the origin -- partial by
design.  Cells separated from the origin by hyperplanes that no
sampled ray crosses are missed, which is the deliberate trade-off the
plan calls out.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane
from .base import BaseExtractor, ShardMapping


class RayShootingExtractor(BaseExtractor):
    """
    Heuristic strategy: random integer rays from the origin.

    :param num_rays: How many rays to shoot.
    :param max_coord: Each ray direction is sampled uniformly from
        ``[-max_coord, max_coord]^D \\ {0}``.
    :param escape_scale_start: First integer ``t`` checked along the
        ray.  Doubled each step until either the sign vector stabilises
        or ``escape_scale_max`` is exceeded.
    :param escape_scale_max: Upper bound on ``t``; beyond this the ray
        is discarded.
    :param seed: RNG seed for reproducibility.
    """

    name = "heuristic"

    def __init__(
        self,
        *,
        num_rays: int = 4096,
        max_coord: int = 5,
        escape_scale_start: int = 1,
        escape_scale_max: int = 2**20,
        seed: Optional[int] = 0,
    ):
        if num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {num_rays}")
        if max_coord <= 0:
            raise ValueError(f"max_coord must be positive, got {max_coord}")
        if escape_scale_start <= 0:
            raise ValueError(
                f"escape_scale_start must be positive, got {escape_scale_start}"
            )
        if escape_scale_max < escape_scale_start:
            raise ValueError(
                f"escape_scale_max ({escape_scale_max}) < "
                f"escape_scale_start ({escape_scale_start})"
            )
        self.num_rays = num_rays
        self.max_coord = max_coord
        self.escape_scale_start = escape_scale_start
        self.escape_scale_max = escape_scale_max
        self.seed = seed

    def extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        if not hyperplanes:
            return {}

        A, c = self.hyperplanes_to_matrix(hyperplanes)
        d = A.shape[1]
        rng = np.random.default_rng(self.seed)
        out: ShardMapping = {}

        for _ in range(self.num_rays):
            direction = rng.integers(
                -self.max_coord, self.max_coord + 1, size=d, dtype=np.int64
            )
            if not np.any(direction):
                continue
            result = self._escape_ray(A, c, direction)
            if result is None:
                continue
            sig, point = result
            out.setdefault(sig, point)
        return out

    def _escape_ray(
        self, A: np.ndarray, c: np.ndarray, direction: np.ndarray
    ):
        """
        Walk along ``direction`` doubling the scale until the sign
        vector stabilises and is on no hyperplane.

        :return: ``(sign_tuple, integer_point)`` or :data:`None`.
        """
        prev_sig = None
        prev_point = None
        t = self.escape_scale_start
        while t <= self.escape_scale_max:
            point = (direction * t).astype(np.int64)
            vals = A @ point + c
            if np.any(vals == 0):
                t *= 2
                continue
            sig = np.where(vals > 0, 1, -1).astype(np.int64)
            if prev_sig is not None and np.array_equal(sig, prev_sig):
                return tuple(sig.tolist()), prev_point
            prev_sig = sig
            prev_point = point
            t *= 2
        return None
