"""
Heuristic shard extractor -- *algebraic, vectorised* ray shooting.

Spec: ``context/shard_extraction/FAST_RAY_SHOOTING_SPEC.md``.

For each random integer direction ``v`` from the origin, the scalar
"escape time" past every hyperplane is computed in closed form:

    t_i = -c_i / (A_i . v)
    t_escape = max_i t_i
    t_final  = floor(t_escape) + 1
    witness  = t_final * v

This lands strictly past every crossing (one integer step beyond the
last one) without any solver call, scaling loop, or root rounding.
Because ``v`` is integer and ``t_final`` is integer, the witness has
exact integer coordinates -- no floating-point coordinates are ever
materialised.

The whole batch runs as three NumPy operations: ``V @ A.T`` for the
crossing matrix, an elementwise safe-divide for ``T``, and a row-wise
max for ``t_escape``.  No Python loop over rays, no scipy.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane
from .base import BaseExtractor, ShardMapping


SignTuple = Tuple[int, ...]


class RayShootingExtractor(BaseExtractor):
    """
    Vectorised algebraic ray-shooting heuristic.

    :param num_rays: How many integer direction rays to sample.
    :param max_coord: Each ray direction is sampled uniformly from
        ``[-max_coord, max_coord]^D \\ {0}``.
    :param seed: RNG seed for reproducibility.
    :raises ValueError: For non-positive ``num_rays`` or ``max_coord``.
    """

    name = "heuristic"

    def __init__(
        self,
        *,
        num_rays: int = 100_000,
        max_coord: int = 5,
        seed: Optional[int] = 0,
    ):
        if num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {num_rays}")
        if max_coord <= 0:
            raise ValueError(f"max_coord must be positive, got {max_coord}")
        self.num_rays = num_rays
        self.max_coord = max_coord
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        if not hyperplanes:
            return {}

        A, c = self.hyperplanes_to_matrix(hyperplanes)
        d = A.shape[1]
        rng = np.random.default_rng(self.seed)

        # Sample integer direction rays.  Re-roll any all-zero row in
        # place (acceptable: max_coord >= 1 so the all-zero outcome has
        # probability (1/(2 max_coord+1))^D -- vanishingly small past
        # D=3, and we only redo the few that hit it).
        V = self._sample_rays(rng, d)
        if V.shape[0] == 0:
            return {}

        points = self._shoot(V, A, c)
        return self._collect_unique_cells(points, A, c)

    # ------------------------------------------------------------------
    # Internals -- all pure NumPy, no Python loops over rays
    # ------------------------------------------------------------------

    def _sample_rays(self, rng: np.random.Generator, d: int) -> np.ndarray:
        """
        Return ``(R, D)`` integer rays with no all-zero rows, each
        reduced to its **primitive direction** (coordinate-GCD divided
        out).

        Primitive reduction keeps witnesses close to the origin and
        prevents "fat rays" -- e.g. ``(2, 4)`` and ``(1, 2)`` collapse
        to the same direction with witness magnitude halved, which
        cascades into faster trajectory walks downstream.
        """
        V = rng.integers(
            -self.max_coord, self.max_coord + 1, size=(self.num_rays, d), dtype=np.int64
        )
        # Reroll any all-zero rays until none remain (cheap: rare in
        # practice, the loop iterates at most once or twice).
        for _ in range(8):
            zero_rows = np.where(~V.any(axis=1))[0]
            if zero_rows.size == 0:
                break
            V[zero_rows] = rng.integers(
                -self.max_coord, self.max_coord + 1,
                size=(zero_rows.size, d), dtype=np.int64,
            )
        # Drop any all-zero rows that survived the reroll budget.
        V = V[V.any(axis=1)]
        # Primitive reduction: divide each row by the gcd of |entries|.
        # ``np.gcd.reduce`` returns 0 only for an all-zero row -- we
        # already dropped those, but the ``gcds == 0`` guard remains as
        # defensive belt-and-suspenders against the reroll exhausting.
        gcds = np.gcd.reduce(np.abs(V), axis=1, keepdims=True)
        gcds[gcds == 0] = 1
        return V // gcds

    def _shoot(
        self, V: np.ndarray, A: np.ndarray, c: np.ndarray
    ) -> np.ndarray:
        """
        Compute the integer witness point for every ray.

        Vectorised over rays:

        * ``M = V @ A.T``  -- shape ``(R, N)``;  ``M[r, i] = v_r . A_i``
        * ``T = -c / M`` with safe-divide; parallel rays
          (``M[r, i] == 0``) get ``0`` per the spec -- they impose no
          escape constraint on this ray.
        * Discard rays that lie exactly on a hyperplane
          (``M[r, i] == 0 and c[i] == 0``  =>  ``A_i . (t v) + c_i == 0``
          for every ``t``).
        * ``t_escape = max(T, axis=1)``;  ``t_final = floor + 1``;
          clipped to ``>= 1`` so a negative ``t_escape`` (all crossings
          behind origin) still steps forward along ``v`` rather than
          backwards.
        """
        # (R, N) = (R, D) @ (D, N)
        M = V @ A.T

        # Identify rays that lie ON some hyperplane.  Drop them -- no
        # finite ``t`` can move such a ray off that hyperplane.
        on_hyperplane = (M == 0) & (c[None, :] == 0)
        keep = ~on_hyperplane.any(axis=1)
        if not keep.any():
            return np.zeros((0, V.shape[1]), dtype=np.int64)
        V = V[keep]
        M = M[keep]

        # Safe divide: parallel hyperplanes (M == 0, c != 0) impose no
        # escape constraint -- substitute 0 so they don't affect the
        # row-wise max (which is dominated by real positive crossings).
        T = np.divide(
            -c[None, :].astype(np.float64),
            M.astype(np.float64),
            out=np.zeros((V.shape[0], M.shape[1]), dtype=np.float64),
            where=M != 0,
        )

        t_escape = T.max(axis=1)
        # floor(t_escape) + 1 lands strictly past every crossing; clip
        # to >= 1 so we always advance forward along ``v`` even when
        # every crossing happened at negative ``t``.
        t_final = np.maximum(np.floor(t_escape).astype(np.int64) + 1, 1)

        # Broadcast scalar t per row over the D-vector v per row.
        return (t_final[:, None] * V).astype(np.int64)

    def _collect_unique_cells(
        self, points: np.ndarray, A: np.ndarray, c: np.ndarray
    ) -> ShardMapping:
        """
        Convert ``(R, D)`` integer witnesses into a deduplicated
        ``{sign_tuple: point}`` map.  Points sitting exactly on any
        hyperplane (sign 0) are dropped -- those would not be inside an
        open cell.
        """
        if points.shape[0] == 0:
            return {}
        # (R, N) sign matrix -- +1 above, -1 below, 0 on the hyperplane.
        vals = points @ A.T + c
        signs = np.sign(vals).astype(np.int64)

        out: ShardMapping = {}
        # One Python loop over UNIQUE cells discovered, not over rays.
        # `np.unique(..., axis=0, return_index=True)` would do this in
        # one call but we additionally need to drop sign-0 rows, so a
        # short explicit loop is clearer.
        for sign_row, point in zip(signs, points):
            if np.any(sign_row == 0):
                continue
            key: SignTuple = tuple(sign_row.tolist())
            out.setdefault(key, point.astype(np.int64))
        return out
