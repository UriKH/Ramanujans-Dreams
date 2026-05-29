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

import time
from multiprocessing import get_context
from typing import List, Optional, Tuple

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane
from .base import BaseExtractor, ShardMapping
from .milp import find_integer_point


SignTuple = Tuple[int, ...]


def _refine_worker(args):
    """Top-level (picklable) MILP refinement of one cell, for the pool.

    Returns ``(key, point_or_None)``; ``None`` means the MILP found no
    integer point (should not happen for a cell that already has a ray
    witness, but the caller keeps the old witness if so).
    """
    A, c, key = args
    pt = find_integer_point(A, c, np.asarray(key, dtype=np.int64))
    return key, pt


class RayShootingExtractor(BaseExtractor):
    """
    Vectorised algebraic ray-shooting heuristic.

    :param num_rays: Maximum number of integer direction rays to sample
        (the cap; adaptive batching may stop earlier).  On low-D
        arrangements the plateau stop fires well before this; on high-D
        ones (where random-ray coverage does not saturate) this cap is the
        effective coverage/time budget.  Default ``2_000_000``.
    :param max_coord: Each ray direction is sampled uniformly from
        ``[-max_coord, max_coord]^D \\ {0}``.
    :param batch_size: Rays are shot in batches of this size; after each
        batch the new-cell yield is checked for the plateau stop.
    :param plateau_ratio: A batch is "low-yield" when it contributes fewer
        than ``plateau_ratio * batch_size`` *new* cells.  ``0`` disables
        early stopping (always shoot the full ``num_rays``).  Default
        ``1e-4`` (the old ``1e-3`` stopped ~10% short on D=5/7 — the cell
        tail is long and low-but-nonzero).
    :param plateau_patience: Stop only after this many *consecutive*
        low-yield batches (default ``3``).  The yield tail is lumpy, so a
        single low batch must not end the search; a sustained streak does.
    :param seed: RNG seed for reproducibility.
    :param refine_witnesses: When ``True``, after discovery polish the ray
        witnesses with the same MILP the exact extractor uses
        (:func:`dreamer.extraction.v2.milp.find_integer_point`), which
        returns the **L1-minimal** integer point of a cell.  To stay cheap
        this is applied **selectively**: only cells whose ray witness has
        ``L1 norm > refine_l1_threshold`` are recomputed (the rest are
        already small enough), so the MILP runs on the few far-out
        witnesses instead of all of them.  Default ``False`` keeps the
        solver-free fast path.
    :param refine_l1_threshold: Only ray witnesses with L1 norm strictly
        greater than this are refined (when ``refine_witnesses=True``).
        ``0`` refines every cell.  Default ``50``.
    :param refine_workers: Process count for the refinement MILPs (they are
        independent per cell).  ``>1`` dispatches them across processes;
        ``1`` (default) runs serially.
    :raises ValueError: For non-positive ``num_rays``/``max_coord``/
        ``batch_size`` or a ``plateau_ratio`` outside ``[0, 1]``.
    """

    name = "heuristic"

    # Below this many cells to refine, the process-pool overhead outweighs
    # the parallel speedup -- just refine serially.
    _PARALLEL_MIN = 200

    def __init__(
        self,
        *,
        num_rays: int = 2_000_000,
        max_coord: int = 5,
        batch_size: int = 20_000,
        plateau_ratio: float = 1e-4,
        plateau_patience: int = 3,
        seed: Optional[int] = 0,
        refine_witnesses: bool = False,
        refine_l1_threshold: float = 50.0,
        refine_workers: int = 1,
    ):
        if num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {num_rays}")
        if max_coord <= 0:
            raise ValueError(f"max_coord must be positive, got {max_coord}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0.0 <= plateau_ratio <= 1.0:
            raise ValueError(f"plateau_ratio must be in [0, 1], got {plateau_ratio}")
        if plateau_patience < 1:
            raise ValueError(f"plateau_patience must be >= 1, got {plateau_patience}")
        if refine_l1_threshold < 0:
            raise ValueError(
                f"refine_l1_threshold must be >= 0, got {refine_l1_threshold}"
            )
        self.num_rays = num_rays
        self.max_coord = max_coord
        self.batch_size = batch_size
        self.plateau_ratio = plateau_ratio
        self.plateau_patience = plateau_patience
        self.seed = seed
        self.refine_witnesses = refine_witnesses
        self.refine_l1_threshold = refine_l1_threshold
        self.refine_workers = refine_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        hyperplanes: List[Hyperplane],
        *,
        deadline: Optional[float] = None,
    ) -> ShardMapping:
        if not hyperplanes:
            return {}

        A, c = self.hyperplanes_to_matrix(hyperplanes)
        d = A.shape[1]
        rng = np.random.default_rng(self.seed)

        # Adaptive budgeting: shoot rays in batches and stop once a batch
        # adds few new cells (diminishing returns) -- so easy arrangements
        # finish fast while hard ones get the full ``num_rays``.  Each
        # batch is the same vectorised NumPy pass; only the loop over
        # *batches* is in Python (a handful of iterations).
        out: ShardMapping = {}
        shot = 0
        low_streak = 0
        while shot < self.num_rays:
            if deadline is not None and time.time() > deadline:
                break
            batch = min(self.batch_size, self.num_rays - shot)
            V = self._sample_rays(rng, d, batch)
            shot += batch
            if V.shape[0] == 0:
                continue
            points = self._shoot(V, A, c)
            before = len(out)
            self._collect_unique_cells_into(points, A, c, out)
            new = len(out) - before
            # Plateau stop: bail only after ``plateau_patience`` *consecutive*
            # batches each fall below the yield threshold.  The cell-yield tail
            # is lumpy (a batch can find nothing, then the next finds a few),
            # so stopping on the first low batch (the old behaviour) quit far
            # too early -- it cost ~10% of the shards on D=5/7.  Requiring a
            # sustained low streak captures that tail cheaply.
            if self.plateau_ratio > 0.0 and shot >= self.batch_size:
                if new < self.plateau_ratio * batch:
                    low_streak += 1
                    if low_streak >= self.plateau_patience:
                        break
                else:
                    low_streak = 0

        if self.refine_witnesses:
            self._refine_witnesses(A, c, out, deadline=deadline)
        return out

    def _refine_witnesses(
        self,
        A: np.ndarray,
        c: np.ndarray,
        out: ShardMapping,
        *,
        deadline: Optional[float] = None,
    ) -> None:
        """
        Replace **far-out** ray witnesses with the L1-minimal integer point
        of their cell (in place).

        Only cells whose ray witness has L1 norm ``> refine_l1_threshold``
        are recomputed -- the rest are already close enough to the origin,
        so the expensive MILP runs on the few outliers instead of every
        cell.  Every refined cell already contains an integer point (its ray
        witness), so the MILP is feasible and returns the closest-to-origin
        point -- never worse than what we had; a ``None`` result leaves the
        old witness untouched, so refinement can only improve the map.

        When ``refine_workers > 1`` and there are enough cells to amortise
        the pool, the (independent) MILPs are dispatched across processes.
        """
        to_refine = [
            key for key, pt in out.items()
            if np.abs(pt).sum() > self.refine_l1_threshold
        ]
        if not to_refine:
            return

        if (
            self.refine_workers
            and self.refine_workers > 1
            and len(to_refine) >= self._PARALLEL_MIN
        ):
            self._refine_parallel(A, c, out, to_refine)
            return

        for key in to_refine:
            if deadline is not None and time.time() > deadline:
                break
            pt = find_integer_point(A, c, np.asarray(key, dtype=np.int64))
            if pt is not None:
                out[key] = pt

    def _refine_parallel(
        self,
        A: np.ndarray,
        c: np.ndarray,
        out: ShardMapping,
        keys: List[SignTuple],
    ) -> None:
        """Refine ``keys`` across a process pool (each MILP is independent)."""
        tasks = [(A, c, key) for key in keys]
        chunksize = max(1, len(tasks) // (self.refine_workers * 4))
        ctx = get_context()
        with ctx.Pool(processes=min(self.refine_workers, len(tasks))) as pool:
            for key, pt in pool.imap_unordered(
                _refine_worker, tasks, chunksize=chunksize
            ):
                if pt is not None:
                    out[key] = pt

    # ------------------------------------------------------------------
    # Internals -- all pure NumPy, no Python loops over rays
    # ------------------------------------------------------------------

    def _sample_rays(
        self, rng: np.random.Generator, d: int, n_rays: int
    ) -> np.ndarray:
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
            -self.max_coord, self.max_coord + 1, size=(n_rays, d), dtype=np.int64
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

    def _collect_unique_cells_into(
        self,
        points: np.ndarray,
        A: np.ndarray,
        c: np.ndarray,
        out: ShardMapping,
    ) -> None:
        """
        Add the witnesses' ``{sign_tuple: point}`` entries into ``out``
        (in place).  When several rays land in the **same** cell, keep the
        witness **nearest the origin** (smallest L1 norm) -- different rays
        escape at very different ``t_final``, so the first to arrive is
        rarely the closest, and a smaller witness means a shorter downstream
        trajectory walk.  Points sitting exactly on any hyperplane (sign 0)
        are dropped -- not inside an open cell.
        """
        if points.shape[0] == 0:
            return
        # (R, N) sign matrix -- +1 above, -1 below, 0 on the hyperplane.
        vals = points @ A.T + c
        signs = np.sign(vals).astype(np.int64)

        for sign_row, point in zip(signs, points):
            if np.any(sign_row == 0):
                continue
            key: SignTuple = tuple(sign_row.tolist())
            pt = point.astype(np.int64)
            existing = out.get(key)
            if existing is None or np.abs(pt).sum() < np.abs(existing).sum():
                out[key] = pt
