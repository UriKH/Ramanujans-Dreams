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

Two generation **phases** feed the same batch runner:

* **generic** origin shooting (above) -- reaches every cell whose
  recession cone is *full-dimensional* (non-zero solid angle on the unit
  sphere).
* **face-aligned** shooting (optional) -- shoots along directions in the
  integer nullspace of a random hyperplane subset, from random integer
  offsets, to reach unbounded cells whose recession cone is
  *lower-dimensional* (tubes/slabs).  These have zero solid angle, so no
  origin ray hits them at any ray count.

Each phase stops independently when its **missing mass** (the probability
that a fresh sample lands in a never-seen cell) plateaus -- estimated by
the Good-Turing statistic ``f1 / n`` (``f1`` = cells seen exactly once,
``n`` = samples that landed in a cell).  Time and an optional ray cap are
shared global budgets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import sympy as sp

from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.logger import Logger
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


def integer_nullspace(A_sub: np.ndarray) -> List[np.ndarray]:
    """Integer basis of the nullspace of ``A_sub`` (rows = hyperplane normals).

    Uses sympy's exact rational nullspace, then clears denominators and
    divides out the gcd so each basis vector is primitive-integer.  An
    empty ``A_sub`` (no constraints) returns the standard basis -- the
    whole space is the nullspace.
    """
    d = A_sub.shape[1]
    if A_sub.shape[0] == 0:
        return [np.eye(d, dtype=np.int64)[i] for i in range(d)]
    M = sp.Matrix(A_sub.tolist())
    out: List[np.ndarray] = []
    for vec in M.nullspace():
        # Clear denominators -> integer vector -> divide by gcd so the
        # direction is primitive (keeps witnesses near the origin).
        lcm = sp.ilcm(*[term.q for term in vec]) if vec.free_symbols == set() else 1
        ints = [int(term * lcm) for term in vec]
        g = int(np.gcd.reduce([abs(x) for x in ints])) or 1
        out.append(np.array([x // g for x in ints], dtype=np.int64))
    return out


def _shoot_from(
    p: np.ndarray, v: np.ndarray, A: np.ndarray, c: np.ndarray
) -> Optional[np.ndarray]:
    """Shoot a ray from integer point ``p`` along integer direction ``v``.

    Returns the integer witness ``p + t_final * v`` strictly past the last
    crossing, or ``None`` if the witness lands exactly on a hyperplane.

    Hyperplanes parallel to ``v`` (``A_i . v == 0``) are not crossed at any
    finite ``t``; their sign at the witness is fixed by ``p`` instead.  This
    is exactly what lets face-aligned shooting reach the slab cells that
    share recession direction ``v`` -- sweeping ``p`` flips those signs.
    """
    denom = A @ v                       # (N,)  A_i . v
    rhs = -(A @ p + c)                  # (N,)  -(A_i . p + c_i)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom != 0, rhs / denom, -np.inf)
    t_escape = t.max() if t.size else 0.0
    if not np.isfinite(t_escape):
        t_escape = 0.0
    t_final = max(int(np.floor(t_escape)) + 1, 1)
    w = p + t_final * v
    if np.any(A @ w + c == 0):
        return None
    return w


@dataclass
class _GTStats:
    """Good-Turing accumulator for one phase.

    ``f1`` = number of distinct cells seen *exactly once*, ``n`` = number
    of samples that landed in a cell.  The missing-mass estimate is
    ``f1 / n``.  Both are maintained in O(1) per landed sample.
    """

    f1: int = 0
    n: int = 0


@dataclass
class _Budget:
    """Shared global budget across all generation phases.

    ``shot`` (samples processed) and the wall-clock are shared, so total
    time and the optional ray cap span both phases.  The per-phase
    Good-Turing state is *not* here -- it is fresh per phase.
    """

    deadline: Optional[float] = None
    max_seconds: Optional[float] = None
    num_rays: Optional[int] = None
    shot: int = 0
    cap_hit: bool = False
    t_start: float = field(default_factory=time.time)

    def exhausted(self) -> bool:
        now = time.time()
        if self.deadline is not None and now > self.deadline:
            return True
        if self.max_seconds is not None and now - self.t_start > self.max_seconds:
            return True
        if self.num_rays is not None and self.shot >= self.num_rays:
            self.cap_hit = True
            return True
        return False


class RayShootingExtractor(BaseExtractor):
    """
    Vectorised algebraic ray-shooting heuristic.

    :param num_rays: Optional hard ceiling on the number of samples
        processed (a safety cap).  ``None`` (default) = unlimited; the
        missing-mass plateau and/or ``max_seconds`` govern instead.  Set a
        finite value only to bound a run regardless of saturation.
    :param max_seconds: Optional wall-clock budget (seconds) for the whole
        shoot (all phases combined).  ``None`` (default) = no time cap.
        This is the recommended primary limiter for high-D scans where the
        space never saturates.
    :param max_coord: Each ray direction / face-aligned offset is sampled
        uniformly from ``[-max_coord, max_coord]^D``.
    :param batch_size: Samples are processed in batches of this size; the
        missing mass is assessed after each batch.
    :param missing_mass: Stop a phase once its **missing-mass** estimate --
        the Good-Turing ``f1 / n`` (fraction of samples that landed in a
        cell seen exactly once) -- stays below this fraction for
        ``plateau_patience`` consecutive batches.  This estimates the
        probability that a fresh sample discovers a new cell, so it is
        scale-invariant and robust to the "plateau then spike" failure of
        a naive running-total ratio: a still-large reservoir keeps ``f1``
        high and refuses to stop.  Default ``5e-4`` (stop at ~0.05% of
        samples discovering new cells).  ``0`` disables early stopping.
    :param plateau_patience: Number of *consecutive* sub-``missing_mass``
        batches required before stopping a phase (default ``3``).  The
        yield tail is lumpy, so a single low batch must not end the search.
    :param seed: RNG seed for reproducibility.
    :param face_aligned: When ``True``, run a second **face-aligned**
        shooting phase after generic shooting to reach unbounded cells with
        lower-dimensional recession cones (tubes/slabs) that origin rays
        structurally miss.  Default ``False``.
    :param face_subsets: Number of random hyperplane subsets to draw in the
        face-aligned phase (each yields one or more nullspace directions).
        Default ``200``.
    :param face_offsets: Number of random integer start offsets to sweep
        per nullspace direction.  Sweeping offsets flips the signs of the
        subset's hyperplanes, enumerating the slab cells that share the
        direction.  Default ``50``.
    :param refine_witnesses: When ``True``, after discovery polish the ray
        witnesses with the same MILP the exact extractor uses
        (:func:`dreamer.extraction.v2.milp.find_integer_point`), which
        returns the **L1-minimal** integer point of a cell.  To stay cheap
        this is applied **selectively**: only cells whose ray witness has
        ``L1 norm > refine_l1_threshold`` are recomputed.  Default
        ``False`` keeps the solver-free fast path.
    :param refine_l1_threshold: Only ray witnesses with L1 norm strictly
        greater than this are refined (when ``refine_witnesses=True``).
        ``0`` refines every cell.  Default ``50``.
    :param refine_workers: Process count for the refinement MILPs (they are
        independent per cell).  ``>1`` dispatches them across processes;
        ``1`` (default) runs serially.
    :raises ValueError: For non-positive ``num_rays``/``max_coord``/
        ``batch_size``/``max_seconds``/``face_subsets``/``face_offsets``, a
        ``missing_mass`` outside ``[0, 1]``, or ``plateau_patience < 1``.
    """

    name = "heuristic"

    # Below this many cells to refine, the process-pool overhead outweighs
    # the parallel speedup -- just refine serially.
    _PARALLEL_MIN = 200

    def __init__(
        self,
        *,
        num_rays: Optional[int] = None,
        max_seconds: Optional[float] = None,
        max_coord: int = 5,
        batch_size: int = 20_000,
        missing_mass: float = 5e-4,
        plateau_patience: int = 3,
        seed: Optional[int] = 0,
        face_aligned: bool = False,
        face_subsets: int = 200,
        face_offsets: int = 50,
        refine_witnesses: bool = False,
        refine_l1_threshold: float = 50.0,
        refine_workers: int = 1,
    ):
        if num_rays is not None and num_rays <= 0:
            raise ValueError(f"num_rays must be positive or None, got {num_rays}")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError(f"max_seconds must be positive or None, got {max_seconds}")
        if max_coord <= 0:
            raise ValueError(f"max_coord must be positive, got {max_coord}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0.0 <= missing_mass <= 1.0:
            raise ValueError(
                f"missing_mass must be in [0, 1], got {missing_mass}"
            )
        if plateau_patience < 1:
            raise ValueError(f"plateau_patience must be >= 1, got {plateau_patience}")
        if face_subsets <= 0:
            raise ValueError(f"face_subsets must be positive, got {face_subsets}")
        if face_offsets <= 0:
            raise ValueError(f"face_offsets must be positive, got {face_offsets}")
        if refine_l1_threshold < 0:
            raise ValueError(
                f"refine_l1_threshold must be >= 0, got {refine_l1_threshold}"
            )
        self.num_rays = num_rays
        self.max_seconds = max_seconds
        self.max_coord = max_coord
        self.batch_size = batch_size
        self.missing_mass = missing_mass
        self.plateau_patience = plateau_patience
        self.seed = seed
        self.face_aligned = face_aligned
        self.face_subsets = face_subsets
        self.face_offsets = face_offsets
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
        rng = np.random.default_rng(self.seed)

        # Each generation strategy is its own phase with a *fresh*
        # Good-Turing counter, so a plateau in generic shooting does not
        # stop the face-aligned phase (which samples a different
        # distribution and may still hold a reservoir).  Time and the
        # optional ray cap are shared via ``budget``.
        out: ShardMapping = {}
        budget = _Budget(
            deadline=deadline,
            max_seconds=self.max_seconds,
            num_rays=self.num_rays,
        )

        self._run_phase(self._generic_batches(A, c, rng), A, c, out, budget)
        if self.face_aligned:
            self._run_phase(self._face_aligned_batches(A, c, rng), A, c, out, budget)

        if budget.cap_hit:
            Logger(
                f"Heuristic hit its ray ceiling of {self.num_rays}; set "
                "HEURISTIC_MAX_SECONDS to time-budget the shoot instead",
                level=Logger.Levels.info,
            ).log()

        if self.refine_witnesses:
            self._refine_witnesses(A, c, out, deadline=deadline)
        return out

    # ------------------------------------------------------------------
    # Phase runner -- Good-Turing stop, shared budget
    # ------------------------------------------------------------------

    def _run_phase(
        self,
        batches: Iterable[np.ndarray],
        A: np.ndarray,
        c: np.ndarray,
        out: ShardMapping,
        budget: _Budget,
    ) -> None:
        """
        Consume witness batches from ``batches`` until this phase's missing
        mass plateaus or the shared ``budget`` (time / ray cap / deadline)
        is hit.  Mutates ``out`` in place (global, min-L1 witness per cell).

        The Good-Turing state (``counts``, ``stats``, ``low_streak``) is
        local to this call -- that is what makes stopping *per-strategy*:
        each phase judges saturation on its own sampling distribution.
        """
        counts: Dict[SignTuple, int] = {}
        stats = _GTStats()
        low_streak = 0
        for V in batches:
            if budget.exhausted():
                break
            budget.shot += int(V.shape[0])
            self._collect_unique_cells_into(V, A, c, out, counts, stats)
            if stats.n <= 0:
                continue
            # Good-Turing missing mass: the fraction of samples that landed
            # in a cell seen exactly once estimates P(next sample is new).
            # While a sizeable reservoir of undiscovered cells remains, f1
            # stays large, so this refuses to stop -- structurally avoiding
            # a false "plateau then spike".  We require a *sustained* low
            # streak because the tail is lumpy.
            if self.missing_mass > 0.0:
                m_hat = stats.f1 / stats.n
                if m_hat < self.missing_mass:
                    low_streak += 1
                    if low_streak >= self.plateau_patience:
                        break
                else:
                    low_streak = 0

    # ------------------------------------------------------------------
    # Generation phases -- generators yielding (b, D) witness batches
    # ------------------------------------------------------------------

    def _generic_batches(
        self, A: np.ndarray, c: np.ndarray, rng: np.random.Generator
    ):
        """Infinite generator of generic origin-shot witness batches.

        Lazily yields one ``batch_size``-worth of witnesses per iteration;
        :meth:`_run_phase` stops pulling once the phase saturates or the
        budget is hit, so the generator is simply abandoned (no infinite
        loop runs).
        """
        d = A.shape[1]
        while True:
            V = self._sample_rays(rng, d, self.batch_size)
            if V.shape[0] == 0:
                continue
            yield self._shoot(V, A, c)

    def _face_aligned_batches(
        self, A: np.ndarray, c: np.ndarray, rng: np.random.Generator
    ):
        """
        Generator of face-aligned witness batches.

        For each of ``face_subsets`` random hyperplane subsets ``S`` (size
        in ``[1, D-1]``), shoot from ``face_offsets`` random integer
        offsets ``p`` along each integer nullspace direction ``v`` of
        ``A[S]``.  Hyperplanes in ``S`` are parallel to ``v`` (sign fixed
        by ``p``); the rest fix their sign by ``sign(A_i . v)`` -- so
        sweeping ``p`` enumerates the slab cells sharing recession
        direction ``v``.

        Buffers witnesses and yields a ``(b, D)`` array each time it fills
        a batch (plus a final partial batch).  Terminates naturally when
        the subsets are exhausted.
        """
        N, D = A.shape
        buf: List[np.ndarray] = []
        for _ in range(self.face_subsets):
            k = int(rng.integers(1, D)) if D > 1 else 1
            S = rng.choice(N, size=min(k, N), replace=False)
            for v in integer_nullspace(A[S]):
                if not np.any(v):
                    continue
                for _ in range(self.face_offsets):
                    p = rng.integers(
                        -self.max_coord, self.max_coord + 1, size=D, dtype=np.int64
                    )
                    w = _shoot_from(p, v, A, c)
                    if w is not None:
                        buf.append(w)
                        if len(buf) >= self.batch_size:
                            yield np.array(buf, dtype=np.int64)
                            buf = []
        if buf:
            yield np.array(buf, dtype=np.int64)

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
        counts: Optional[Dict[SignTuple, int]] = None,
        stats: Optional[_GTStats] = None,
    ) -> None:
        """
        Add the witnesses' ``{sign_tuple: point}`` entries into ``out``
        (in place).  When several rays land in the **same** cell, keep the
        witness **nearest the origin** (smallest L1 norm) -- different rays
        escape at very different ``t_final``, so the first to arrive is
        rarely the closest, and a smaller witness means a shorter downstream
        trajectory walk.  Points sitting exactly on any hyperplane (sign 0)
        are dropped -- not inside an open cell.

        When ``counts``/``stats`` are supplied, also maintain the phase's
        Good-Turing accumulator in O(1) per landed sample: a cell's first
        landing creates a singleton (``f1 += 1``); its second removes it
        (``f1 -= 1``); ``n`` counts every landed sample.
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
            if counts is not None and stats is not None:
                old = counts.get(key, 0)
                if old == 0:
                    stats.f1 += 1      # 0 -> 1 : new singleton
                elif old == 1:
                    stats.f1 -= 1      # 1 -> 2 : no longer a singleton
                counts[key] = old + 1
                stats.n += 1
