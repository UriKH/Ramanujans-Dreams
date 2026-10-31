r"""
Canonical-teleportation symmetry strategies for shard extraction.

A CMF family may have a symmetry group acting on the lattice that maps
shards onto shards (symmetric shards are "the same" for our purposes).
Rather than fold the symmetry into the search geometry (injecting
ordering hyperplanes broke the Avis-Fukuda parent tree and silently
dropped in-domain cells), we explore the *unconstrained* arrangement and
**teleport** discovered interior points into a fundamental domain, then
deduplicate by the resulting canonical sign signatures.

Extensibility
=============
The transform is specific to the CMF family, so it lives behind the
:class:`SymmetryStrategy` interface (one vectorised
``apply(points) -> canonical_points`` method) rather than being hardcoded
into the extractors.  :func:`symmetry_for_cmf` is the factory the
extraction layer calls; adding a new family means adding a branch there
(or a new strategy class), with no change to ``cells.py`` /
``ray_extractor.py``.

The :math:`pFq` family
======================
Its symmetry group is :math:`S_p \times S_q`: the first ``p`` (numerator)
coordinates may be permuted freely among themselves, and the last ``q``
(denominator) coordinates likewise.  Because the v2 extractors work in
the *shifted* lattice (absolute coordinate ``x = z + shift``), the
canonical form sorts the **absolute** coordinates and maps the result
back to the shifted lattice.

Crucially, two coordinates may be swapped only if their shifts share the
same fractional part — otherwise the swap would not map the shifted
integer lattice onto itself (this matches the legacy
``initial_points.__same_shift_indices`` grouping; the earlier v2 attempt
that sorted whole blocks unconditionally was incorrect for non-uniform
shifts).  Within a fractional-shift group the swap is a valid lattice
symmetry even when the integer parts differ, which is why we sort the
absolute ``x`` and subtract the per-coordinate shift afterwards (for the
common all-zero shift this reduces to a plain block sort).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple
from ramanujantools.cmf import CMF, MeijerG, pFq as rt_pFq
import numpy as np
import sympy as sp

from dreamer.utils.logger import Logger

Range = List[Tuple[int, int | None]]


class SymmetryStrategy(ABC):
    """Maps points to a canonical representative of their symmetry orbit."""

    @abstractmethod
    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Return the canonical form of each row of ``points``.

        :param points: ``(N, D)`` array (one point per row).  May be
            integer or float.
        :return: ``(N, D)`` array of canonical points.  Two points in the
            same orbit map to the *same* canonical point, so feeding the
            canonical points through ``sign(P @ A.T + c)`` yields an
            orbit-invariant signature for deduplication.
        """
        raise NotImplementedError

    def canonical_point(self, point: np.ndarray) -> np.ndarray:
        """Canonical form of a single ``(D,)`` point (row-wise convenience)."""
        return self.apply(np.asarray(point)[None, :])[0]


class BlockSortSymmetry(SymmetryStrategy):
    r"""
    Canonicalise by sorting coordinate groups of the **absolute** point.

    :param groups: Each group is a sequence of column indices whose
        coordinates may be freely permuted (i.e. an :math:`S_k` factor).
        Groups must be disjoint; singletons and the empty list are no-ops.
    :param shift: Per-coordinate shift ``s`` such that the absolute
        coordinate is ``x = z + s`` (``z`` being the value in ``points``).
        Sorting is applied to ``x`` and undone afterwards, so coordinates
        with different integer shifts inside a group still canonicalise
        correctly.  ``None`` (default) means no shift (sort ``z`` directly).
    """

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        shift: Optional[Sequence[float]] = None,
    ):
        # Keep only groups that actually permit a non-trivial permutation.
        self.groups: List[np.ndarray] = [
            np.asarray(g, dtype=np.intp)
            for g in groups
            if len(np.atleast_1d(np.asarray(g))) > 1
        ]
        self.shift = None if shift is None else np.asarray(shift, dtype=np.float64)

    def apply(self, points: np.ndarray) -> np.ndarray:
        P = np.array(points, dtype=np.float64, copy=True)
        if self.shift is not None:
            P += self.shift  # work in absolute coordinates
        for g in self.groups:
            # Sort each group's columns ascending; any orbit member yields
            # the same sorted tuple, so the signature is orbit-invariant.
            P[:, g] = np.sort(P[:, g], axis=1)
        if self.shift is not None:
            P -= self.shift  # back to the shifted lattice
        # Preserve integerness when the inputs (and the net transform) are
        # integral: for a fractional-shift group the subtraction restores
        # integers exactly, but float round-off could leave 1e-16 dust.
        if np.issubdtype(np.asarray(points).dtype, np.integer):
            P = np.rint(P)
        return P


class Grouping(ABC):
    """Builds the column-index groups for a given CMF family and shift."""

    def __init__(self, cmf: CMF, shift: Sequence[float]):
        self.cmf = cmf
        self.shift = shift
    
    @abstractmethod
    def groups(self, shift: Sequence[float]) -> Range:
        """
        Return the column-index groups for the given per-coordinate shift.

        :param shift: Per-coordinate shift in the same column order the
            extraction matrices use (numerator block first, then denominator).
        :return: List of integer arrays, each array being a group of column
            indices whose coordinates may be freely permuted.
        """
        raise NotImplementedError

    def reduce_shift(self, shift: Sequence[float]) -> np.ndarray:
        """
        Reduce the per-coordinate shift to its fractional part.

        :param shift: Per-coordinate shift in the same column order the
            extraction matrices use (numerator block first, then denominator).
        :return: Array of fractional shifts.
        """
        return np.array([v - int(v) for v in shift], dtype=np.float64)

    def group(self) -> List[np.ndarray]:
        """
        Return the column-index groups for the given per-coordinate shift.

        Each block ``(start, stop)`` from :meth:`groups` is split by equal
        fractional shift.  The returned indices are **absolute** column indices
        (offset by the block's ``start``), so they can be used directly on the
        full point matrix.  ``stop=None`` means "to the last column".
        """
        reduced = self.reduce_shift(self.shift)
        groups: List[np.ndarray] = []

        for start, stop in self.groups(self.shift):
            stop = len(reduced) if stop is None else stop
            block = reduced[start:stop]
            for off in np.unique(block):
                groups.append(np.where(block == off)[0] + start)
        return groups


class pFqGrouping(Grouping):
    r"""Builds the column-index groups for the :math:`pFq` :math:`S_p \times S_q` symmetry."""

    def __init__(self, cmf: rt_pFq, shift: Sequence[float]):
        super().__init__(cmf, shift)
        self.p = cmf.p
        self.q = cmf.q

    def groups(self, shift: Sequence[float]) -> Range:
        if self.p + self.q != len(shift):
            raise ValueError(
                f"p + q must equal len(shift): {len(shift)} != {self.p} + {self.q}"
            )
        return [(0, self.p), (self.p, None)]


class MeijerGGrouping(Grouping):
    r"""Builds the column-index groups for the :math:`MeijerG` :math:`S_m \times S_{q-m} \times S_n \times S_{p-n}` symmetry."""

    def __init__(self, cmf: MeijerG, shift: Sequence[float]):
        super().__init__(cmf, shift)
        self.p = cmf.p
        self.q = cmf.q
        self.m = cmf.m
        self.n = cmf.n

    def groups(self, shift: Sequence[float]) -> Range:
        if self.p + self.q != len(shift):
            raise ValueError(
                f"p + q must equal len(shift): {len(shift)} != {self.p} + {self.q}"
            )
        if self.n > self.p or self.m > self.q:
            raise ValueError(
                f"n must be <= p and m must be <= q: n={self.n}, p={self.p}, m={self.m}, q={self.q}"
            )
        # ramanujantools orders the MeijerG axes a0..a_{p-1}, b0..b_{q-1}
        # (``cmf.matrices.keys()``), so the blocks are a[:n], a[n:], b[:m], b[m:].
        return [(0, self.n), (self.n, self.p), (self.p, self.p + self.m), (self.p + self.m, None)]


class GroupingFactory:
    """Factory for building the column-index groups for a given CMF family and shift."""

    @staticmethod
    def create(cmf: CMF, shift: Sequence[float]) -> Grouping | None:
        match cmf:
            case rt_pFq():
                return pFqGrouping(cmf, shift)
            case MeijerG():
                return MeijerGGrouping(cmf, shift)
            case _:
                return None


def symmetry_for_cmf(cmf: CMF, shift: Sequence[float]) -> Optional[SymmetryStrategy]:
    r"""
    Build the :class:`SymmetryStrategy` for ``cmf`` (or ``None`` if the CMF
    family has no registered symmetry).

    This is the single place the extraction layer consults, keeping the
    family-specific logic out of the extractors.  Currently handles the
    :math:`pFq` family (:math:`S_p \times S_q`); extend here for others.

    :param cmf: A ramanujantools CMF instance.
    :param shift: Per-coordinate shift in the same column order the
        extraction matrices use (numerator block first, then denominator).
    """
    grouping = GroupingFactory.create(cmf, shift)
    if grouping is None:
        return None
    groups = verify_symmetry_groups(cmf, grouping.group(), shift)
    return BlockSortSymmetry(groups, shift=list(shift))


def _evaluate(cmf: CMF, axis: sp.Symbol, point: dict) -> sp.Matrix:
    """``M_axis`` evaluated at an exact rational ``point``."""
    return sp.Matrix(cmf.matrices[axis]).xreplace(point)


def _is_zero_matrix(matrix: sp.Matrix) -> bool:
    """Exact zero test; only non-numeric leftovers are simplified."""
    return all(
        (entry == 0) if entry.is_Number else sp.simplify(entry) == 0
        for entry in matrix
    )


def _is_exact_swap_symmetry(
    cmf: CMF,
    i: int,
    j: int,
    points: Sequence[dict],
    at_points: Sequence[dict],
) -> bool:
    r"""
    Whether swapping axes ``i`` and ``j`` is an exact symmetry of ``cmf``.

    The swap :math:`\sigma` is a symmetry iff
    :math:`M_{\sigma(k)}(\sigma x) = M_k(x)` for every axis :math:`k`.  Both
    sides are rational functions, so the identity is tested exactly at the
    given generic rational ``points`` (a non-identity vanishes at a random
    rational point with negligible probability).

    :param points: Axis-to-value substitutions (exact rationals).
    :param at_points: Per point, ``{axis: M_axis(point)}`` (the unswapped side,
        shared by every swap tested at that point).
    :return: ``True`` if the identity holds at every point.
    """
    axes = list(cmf.matrices.keys())
    target = {axes[i]: axes[j], axes[j]: axes[i]}
    for point, evaluated in zip(points, at_points):
        swapped = {**point, axes[i]: point[axes[j]], axes[j]: point[axes[i]]}
        for k in axes:
            if not _is_zero_matrix(evaluated[k] - _evaluate(cmf, target.get(k, k), swapped)):
                return False
    return True


def _generic_points(axes: Sequence[sp.Symbol], count: int, seed: int) -> List[dict]:
    """``count`` generic rational substitutions for ``axes`` (no poles hit in practice)."""
    rng = np.random.default_rng(seed)
    return [
        {ax: sp.Rational(int(rng.integers(-997, 997)), int(rng.integers(101, 997))) for ax in axes}
        for _ in range(count)
    ]


def verify_symmetry_groups(
    cmf: CMF,
    groups: Sequence[Sequence[int]],
    shift: Sequence[float],
    *,
    num_points: int = 2,
    seed: int = 0,
) -> List[np.ndarray]:
    r"""
    Keep only the column groups that are verified exact symmetries of ``cmf``.

    This is a cross-check of a family's *declared* groups (from its
    :class:`Grouping`), not the definition of its symmetry.  A group of
    columns :math:`g_0 < g_1 < \dots` is kept iff every coordinate in it has
    the same fractional shift and every adjacent swap :math:`(g_t, g_{t+1})`
    is an exact symmetry (adjacent transpositions generate :math:`S_k`, so
    the whole group is then a symmetry).  A failing group is dropped with a
    warning: deduplication can only become weaker, so no non-symmetric shard
    is ever merged away.

    :param cmf: The ramanujantools CMF the groups were declared for.
    :param groups: Absolute column-index groups (as from :meth:`Grouping.group`).
    :param shift: Per-coordinate shift (same column order as the groups).
    :param num_points: Generic rational points to test each swap at.
    :param seed: Seed for the test points (deterministic by default).
    :return: The verified groups (groups of size < 2 are dropped as no-ops).
    """
    axes = list(cmf.matrices.keys())
    frac = [sp.Rational(s) - sp.floor(sp.Rational(s)) for s in shift]
    points = _generic_points(axes, num_points, seed)
    at_points = [{k: _evaluate(cmf, k, point) for k in axes} for point in points]
    kept: List[np.ndarray] = []
    for group in groups:
        cols = sorted(int(c) for c in np.atleast_1d(np.asarray(group)))
        if len(cols) < 2:
            continue
        same_shift = len({frac[c] for c in cols}) == 1
        if same_shift and all(
            _is_exact_swap_symmetry(cmf, a, b, points, at_points) for a, b in zip(cols, cols[1:])
        ):
            kept.append(np.asarray(cols, dtype=np.intp))
            continue
        Logger(
            f"Symmetry group {[str(axes[c]) for c in cols]} of {type(cmf).__name__} is not an "
            "exact symmetry of the CMF; dropping it (less deduplication, no shards lost).",
            Logger.Levels.warning,
        ).log()
    return kept