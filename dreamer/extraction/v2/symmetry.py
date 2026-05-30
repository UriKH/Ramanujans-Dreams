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
from typing import List, Optional, Sequence

import numpy as np


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


def _fractional_shift_groups(
    p: int, q: int, shift: Sequence[float]
) -> List[np.ndarray]:
    r"""
    Build the column-index groups for the :math:`pFq` :math:`S_p \times S_q`
    symmetry, partitioned by equal fractional shift.

    The first ``p`` columns are the numerator block, the last ``q`` the
    denominator block; within each block, only coordinates whose shifts
    share a fractional part may be swapped.  Mirrors the legacy
    ``initial_points.__same_shift_indices``.
    """
    if p + q != len(shift):
        raise ValueError(
            f"p + q must equal len(shift): {len(shift)} != {p} + {q}"
        )
    reduced = np.array([v - int(v) for v in shift], dtype=np.float64)
    groups: List[np.ndarray] = []
    # Numerator block: indices [0, p)
    nom = reduced[:p]
    for off in np.unique(nom):
        groups.append(np.where(nom == off)[0])
    # Denominator block: indices [p, p + q); offset back into full coords.
    denom = reduced[p:]
    for off in np.unique(denom):
        groups.append(p + np.where(denom == off)[0])
    return groups


def symmetry_for_cmf(cmf, shift: Sequence[float]) -> Optional[SymmetryStrategy]:
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
    # Imported lazily so this module does not hard-depend on ramanujantools
    # at import time (and to avoid a heavy import for the no-symmetry path).
    try:
        from ramanujantools.cmf import pFq as rt_pFq
    except Exception:  # pragma: no cover - ramanujantools always present in prod
        return None

    if isinstance(cmf, rt_pFq):
        groups = _fractional_shift_groups(cmf.p, cmf.q, list(shift))
        return BlockSortSymmetry(groups, shift=list(shift))
    return None
