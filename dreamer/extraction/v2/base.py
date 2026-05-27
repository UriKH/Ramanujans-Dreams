"""
Abstract base class and shared types for v2 shard extractors.

A v2 extractor takes a list of canonical :class:`Hyperplane` objects and
returns one integer-coordinate representative for every *unbounded* cell
(shard) of the resulting hyperplane arrangement.

Each shard is identified by a sign-encoding tuple of length ``len(hps)``
whose ``i``-th entry is ``+1`` if the representative satisfies
``a_i . x + c_i > 0`` for hyperplane ``i`` and ``-1`` if it satisfies
``a_i . x + c_i < 0``.  Points lying exactly on a hyperplane are never
returned (no zero entries).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane


SignEncoding = Tuple[int, ...]
"""Tuple of ``+1`` / ``-1`` of length ``N`` (= number of hyperplanes)."""

ShardMapping = Dict[SignEncoding, np.ndarray]
"""Mapping from sign-encoding to one integer point inside the cell."""


class BaseExtractor(ABC):
    """
    Abstract strategy for extracting unbounded shards from a hyperplane
    arrangement.

    Subclasses implement :meth:`extract`.  All strategies share the same
    input/output contract so :class:`ExtractionManager` can route between
    them transparently.
    """

    name: str = "base"

    @abstractmethod
    def extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        """
        Compute one integer representative per unbounded shard.

        :param hyperplanes: Canonical hyperplanes (each in ``a . x + c``
            form via :class:`Hyperplane.__post_init__`).  The order of
            this list defines the order of bits in every returned
            :data:`SignEncoding`.
        :return: A :data:`ShardMapping` -- never contains zero entries
            in any encoding and never returns the empty cell.
        :raises RuntimeError: When the underlying algorithm cannot
            complete (e.g. exact strategy times out).
        """
        raise NotImplementedError

    @staticmethod
    def hyperplanes_to_matrix(
        hyperplanes: List[Hyperplane],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack hyperplanes into integer ``(A, c)`` such that hyperplane
        ``i`` is ``A[i] . x + c[i] = 0``.

        :param hyperplanes: Canonical hyperplanes sharing a common
            ``symbols`` order.
        :return: ``(A, c)`` with ``A`` shape ``(N, D)`` and ``c`` shape
            ``(N,)``, both :class:`numpy.int64`.
        :raises ValueError: If the list is empty or hyperplanes disagree
            on the symbol set.
        """
        if not hyperplanes:
            raise ValueError("hyperplanes_to_matrix requires at least one hyperplane")
        symbols = hyperplanes[0].symbols
        for hp in hyperplanes[1:]:
            if hp.symbols != symbols:
                raise ValueError(
                    "All hyperplanes must share the same symbol order; "
                    f"got {hp.symbols} vs {symbols}"
                )
        rows = []
        consts = []
        for hp in hyperplanes:
            lin, free = hp.vectors
            rows.append(np.asarray(lin, dtype=np.int64))
            consts.append(int(free))
        return np.stack(rows, axis=0).astype(np.int64), np.asarray(consts, dtype=np.int64)
