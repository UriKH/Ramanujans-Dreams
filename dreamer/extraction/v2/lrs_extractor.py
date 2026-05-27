"""
Exact shard extractor backed by the lrslib ``lrs`` binary.

Pipeline per call:

1. Enumerate every non-empty cell of the arrangement (sign vectors)
   using :func:`dreamer.extraction.v2.cells.enumerate_cells`.
2. For each cell, ask ``lrs`` for the V-representation and discard
   bounded cells (cells whose V-rep has no rays).
3. For each surviving unbounded cell, ask
   :func:`dreamer.extraction.v2.milp.find_integer_point` for one
   integer-coordinate witness, dropping cells with no integer point.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional

import numpy as np

from dreamer.extraction.hyperplanes import Hyperplane
from .base import BaseExtractor, ShardMapping
from .cells import enumerate_cells
from .lrs_io import format_hrep, lrs_available, parse_vrep_unbounded, run_lrs
from .milp import find_integer_point


class LrslibExtractor(BaseExtractor):
    """
    Exact strategy: enumerate cells, classify by unboundedness via
    ``lrs``, witness each unbounded cell with an MILP integer point.

    :param binary: Name or path of the ``lrs`` executable.
    :param per_call_timeout: Timeout (seconds) for each ``lrs`` call.
    :param max_cells: Safety cap on the cell enumeration.
    :param milp_bound: Symmetric box ``|x_d| <= milp_bound`` passed to
        the MILP solver -- larger lets it reach integer points in cells
        whose interior is far from the origin.
    :param seed: RNG seed used when picking the starting cell.
    :raises FileNotFoundError: If ``lrs`` is not on ``PATH``.
    """

    name = "exact"

    def __init__(
        self,
        *,
        binary: str = "lrs",
        per_call_timeout: float = 60.0,
        max_cells: int = 100_000,
        milp_bound: int = 10**6,
        seed: Optional[int] = 0,
    ):
        if not lrs_available(binary):
            raise FileNotFoundError(
                f"lrs binary not found on PATH (looked for {binary!r}). "
                "Install lrslib (e.g. `apt-get install lrslib` on WSL/Ubuntu)."
            )
        self.binary = binary
        self.per_call_timeout = per_call_timeout
        self.max_cells = max_cells
        self.milp_bound = milp_bound
        self.seed = seed

    def extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        if not hyperplanes:
            return {}

        A, c = self.hyperplanes_to_matrix(hyperplanes)
        cells = enumerate_cells(A, c, max_cells=self.max_cells, seed=self.seed)

        out: ShardMapping = {}
        for sig in cells:
            sig_arr = np.asarray(sig, dtype=np.int64)
            if not self._is_unbounded(A, c, sig_arr):
                continue
            point = find_integer_point(A, c, sig_arr, bound=self.milp_bound)
            if point is None:
                continue
            out[sig] = point
        return out

    def _is_unbounded(
        self, A: np.ndarray, c: np.ndarray, sign_vector: np.ndarray
    ) -> bool:
        """
        Ask ``lrs`` whether the cell is unbounded.

        :raises RuntimeError: If ``lrs`` times out or exits non-zero.
        """
        hrep = format_hrep(A, c, sign_vector, name="cell")
        try:
            vrep = run_lrs(hrep, binary=self.binary, timeout=self.per_call_timeout)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"lrs timed out after {self.per_call_timeout}s on a cell "
                f"({A.shape[0]} hyperplanes in dim {A.shape[1]})"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"lrs failed (exit {exc.returncode}): {exc.stderr!r}"
            ) from exc
        return parse_vrep_unbounded(vrep)
