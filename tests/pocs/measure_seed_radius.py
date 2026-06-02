"""
Experiment: does seeding the exact reverse-search base near the origin
front-load "fat" integer-rich cells (more shards per enumerated cell)?

Reverse search reaches every cell regardless of the base; only the ORDER
changes.  Under a wall-clock deadline, order is everything -- we want the
shard-yielding cells first.  This enumerates the first N cells from bases
sampled at different radii and counts how many become shards (unbounded +
an integer point exists).

Run::  python examples/measure_seed_radius.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.base import BaseExtractor  # noqa: E402
from dreamer.extraction.v2 import cells as C  # noqa: E402
from dreamer.extraction.v2.milp import find_integer_point  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


def build_arrangement():
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    formatter = pFq(constant, 4, 3, 1)
    cmf_data = formatter.to_cmf()
    extractor = ShardExtractor(constant, cmf_data)
    hps = extractor._extract_cmf_hps()
    shifted = [hp.apply_shift(cmf_data.shift) for hp in hps]
    return BaseExtractor.hyperplanes_to_matrix(shifted)


def yield_in_first_n(A, c, radius, n_cells, seed=0):
    """Enumerate the first ``n_cells`` from a base sampled at ``radius``;
    return (cells_seen, shards_found, base_l1)."""
    rng = np.random.default_rng(seed)
    base = C._find_start_cell(A, c, rng=rng, radius=radius, max_attempts=2000)
    is_feasible = C._make_feasibility_checker(A, c, epsilon=1e-6)
    is_unbounded = C.make_unbounded_checker(A)
    n_hp = A.shape[0]
    seen = shards = 0
    deadline = time.time() + 600
    for sig in C._reverse_search_iter(
        base, base, is_feasible, n_hp, max_cells=n_cells, deadline=deadline
    ):
        seen += 1
        sv = np.asarray(sig, dtype=np.int64)
        if is_unbounded(sv) and find_integer_point(A, c, sv) is not None:
            shards += 1
        if seen >= n_cells:
            break
    return seen, shards


def main() -> int:
    A, c = build_arrangement()
    n, d = A.shape
    print(f"arrangement: N={n}, D={d}")
    n_cells = 2000
    print(f"shards found in first {n_cells} cells, across base seeds 0..4:\n")
    print(f"  {'radius':>8}  " + "  ".join(f"seed{s}" for s in range(5)) + "    mean")
    for radius in (3, 1000):
        row = []
        for seed in range(5):
            _, shards = yield_in_first_n(A, c, radius, n_cells, seed=seed)
            row.append(shards)
        mean = sum(row) / len(row)
        print(f"  {radius:>8}  " + "  ".join(f"{x:>5}" for x in row) + f"  {mean:>6.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
