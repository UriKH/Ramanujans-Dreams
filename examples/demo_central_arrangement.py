"""
Demo: enumerate shards via the CENTRAL arrangement (escape directions).

The idea
--------
A shard is an *unbounded* cell.  Every unbounded cell has a recession cone
-- a set of directions you can travel forever without leaving it.  Those
recession cones are exactly the cells of the **central arrangement**: the
same hyperplanes pushed through the origin,

    central cell:  { d : s_i (A_i . d) > 0  for all i }      (note: NO c_i)

There is one central cone per unbounded affine cell (a bijection).  The
central arrangement lives in the same dimension but has *no bounded cells*
and far fewer cells overall -- so enumerating it, then shooting one ray per
cone to land an integer witness, is the principled alternative to firing
millions of random rays (whose hit-rate per cone collapses as D grows).

Crucially, enumerating the central arrangement needs **no new code**: it is
``cells.enumerate_cells(A, c=0)`` -- the constants are what distinguish the
affine cells from the central cones, and dropping them homogenises the
arrangement.

This script demonstrates, on a small CMF where the full affine arrangement
can be enumerated exhaustively, that:

  (#central cones)  ==  (#unbounded affine cells found by the exact method)

and that shooting one ray per cone reaches exactly the exact method's
unbounded-cell set.  Run::  python examples/demo_central_arrangement.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.base import BaseExtractor  # noqa: E402
from dreamer.extraction.v2 import cells  # noqa: E402
from dreamer.extraction.v2.milp import find_integer_point  # noqa: E402
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


def build_arrangement():
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    formatter = pFq(constant, 2, 1, -1)  # small: full affine enumeration is feasible
    cmf_data = formatter.to_cmf()
    extractor = ShardExtractor(constant, cmf_data)
    hps = extractor._extract_cmf_hps()
    shifted = [hp.apply_shift(cmf_data.shift) for hp in hps]
    A, c = BaseExtractor.hyperplanes_to_matrix(shifted)
    return A, c, cmf_data.cmf_name


def main() -> int:
    A, c, name = build_arrangement()
    n, d = A.shape
    print(f"CMF: {name}   N={n} hyperplanes, D={d}\n")

    # --- Ground truth: the exact method's unbounded affine cells. ---
    all_affine = cells.enumerate_cells(A, c, seed=0)
    is_unbounded = cells.make_unbounded_checker(A)
    exact_unbounded = {s for s in all_affine if is_unbounded(np.asarray(s, dtype=np.int64))}
    print(f"exact method: {len(all_affine)} affine cells total, "
          f"{len(exact_unbounded)} unbounded (shards)")

    # --- Central arrangement: same hyperplanes through the origin (c = 0). ---
    zero_c = np.zeros(n, dtype=np.int64)
    central = cells.enumerate_cells(A, zero_c, seed=0)
    print(f"central arrangement: {len(central)} cones "
          f"(one expected per unbounded affine cell)")

    # --- One ray per cone -> integer witness -> affine cell. ---
    shooter = RayShootingExtractor(seed=0)
    reached = {}
    no_direction = 0
    for s_central in central:
        # A representative integer direction strictly inside the cone.
        direction = find_integer_point(A, zero_c, np.asarray(s_central, dtype=np.int64))
        if direction is None:
            no_direction += 1
            continue
        pts = shooter._shoot(np.array([direction], dtype=np.int64), A, c)
        if pts.shape[0] == 0:
            continue
        w = pts[0]
        affine_sign = tuple(np.sign(w @ A.T + c).astype(int).tolist())
        if 0 in affine_sign:
            continue
        reached.setdefault(affine_sign, w)

    print(f"\nshooting one ray per cone reached {len(reached)} distinct "
          f"unbounded affine cells")
    if no_direction:
        print(f"  ({no_direction} cones had no integer direction at margin 1 "
              "-- thin cones)")

    # --- Compare to ground truth. ---
    missed = exact_unbounded - set(reached)
    extra = set(reached) - exact_unbounded
    print("\ncomparison vs exact:")
    print(f"  reached & exact agree on : {len(set(reached) & exact_unbounded)} cells")
    print(f"  unbounded cells MISSED   : {len(missed)}")
    print(f"  cells reached but NOT unbounded per exact (should be 0): {len(extra)}")
    if not missed and not extra:
        print("\n  => central-arrangement enumeration reproduces the exact "
              "method's\n     unbounded-cell set EXACTLY, with one ray per cone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
