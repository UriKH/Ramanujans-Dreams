"""
Step 1 prototype: face-aligned ray shooting.

Generic origin rays find only FULL-dimensional recession-cone cells.  To
reach the missed LOWER-dim recession ("tube/slab") cells we shoot along
directions that lie in the intersection of a subset S of hyperplanes
(v in the EXACT integer nullspace of A_S, so A_i.v = 0 for i in S), from
several random start offsets p.  Then:
  - for i not in S: sign = sign(A_i . v)            (fixed by direction)
  - for i in S    : sign = sign(A_i . p + c_i)      (varies with offset p)
so sweeping p enumerates the slab cells sharing recession direction v.

This script
  (1) validates the idea on a SYNTHETIC strip whose tube cell the baseline
      provably misses, then
  (2) measures, on a real CMF, how many NEW integer-containing unbounded
      cells face-aligned shooting finds beyond the baseline (no ground
      truth needed: a new cell with an integer witness whose recession
      cone is non-trivial is a genuine recovered shard).

Run::  python examples/prototype_face_aligned.py [p q]
"""
from __future__ import annotations

import os
import sys
from math import gcd
from functools import reduce

import numpy as np
import sympy as sp
from scipy.optimize import linprog

_HERE = os.path.abspath(os.path.dirname(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dreamer import config, zeta  # noqa: E402
from dreamer.extraction.extractor import ShardExtractor  # noqa: E402
from dreamer.extraction.v2.base import BaseExtractor  # noqa: E402
from dreamer.extraction.v2 import cells  # noqa: E402
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


# --------------------------------------------------------------------------
# Core: integer nullspace, offset shooting, face-aligned discovery
# --------------------------------------------------------------------------

def integer_nullspace(A_sub):
    """Exact integer basis of {v : A_sub @ v = 0} (rows scaled to integers).

    Uses sympy for an exact rational nullspace so A_i . v = 0 holds
    EXACTLY (float rounding would leave A_i.v small-but-nonzero and defeat
    the whole point)."""
    M = sp.Matrix(A_sub.tolist())
    basis = []
    for vec in M.nullspace():
        denoms = [sp.Rational(x).q for x in vec]
        lcm = reduce(lambda a, b: a * b // gcd(a, b), denoms, 1)
        ivec = np.array([int(x * lcm) for x in vec], dtype=np.int64)
        g = reduce(gcd, np.abs(ivec[ivec != 0]).tolist(), 0)
        if g > 1:
            ivec //= g
        basis.append(ivec)
    return basis


def shoot_from(p, v, A, c):
    """Integer witness past every crossing along the ray p + t*v (t>=0)."""
    Av = A @ v
    Apc = A @ p + c
    nz = Av != 0
    t_escape = float((-Apc[nz] / Av[nz]).max()) if nz.any() else 0.0
    t_final = max(int(np.floor(t_escape)) + 1, 1)
    w = p + t_final * v
    if np.any(A @ w + c == 0):
        return None  # landed on a hyperplane
    return w


def face_aligned_shards(A, c, *, n_subsets=4000, n_offsets=8,
                        offset_coord=4, seed=0):
    """Discover cells via subset-nullspace directions + random offsets."""
    rng = np.random.default_rng(seed)
    N, D = A.shape
    out = {}
    for _ in range(n_subsets):
        k = int(rng.integers(1, D))            # subset size 1..D-1
        S = rng.choice(N, size=k, replace=False)
        basis = integer_nullspace(A[S])
        if not basis:
            continue
        coeffs = rng.integers(-3, 4, size=len(basis))
        v = sum(int(co) * b for co, b in zip(coeffs, basis))
        if not np.any(v):
            continue
        for _ in range(n_offsets):
            p = rng.integers(-offset_coord, offset_coord + 1, size=D).astype(np.int64)
            w = shoot_from(p, v.astype(np.int64), A, c)
            if w is None:
                continue
            sign = tuple(np.sign(A @ w + c).astype(int).tolist())
            prev = out.get(sign)
            if prev is None or np.abs(w).sum() < np.abs(prev).sum():
                out[sign] = w
    return out


def recession_margin(A, sign):
    """max t s.t. s_i (A_i . d) >= t for all i, d in [-1,1]^D.

    t > 0  => full-dimensional recession cone (a generic ray CAN hit it, so a
              baseline miss is just finite-ray budget).
    t == 0 => lower-dimensional recession cone (tube/slab) -> generic rays
              structurally CANNOT reach it; only face-aligned shooting can."""
    A = np.asarray(A, dtype=np.float64)
    s = np.asarray(sign, dtype=np.float64)
    n, d = A.shape
    sA = s[:, None] * A
    obj = np.zeros(d + 1); obj[-1] = -1.0
    A_ub = np.hstack([-sA, np.ones((n, 1))]); b_ub = np.zeros(n)
    bounds = [(-1.0, 1.0)] * d + [(0.0, None)]
    res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    return float(res.x[-1]) if res.success else 0.0


def baseline_shards(A, c, seed=0):
    sh = RayShootingExtractor(seed=seed)
    rng = np.random.default_rng(seed)
    out, D = {}, A.shape[1]
    for _ in range(50):
        V = sh._sample_rays(rng, D, sh.batch_size)
        if V.shape[0] == 0:
            continue
        before = len(out)
        sh._collect_unique_cells_into(sh._shoot(V, A, c), A, c, out)
        if len(out) - before < sh.missing_mass * sh.batch_size:
            break
    return out


# --------------------------------------------------------------------------
# (1) Synthetic validation
# --------------------------------------------------------------------------

def synthetic_strip_test():
    print("=" * 60)
    print("(1) SYNTHETIC: strip {0<x<3} in R^2, unbounded along y")
    print("    (a 1-D recession cone -> generic origin rays must miss it)")
    A = np.array([[1, 0], [1, 0]], dtype=np.int64)   # x=0 and x=3
    c = np.array([0, -3], dtype=np.int64)
    strip = (1, -1)  # x>0 and x-3<0
    base = baseline_shards(A, c)
    face = face_aligned_shards(A, c, n_subsets=50, n_offsets=12, offset_coord=4)
    print(f"    baseline found the strip cell {strip}? {strip in base}")
    print(f"    face-aligned found the strip cell {strip}? {strip in face}")
    if strip in face:
        print(f"      witness = {face[strip].tolist()}  (expect x in (0,3))")
    print()


# --------------------------------------------------------------------------
# (2) Real CMF: new shards beyond baseline
# --------------------------------------------------------------------------

def real_cmf_test(p, q):
    print("=" * 60)
    print(f"(2) REAL CMF pFq({p},{q}) — new shards beyond baseline")
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    cmf_data = pFq(constant, p, q, 1).to_cmf()
    ex = ShardExtractor(constant, cmf_data)
    hps = ex._extract_cmf_hps()
    shifted = [h.apply_shift(cmf_data.shift) for h in hps]
    A, c = BaseExtractor.hyperplanes_to_matrix(shifted)
    print(f"    N={A.shape[0]}, D={A.shape[1]}")

    base = baseline_shards(A, c)
    face = face_aligned_shards(A, c, n_subsets=4000, n_offsets=8, offset_coord=4)
    new = set(face) - set(base)
    print(f"    baseline shards        : {len(base)}")
    print(f"    face-aligned shards    : {len(face)}")
    print(f"    NEW beyond baseline    : {len(new)}")
    if new:
        is_unb = cells.make_unbounded_checker(A)
        bad = sum(1 for s in new if not is_unb(np.asarray(s, dtype=np.int64)))
        print(f"    new cells NOT unbounded (must be 0): {bad}")
        # Classify the new cells: structural (low-dim recession, only
        # face-aligned can find) vs full-dim (baseline just under-sampled).
        low = full = 0
        for s in new:
            if recession_margin(A, np.asarray(s, dtype=np.int64)) > 1e-7:
                full += 1
            else:
                low += 1
        print(f"    of the {len(new)} new: LOW-dim recession (structural, "
              f"only face-aligned): {low}; FULL-dim (budget): {full}")
        print(f"    => recovered {len(new)} genuine shards the baseline missed")
    print()


def main() -> int:
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    q = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    synthetic_strip_test()
    real_cmf_test(p, q)
    return 0


if __name__ == "__main__":
    sys.exit(main())
