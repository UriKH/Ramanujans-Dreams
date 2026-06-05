"""
Compare heuristic witness quality with and without MILP refinement
(``HEURISTIC_REFINE_WITNESSES`` / ``RayShootingExtractor(refine_witnesses=)``)
on a real CMF arrangement.

Both runs use the same seed -> the same cells and the same raw ray
witnesses; refinement then replaces each witness with the L1-minimal
integer point of its cell.  We report, for each run: the shard count
(must match), the witness-size distribution (L1 norm and max coordinate),
and the runtime -- so the quality gain and its time cost are both visible.

Run::  python examples/compare_refine_witnesses.py
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
from dreamer.extraction.v2.ray_extractor import RayShootingExtractor  # noqa: E402
from dreamer.loading import pFq  # noqa: E402


def build_hyperplanes():
    config.configure(
        extraction={'INIT_POINT_MAX_COORD': 3, 'IGNORE_DUPLICATE_SEARCHABLES': False},
        logging={'GENERATE_LOGS': False},
    )
    constant = zeta(2)
    formatter = pFq(constant, 4, 3, 1)
    cmf_data = formatter.to_cmf()
    extractor = ShardExtractor(constant, cmf_data)
    hps = extractor._extract_cmf_hps()
    return [hp.apply_shift(cmf_data.shift) for hp in hps], cmf_data.cmf_name


def run(hps, *, refine=False, threshold=50.0, workers=1):
    ext = RayShootingExtractor(
        seed=0, refine_witnesses=refine,
        refine_l1_threshold=threshold, refine_workers=workers,
    )
    t0 = time.perf_counter()
    res = ext.extract(hps)
    dt = time.perf_counter() - t0
    l1 = np.array([int(np.abs(p).sum()) for p in res.values()])
    maxc = np.array([int(np.abs(p).max()) for p in res.values()])
    return res, dt, l1, maxc


def stats(name, dt, l1, maxc):
    print(f"  {name:<14} {dt:>7.2f}s  shards={len(l1)}")
    print(f"      L1 norm : mean={l1.mean():7.1f}  median={np.median(l1):6.0f}"
          f"  p95={np.percentile(l1,95):6.0f}  max={l1.max():6.0f}")
    print(f"      max|coord|: mean={maxc.mean():6.1f}  median={np.median(maxc):5.0f}"
          f"  p95={np.percentile(maxc,95):5.0f}  max={maxc.max():5.0f}")


def main() -> int:
    import os
    hps, name = build_hyperplanes()
    print(f"CMF: {name}  ({len(hps)} hyperplanes)\n")

    raw, dt0, l1_0, mc0 = run(hps, refine=False)
    allref, dt1, l1_1, mc1 = run(hps, refine=True, threshold=0)
    sel, dt2, l1_2, mc2 = run(hps, refine=True, threshold=50)
    workers = min(8, os.cpu_count() or 1)
    selp, dt3, l1_3, mc3 = run(hps, refine=True, threshold=50, workers=workers)

    above = int((l1_0 > 50).sum())
    print("raw ray witnesses (no refinement):")
    stats("raw", dt0, l1_0, mc0)
    print(f"\nrefine ALL (threshold=0): {len(allref)} MILPs")
    stats("refine-all", dt1, l1_1, mc1)
    print(f"\nrefine SELECTIVE (threshold=50): only {above} of {len(raw)} "
          f"witnesses exceed L1=50")
    stats("selective", dt2, l1_2, mc2)
    print(f"\nrefine SELECTIVE + parallel ({workers} workers):")
    stats("sel+par", dt3, l1_3, mc3)

    print("\nsanity:")
    print(f"  same cells across all runs : "
          f"{set(raw) == set(allref) == set(sel) == set(selp)}")
    print(f"  selective == parallel pts  : "
          f"{all(sel[k].tolist() == selp[k].tolist() for k in sel)}")
    print("\ntakeaway:")
    print(f"  refine-all cost     : +{dt1 - dt0:5.2f}s")
    print(f"  selective cost      : +{dt2 - dt0:5.2f}s  "
          f"({(dt1 - dt0) / max(dt2 - dt0, 1e-9):.0f}x cheaper than refine-all)")
    print(f"  selective+parallel  : +{dt3 - dt0:5.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
