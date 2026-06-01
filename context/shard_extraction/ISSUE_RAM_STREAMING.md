# Issue: Shard Extraction RAM Bloat + Crash Safety

## Problem

`RayShootingExtractor` holds two large in-memory dicts throughout an entire run:

- `out: Dict[SignTuple, np.ndarray]` — sign-tuple → witness array for every found shard
- `counts: Dict[SignTuple, int]` — sign-tuple → hit count for Good-Turing stopping

For a multi-hour 11D (6F5) scan finding ~1M+ shards, combined RAM is 5–7 GB.
A forced shutdown at any point loses **all** discovered shards — `shards.jsonl` is
only written after `extract()` returns.

## Suggested solutions

**A. Stream witnesses to disk, keep a compact `seen` set (recommended)**
Flush `out` to `shards.jsonl` every N shards (e.g. 10k). After flushing, clear
`out` and move keys into a compact `seen: Set[bytes]` (sign vectors encoded as
packed bytes, ~50 bytes/key vs ~3 KB for a Python tuple). Peak witness RAM becomes
`N × ~3 KB = ~30 MB`. `counts` stays as-is (current phase only, freed when phase
ends). Uses existing `write_shard_records` infrastructure.

**B. Hash-approximate `counts`**
Replace `counts: Dict[SignTuple, int]` with a fixed-size numpy counter array
indexed by a 64-bit hash. Cuts the `counts` dict to near-zero RAM, accepts a
~10⁻⁷ per-pair collision probability. More complex; not needed unless `counts`
itself becomes the bottleneck.

**C. Drop per-cell `counts`, use batch-level f₁ estimate**
Estimate Good-Turing f₁/n from batch statistics only (no per-cell tracking).
Zero overhead for `counts`, but less accurate stopping criterion.

## Chosen approach (when implemented)

**Option A.** Implement streaming writes + `seen` set. Adds a `checkpoint_interval`
config knob (default 10_000 shards). Keep `counts` exact for Good-Turing accuracy —
it is already per-phase (freed after each phase ends), so its peak size is bounded
by one phase's total hits, not the entire run.

Key files:
- `dreamer/extraction/v2/ray_extractor.py` — `_run_phase`: add checkpoint flush loop
- `dreamer/configs/extraction.py` — add `HEURISTIC_CHECKPOINT_INTERVAL: int = 10_000`
- `dreamer/extraction/v2/manager.py` / `dreamer/extraction/extractor.py` — forward knob
