# Ramanujan's Dreams — Development Roadmap

> **Doc type:** Operational state. Tracks active goals, recent completions,
> and open questions — the only doc in `context/` that's *expected* to
> change every few days. Update at decision points (goal completed,
> priority shifted, design question resolved) — not for every
> implementation detail.

---

## Project Overview

Ramanujan's Dreams is a Python package for searching and exploring Continued Matrix Fields (CMF) spaces and the recurrence formulas they produce for universal constants.

---

## Context structure
In `context` directory you can find files with information about the different aspects of Ramanujan's Dreams project and details about needed or already impelemented features:
- `context/background`:
  - [`CODE_QUALITY.md`](context/background/CODE_QUALITY.md) contains information about the expected code standards you should follow.
  - [`DESIGN.md`](context/background/DESIGN.md), [`PIPELINE.md`](context/background/PIPELINE.md) contain information about the overall design and structure of this system.
  - [`MATH_OBJECTS.md`](context/background/MATH_OBJECTS.md) contains information about math objects used throught this project.
  - [`MISSING_TESTS.md`](context/background/MISSING_TESTS.md) contains information about tests which should be implemented.
- [`DEFINITION_OF_DONE.md`](context\DEFINITION_OF_DONE.md) contains the definition of done task.

### IMPORTANT:
- This is list partial, if you miss information or data ask the user and explore this directory.
- If you find any indiscrepancies between files in this folder let the user know and ask how to act.

---

## Overall design (current)

**Note: make sure to update this section as the design evolves**

The system is desigend as a modular pipeline:

1. Loading: creating a CMF to search constants in.
2. Extraction: extracting the hyperplanes which divide the space of each CMF into smaller cells called shards. In this stage we locate the shard objects.
3. Analysis: each shard is checked for the constant we want to search for in it. If it doesn't contain the constant, it's discarded. If it does, we will use the shard in the next steps.
4. Search: running different search algorithms inside each shard gathering data about the different trajectories.

---

## Reminders (Will turn into tasks in the backlog or active list)

- Add coverage policy.
- Add definition of done.
- Add coding standards file.
- Add git interactions instructions.
- Discuss pickle files usage. Were do we want pickle files and should we use them only temporarily?
- Remind to update the Next Up task list.

---

## Backlog (Will later be moved to Active list)

- [ ] Make sure the samplers are deterministic using a seed (RNG).
- [ ] Graceful handling of forced shutdowns — ensure Tier-1 data already computed is not lost on early exit (mini-chunk writes or checkpoint flushing).

---

## Near-Term Tasks (Active / Next Up)

- [x] 2026-05-31 — Shard symmetry utilization via **canonical teleportation** (S_p × S_q for pFq).  One representative per orbit is extracted; integrated into both the exact and heuristic methods.  Spec: [`context/shard_symmetries/SYMMETRIES.md`](shard_symmetries/SYMMETRIES.md).
  * **Replaced the earlier (reverted) fundamental-domain approach.**  The first attempt injected ordering hyperplanes (`x_i >= x_{i+1}`) as LP constraints / domain-rejected the reverse-search base.  That was **wrong**: intersecting the arrangement with the convex domain cone disconnects the Avis–Fukuda parent tree and silently drops in-domain shards (measured: pFq(3,1) returned a strict subset).  All of that plumbing (`sym_p`/`sym_q` everywhere) was removed.
  * **Method (correct, verified):** explore the *unconstrained* arrangement; teleport each discovered interior point into a fundamental domain (block-sort coordinates per S_p / S_q) and fingerprint by `sign(A @ canonical_point + c)`.  This is an exact orbit invariant — needs **no symmetry-group enumeration**, so it scales (heuristic is O(D log D)/point, fully vectorised; works to 6F5).
  * **The one critical subtlety (a bug found and fixed mid-implementation):** the exact method must fingerprint a **strictly-interior (slack-maximising) point**, never an LP *vertex*.  CBC's feasibility solution is a vertex lying on hyperplanes / on the `x_i=x_j` walls; teleporting it mis-counts orbits (pFq(3,1): 44 vs true 50; pFq(2,2): 102 vs 100).  `cells._make_interior_point_finder` now returns the scipy slack-max point.  Verified the canonical BFS hits the **exact** true orbit counts: pFq(2,1)=28, (3,1)=50, (2,2)=100; pFq(4,3)=7756 orbits (vs >100k cells) in ~400s.
  * **Code:** new `v2/symmetry.py` (`SymmetryStrategy`, `BlockSortSymmetry`, `symmetry_for_cmf`); `cells.iter_cells_canonical`/`enumerate_cells_canonical` (canonical-teleport BFS) selected when a symmetry is active, Avis–Fukuda otherwise; `ray_extractor` teleports witnesses (generic integer points → exact, ratio 1.000 through 4F3); `lrs_extractor`/`manager`/`extractor._discover_via_v2` thread an injected `symmetry` object built by `symmetry_for_cmf`.  Fractional-shift grouping (only equal-fractional-shift coords swap) matches legacy `__same_shift_indices`.
  * **Tests:** `TestBlockSortSymmetry`, `TestSymmetryReduction` (orbit-count vs ground-truth group action, heuristic canonical witnesses, manager/extractor forwarding), `TestSymmetryFactory`.  Full suite **430 passed**.


- [x] 2026-06-01 — Summary scoped to current run only (hardened).
  * `system.py`: silently-swallowed `derive_cmf_and_shard_ids` failures now emit a `Logger.warning`; the `or None` fallback that exposed all orphan JSONLs on failure was removed — an empty dict is always passed so the summary stays scoped to this run's shards.
  * `summary.py` `_collect_shard_stats`: after the JSONL pass, empty `_ShardStats` rows are backfilled for any shard in `this_run_shards` without a JSONL file (extracted but not yet searched / filtered by analysis), so the table always lists every extracted shard even with zero trajectories.
  * `build_summary_markdown`: distinguishes "extraction produced no shards" from "no JSONL files on disk" in the fallback message.

- [x] 2026-06-01 — Separate timeouts for exact and heuristic extraction methods.
  * `TIMEOUT_SECONDS` removed; replaced by `EXACT_TIMEOUT_SECONDS` (default 3600 s) and `HEURISTIC_TIMEOUT_SECONDS` (default 3600 s) in `extraction_config`.
  * Under `"auto"`: exact extractor gets `EXACT_TIMEOUT_SECONDS`; on timeout the heuristic independently gets `HEURISTIC_TIMEOUT_SECONDS`.
  * Under `"exact"` / `"heuristic"` alone, only the matching knob applies.
  * `extractor.py`, `main_example.py`, `benchmark_extraction.py`, `ray_extractor.py` comment, and `test_extractor.py` updated; 430 tests pass.

- [ ] **Multi-constant shards** — sample trajectories once per CMF and evaluate all target constants on the same trajectory.
  * **Motivation:** searching N constants in the same CMF currently runs extraction and trajectory walks N times independently.  Sharing the walk (constant-independent) and only branching at delta/p-q/identified (constant-dependent) eliminates redundant computation.
  * **API change (loading stage):** CMF formatters (`pFq`, etc.) accept a single constant or a list of constants instead of binding to exactly one.  The pipeline groups CMFs by structure (family + params + shift) and creates one multi-constant `Shard` per unique CMF.  Constants not listed in `System.run()` are silently dropped at load time.
  * **Shard / Searchable:** hold a list of `Constant` objects instead of one.  The trajectory matrix walk runs once per (trajectory, shard) pair.  Per-constant attributes — `delta`, `p vector`, `q vector`, `identified` — are stored as `{const_name: value}` dicts.
  * **JSONL format:** one file per shard (no longer one per constant per shard).  `delta` and identification fields become nested dicts keyed by constant name.  `search_data.py` must be updated to browse and display this structure.
  * **Analysis stage:** a shard passes if it identifies *any* of its constants above the threshold.
  * **`search_data.py`:** update `list-shards`, `list-trajectories`, `show-trajectory` subcommands to reflect multi-constant records.
  * **Design decision (2026-06-01):** confirmed by user — single JSONL per shard, delta stored as `{const_name: value}`.

- [ ] **Shard streaming + RAM cap during extraction.** See full issue spec in [`context/shard_extraction/ISSUE_RAM_STREAMING.md`](shard_extraction/ISSUE_RAM_STREAMING.md).  Short version: `out` (witnesses) + `counts` (Good-Turing) are held entirely in memory — 5–7 GB for a 1M-shard run, forced shutdown loses everything.  Chosen fix: stream witnesses to `shards.jsonl` every 10k shards + compact `seen: Set[bytes]` for dedup; `counts` stays exact (it is already per-phase).  **RAM policy (general):** any stage accumulating large in-memory dicts should checkpoint periodically — ask "what happens on forced shutdown after 2 hours?" before finalising any long-running stage.



- [ ] Implement shard extraction (upgrade). See full instructions in `SHARD_EXTRACTION_PLAN.md`.
  * Done in two steps on 2026-05-27: (1) v1 strategy-pattern package landed in `dreamer/extraction/v2/`; (2) `ExtractionManager` is now the default path via `extraction_config.STRATEGY="auto"` with timeout-protected fallback to heuristic.  Legacy lattice scan is retained behind `STRATEGY="legacy"`.  See Completed for both entries plus the benchmark numbers.
  * **Still open under this task:** pFq-symmetry deduplication in v2 (v2 currently finds the symmetric duplicates that `IGNORE_DUPLICATE_SEARCHABLES=True` was filtering in legacy — see the backlog item "shard symmetry utilization"), and a real-CMF correctness sweep beyond the small benchmark.

<!-- Tier design decisions (resolved 2026-05-20):
     - Tier-1: delta + identified + core DTO fields (limit, recurrence, p/q vectors, order).
       Always computed in main thread during both Analysis and Search.
       Analysis stage computes ONLY Tier-1 — no extra attributes.
     - Tier-2: configurable extras (eigenvalues, spectral_gap, gcd_slope, convergence_class).
       Computed by background worker processes during Search. NOT redundant with Tier-1:
       these require additional computation beyond the trajectory walk and would block the
       main thread if computed inline.
     - Tier-3: expensive extras (asymptotics, kamidelta). Post-process stage only (deferred).
     - Global queues across shards (vs per-shard): deferred — right direction but adds
       shutdown complexity. Current per-shard queues are correct. -->

<!-- IGNORE THIS FOR NOW: - [ ] The whole part of the background computation is that it can happen even the main processes running the search moved to the next shard. This means that we don't need each time to create  dedicated background_attribute_worker-s and dedicated_file_writer. We can just use the same queues for all shards. To maintain correct load blancing, if we see that the background processes can't keep up, we can provide more background processes before  -->

---

## Mid-Term Goals

- [ ] Add new attributes computed for each trajectory and recurrence relation which will be added to the CMF Atlas. Initial implementaion in `dreamer.utils.storage.trajectory_attributes`. This should be very well connected later to the DB (CMF Atlas), starting direction here is the usage of the `dreamer.utils.storage.dtos`.


---

## Long-Term Vision

- [ ] The CMF Atlas - Create a database structure mapping constants to their respective CMFs and vice versa. Mapping each trajectory to the shard it belongs to, and the shard to the CMF it belongs to.
- [ ] Compute many important trajectory and recurrence relation's attributes which will later be added to the CMF Atlas.
- [ ] Upgraded shard extraction methods:
  - [ ] Shard finding - current methods do not find all shards.
  - [ ] A point in each shard - current methods just enumerate points and that is how we get each shard and a point in it. Given a Shard we want to find an integer coordinates point in it efficiently.
  - [ ] Shard symmetry utilization - shards could be considered symmetric under certain conditions, we should use it to avoid redundant computations. For CMFs in the pFq family we know something about their symmetric nature. See `SYMMETRIES.md` (if empty remind me to create it).
- [ ] Add advanced search algorithms. See `SEARCH_ALGORITHMS.md` (if empty remind me to create it).
- [ ] Support search of multiple constants in the same CMF simultaneously — now tracked as an active task in Near-Term Tasks.
- [ ] Discuss and develop how to deal with forced shutdowns and gracefull exist - ensuring data will not be completely lost.

---

## Completed

- [x] 2026-05-30 — Heuristic stopping criterion → Good-Turing missing-mass + face-aligned shooting (P2).
  * **Stopping criterion (the real fix).** The running-total marginal-gain stop (`new / total_found < rel_improvement`) was statistically wrong: it fires merely because the running total is large, even while yield is still linear (the user found 5e-3→O(10k), 1e-3→140k, 1e-4→800k on 6F5).  A naive `new / batch_size` is a single-batch estimate of the missing mass — highest variance.  Replaced with the **Good-Turing** estimator `m̂ = f1 / n` (`f1` = cells seen exactly once, `n` = samples landed), maintained O(1) per sample in `_collect_unique_cells_into`.  It estimates P(next sample is a new cell), so a still-large reservoir keeps `f1` high and **refuses to stop** — structurally immune to the "plateau then spike" failure the user worried about.  Config knob renamed `HEURISTIC_REL_IMPROVEMENT` → `HEURISTIC_MISSING_MASS` (default 5e-4).
  * **Per-strategy stopping.** Extracted the batch loop into a reusable `_run_phase(batches, ..., budget)` with a *fresh* Good-Turing tracker per phase; `_Budget` (time / ray cap / deadline) is shared across phases.  So a plateau in generic shooting no longer halts a later phase — required for P2.
  * **num_rays optional.** Default `None` (unlimited); `max_seconds` + the missing-mass plateau are the practical limiters.  Logs once if a finite ray ceiling is actually hit.
  * **P2 face-aligned shooting.** New `integer_nullspace` (exact sympy) + `_shoot_from` + `_face_aligned_batches` generator: for a random hyperplane subset S, shoot along `null(A[S])` from random integer offsets — reaches unbounded cells with *lower-dimensional* recession cones (tubes/slabs) that origin rays hit with probability 0.  Off by default (`HEURISTIC_FACE_ALIGNED`, `_FACE_SUBSETS=200`, `_FACE_OFFSETS=50`), threaded through manager + extractor.  **Measured on pFq(4,3) D=7: +283 new integer-containing cells beyond generic (10328 → 10611), all verified unbounded by the recession-cone checker, all witnesses inside their cells.**
  * Tests: Good-Turing singletons drive m̂; sustained-plateau stop; per-strategy independence; baseline misses tube / face-aligned finds it (witness inside); `integer_nullspace` primitive; num_rays=None unlimited; renamed/extended forwarding tests.  Full suite **415 passed**.  Commit `0199981`.

- [x] 2026-05-29 — Exact near-origin base seeding (E1).  `cells._find_start_cell` now samples the reverse-search base from a **tight near-origin box first**, widening geometrically only if the tight box yields no off-hyperplane point.  Reverse search reaches every cell regardless of base — only the *order* changes — but near the origin cells are "fat" and integer-rich, so under a deadline this front-loads shard-yielding cells.  **Measured ~1.6× more shards** in the first 2000 enumerated cells on pFq(4,3) D=7 (mean 187 vs 112 across seeds 0–4), robust across seeds (the earlier single radius-8 collapse was one unlucky base, not the trend).  Experiment script: `examples/measure_seed_radius.py`.  Tests: prefers the near-origin middle cell when hyperplanes are far out; widens when the tight box is entirely on hyperplanes.

- [x] 2026-05-29 — Heuristic better start points.  (H1) `RayShootingExtractor._collect_unique_cells_into` now keeps the **nearest-to-origin (min-L1)** witness when several rays land in the same cell, instead of first-wins (different rays escape at very different `t_final`, so first-arrival is rarely closest).  (H2) New opt-in `refine_witnesses` flag (config `HEURISTIC_REFINE_WITNESSES`, default False) runs the Stage-D MILP once per discovered cell to return the **provably L1-minimal** integer start point (~2 ms/cell); threaded through `ExtractionManager(heuristic_refine=...)` and `ShardExtractor._discover_via_v2`, so it also polishes the heuristic top-up under `auto`.  Default path stays solver-free.  Tests: collision keeps nearest (both orders), refine returns the MILP minimum and never worsens a witness, manager forwards the flag.

- [x] 2026-05-29 — Independent cross-check of the unbounded classifier + root-cause of exact's low yield.  Built `examples/verify_unbounded_checkers.py`: an independent **dual (Stiemke)** unbounded checker (`scipy`, `A^T w = 0`, `s_i w_i >= 1` feasible ⇔ bounded) cross-checked against the production **primal** recession-cone checker on real CMF arrangements.  **0 disagreements** on pFq(2,1) D=3 (39 cells, exhaustive) and pFq(4,3) D=7 (sampled) ⇒ classification is correct.  Corrected two earlier misconceptions: cells are **~82–92% unbounded** (NOT "~96% bounded"), and exact's low shard yield is a **margin mismatch** — enumeration admits cells at slack `eps=1e-6`, but a usable shard needs an integer point (margin `>= 1`); ~93% of enumerated unbounded cells are real-but-integer-empty "thin" cells, correctly dropped by the MILP (verified: 0 integer-gaps, all drops fail the continuous margin-1 relaxation).  Full write-up in the plan doc (`~/.claude/plans/graceful-drifting-wilkes.md`).

- [x] 2026-05-27 — Per-cell unbounded check via in-process recession-cone LP (replaces the lrs subprocess as the default).  A full-dim cell is unbounded iff its recession cone `{d : s_i A_i d >= 0}` is nontrivial; `cells.make_unbounded_checker` decides this with one hot-started LP (`max sum_i s_i (A_i.d)` over the cone ∩ unit box: 0 ⇒ bounded, >0 ⇒ unbounded), handling the lineality case once via `rank(A) < D ⇒ all unbounded`.  `LrslibExtractor` gains `unbounded_check` (`"lp"` default / `"lrs"` opt-in cross-check); `"lp"` needs no lrs binary at all.  Config knob `EXACT_UNBOUNDED_CHECK` threaded through the manager.  Measured 0.48 vs 1.85 ms/cell (4x) with 100% agreement vs lrs on the real D=7 pFq(4,3) arrangement.  Tests: LP↔lrs end-to-end agreement, scipy fallback, lineality, triangle bounded-cell.
  * **Original task 61 (validation queue) deliberately dropped:** the queue only pays off while validation is expensive; with the LP check at ~0.5 ms validation is no longer the bottleneck (enumeration ~20 ms/cell is), so a producer/consumer split would parallelise the cheap part.  Parallel *enumeration* is the better lever — done next.

- [x] 2026-05-27 — Salvage-aware parallel exact extraction.  `cells.reverse_search_seeds` + `cells.iter_subtree` expose the base cell and its disjoint root subtrees; `LrslibExtractor._extract_parallel` dispatches each subtree to a worker (`_subtree_extract_worker`, top-level/picklable) that **enumerates + classifies (LP) + locates (MILP)** its own subtree and returns `(shards, timed_out)`.  The main process merges them with the base cell; if any worker tripped the deadline it re-raises `ExtractionTimeout(partial=merged)` so the manager still unions with the heuristic — i.e. parallelism **without** losing Task-3 salvage.  Config `EXACT_NUM_WORKERS` (default 1) threaded through the manager; `>1` forces the LP check (LP==lrs, just faster).  Verified `num_workers={1,2,4,8}` give the identical shard set; ~2.4x at 8 workers on a D=5 case (sub-linear: uneven subtree sizes + pool overhead on small problems).  Tests: parallel==serial shard set/points, parallel salvages partial on timeout.

- [x] 2026-05-27 — `max_cells` is now an *optional* ceiling (default `None`), so the timeout is the real stop.  Safe because reverse search is memoryless (`O(depth)` stack, no `seen` set) — a `None`/large cap does not bloat RAM during enumeration; only the kept output (unbounded shards) grows, which is legitimate data.  Threaded `Optional[int]` through `_reverse_search_iter`/`iter_cells`/`iter_subtree`/`enumerate_cells`/`_enumerate_parallel` and `LrslibExtractor.max_cells` (all guard `if max_cells is not None`).  *Caveat for future:* a truly enormous completed run still holds all shards in memory — if that ever bites, stream shards to `shards.jsonl` as found (Task-2 cache) and/or add an RSS watchdog; not needed yet.

- [x] 2026-05-27 — Shard-object creation sped up ~319x by **hoisting `apply_shift`** out of the per-shard loop (NOT by parallelising).  `Shard.__init__`/`from_cmf_data` gained `hyperplanes_already_shifted`; `ShardExtractor.extract` now shifts the hyperplanes once per CMF and reuses them across all shards (the shift is identical for every shard).  Measured 24.64 → 0.08 ms/shard on D=7 pFq(4,3) (N=33), i.e. ~50s → ~0.15s for 2000 shards.  Parallelising `Shard` creation was rejected: per-shard sympy was the real cost (now gone), and pickling Shard objects (numpy arrays + CMF ref) back from workers would likely cost more than it saves.

- [x] 2026-05-27 — Heuristic adaptive ray budgeting.  `RayShootingExtractor` shoots rays in batches (`batch_size`, default 20k) up to `num_rays` (cap raised to 1M) and stops early once a batch adds fewer than `plateau_ratio * batch_size` new cells (default 1e-3) — easy arrangements finish fast, hard ones get the full budget.  Honours the `deadline` between batches.  `plateau_ratio=0` disables early stopping; each batch is still the same vectorised NumPy pass (loop is over batches, not rays).  Tests: plateau stops early on a trivial arrangement, disabled runs full budget, deadline halts the loop.

- [x] 2026-05-27 — Per-CMF `shards.jsonl` load-to-skip cache.  The write side already existed (`atlas_writer.write_shard_records` → `<EXPORT_CMFS>/<const>/<cmf>__shards.jsonl`, idempotent ShardDTO append).  Added `atlas_writer.read_shard_records` + shared `shard_records_path`, a new `extraction_config.LOAD_SHARD_CACHE` flag (default False), and `ShardExtractor._load_cached_encodings`: when the flag is on and a cache exists, hyperplanes are recomputed (canonical order ⇒ `encoding[i]` still labels `hps[i]`) and shards are rebuilt from the cached encoding + interior point, **skipping enumeration entirely**.  Stale caches (encoding length ≠ current hyperplane count) are detected and ignored so a changed CMF can't mis-align signs.  New shards still append idempotently.  Tests: cache-hit skips discovery, flag-off ignores cache, stale-cache falls back to extraction.

- [x] 2026-05-27 — Auto fallback now *salvages* exact's partial work instead of discarding it.  `LrslibExtractor.extract` was restructured to **interleave** enumeration with classification (lrs) + point-finding (MILP) via the new streaming `cells.iter_cells` generator, so at any instant it holds fully-formed unbounded shards.  `ExtractionTimeout` now carries a `partial` payload; on deadline the extractor re-raises with everything completed so far, and `ExtractionManager._auto_extract` unions that with the heuristic's cells — exact's MILP points (near-origin) win on overlap (`{**heuristic, **partial}`).  Tests: interleaved partial salvage carries classified shards, manager union prefers exact's point.  Note: on the huge D=7 case exact still contributes only a small fraction within the budget (enumeration-bound); the salvage mainly helps the medium regime where exact nearly finishes.

- [x] 2026-05-27 — Fast algebraic ray-shooting heuristic (`shard_extraction/FAST_RAY_SHOOTING_SPEC.md`).  `ray_extractor.py` rewritten to a fully vectorised, solver-free NumPy formulation (`t_escape = max -c_i / (V @ A^T)_i`, `t_final = floor(t_escape)+1`, witness `t_final * v`), with GCD primitive-ray reduction and default `num_rays=100_000`.  ~100× faster than the old scalar loop; ~480/485 cells covered on D=7 in ~0.8s, witness coords p50=5/p95=15.

- [x] 2026-05-27 — Exact-method optimisation (`shard_extraction/EXACT_EXTRACTION_OPTIMIZATION_SPEC.md`).
  * **Phase 1:** `cells.py` uses a hot-started stateful python-mip (CBC) feasibility solver — LP built once, sign flips become `y_i` bound swaps, basis reused across checks.  scipy `linprog` kept as automatic fallback when `mip` is unavailable (`mip` added to `pyproject.toml`).  ~5× per-check speedup (0.43 vs 2.14 ms/check, D=7/N=20), 100% agreement vs scipy.  Fixed: default `epsilon` 1e-9 → 1e-6 (must stay above CBC's ~1e-7 feasibility tolerance or empty cells read as feasible).
  * **Phase 2 (Avis–Fukuda reverse search):** `enumerate_cells` rewritten from BFS+`seen` to memoryless reverse search — each cell has a unique parent (min-index separating-feasible flip vs a generic base cell), enumerated by walking the spanning tree forwards.  No `seen` set → `O(depth)` memory and disjoint subtrees dispatch cleanly across processes via `num_workers>1` (top-level `_subtree_worker`, each builds its own solver).  Verified against a 2^N brute-force sweep and serial==parallel.
  * **Fallback-stall fix (the actual D=7 bug):** the old `ExtractionManager` ran exact in a `ThreadPoolExecutor`; on timeout the `with` block's `shutdown(wait=True)` blocked until the 20-min exact run finished, so the heuristic never started.  Replaced with a **cooperative deadline**: manager runs exact synchronously with `deadline=time.time()+timeout`; `enumerate_cells`/lrs loop check the clock each iteration and raise `ExtractionTimeout`; manager catches it and falls back instantly.  `deadline` threaded through `BaseExtractor.extract` (ray ignores it).  Verified end-to-end: D=7 pFq(4,3) `auto` now returns 3789 heuristic shards in bounded time instead of stalling.
  * **Key finding (unchanged):** reverse search lowers memory and parallelises, but cannot reduce the cell *count*.  pFq(4,3) D=7 has >100k cells, so exact still can't complete there within a sane budget — the deadline→heuristic fallback is what makes the run usable.  Exact is for moderate N·D; heuristic remains the default for large pFq.

- [x] 2026-05-27 — Shard extraction v2 wired into the main pipeline as the default:
  * `extraction_config.STRATEGY` (default ``"auto"``) selects between v2 (``"auto" | "exact" | "heuristic"``) and the preserved brute-force path (``"legacy"``).  Companion knob `STRATEGY_TIMEOUT_SECONDS` (default 3600s) is the wall-clock cap on the exact strategy under ``"auto"``.
  * [extractor.py](../dreamer/extraction/extractor.py) — `ShardExtractor.extract` now dispatches via two private helpers `_discover_via_legacy` (unchanged code) and `_discover_via_v2` (calls `ExtractionManager` with shifted hyperplanes, translates integer witnesses back to absolute coords).  `selected_points` augmentation still applies to either path.  When `IGNORE_DUPLICATE_SEARCHABLES=True` is combined with a v2 strategy on a pFq CMF, the user is warned that v2 doesn't yet dedup symmetries (pointer to the symmetry-utilisation backlog item).
  * [main_example.py](../examples/main_example.py) — exposes `STRATEGY` and `STRATEGY_TIMEOUT_SECONDS` so the example doc shows the new knobs explicitly.
  * [examples/benchmark_extraction.py](../examples/benchmark_extraction.py) — short isolated harness that builds the same pFq(log(2), 2, 1, -1) CMF used in main_example, runs each of the four strategies, prints runtime + shard counts + Jaccard overlap of sign-encoding sets.  First run: legacy = 12 shards / 9.2s vs v2 = 18 shards / ~0.5s; the 12 legacy encodings are a strict subset of the 18 v2 encodings (legacy was deduping pFq symmetries, v2 sees all of them).
  * Tests: 4 new cases in [tests/test_extractor.py](../tests/test_extractor.py) cover default strategy, unknown strategy, end-to-end heuristic, manager-spy on the ``"auto"`` path, and the pFq dedup warning.  All 39 extraction tests pass; full suite green.

- [x] 2026-05-27 — Shard extraction v1 (strategy pattern, side-by-side with the old `initial_points` pipeline):
  * New package [dreamer/extraction/v2/](../dreamer/extraction/v2/) with `BaseExtractor` ABC + three strategies + `ExtractionManager`:
    * [base.py](../dreamer/extraction/v2/base.py) — abstract `BaseExtractor.extract(hps) -> {sign-encoding: int point}`; shared `hyperplanes_to_matrix` packs canonical hps into integer ``(A, c)``.
    * [cells.py](../dreamer/extraction/v2/cells.py) — non-empty cell enumeration via flip-graph BFS using `scipy.optimize.linprog`; bounded by `max_cells` to fail loudly on pathological arrangements.
    * [milp.py](../dreamer/extraction/v2/milp.py) — integer-point witness via `scipy.optimize.milp`; strict ``> 0`` is tightened to ``>= 1`` (sound because all coeffs and ``x`` are integer).
    * [lrs_io.py](../dreamer/extraction/v2/lrs_io.py) — subprocess wrapper around the `lrs` CLI (no C-API deps), tempfile H-rep writer, V-rep parser detecting any ``0 ...`` ray line.
    * [lrs_extractor.py](../dreamer/extraction/v2/lrs_extractor.py) — exact strategy: enumerate cells → classify unbounded via `lrs` → MILP integer point.
    * [ray_extractor.py](../dreamer/extraction/v2/ray_extractor.py) — heuristic strategy: random integer rays from origin, doubling scale until the sign vector stabilises.
    * [manager.py](../dreamer/extraction/v2/manager.py) — `"auto" | "exact" | "heuristic"` router; ``auto`` runs exact in a `ThreadPoolExecutor` with a wall-clock cap and falls back to heuristic on timeout, exception, or missing `lrs` binary.
  * Tests at [tests/test_extraction_v2.py](../tests/test_extraction_v2.py) — 31 cases covering matrix packing, MILP feasibility, lrs format/parse, cell enumeration (axes/strip/triangle), ray shooter, `LrslibExtractor` mocked + end-to-end against the real binary, and all `ExtractionManager` branches. All pass under WSL `rama`.
  * **Not** wired into `ShardExtractor` — the old `initial_points` pipeline remains the production path. Migration is a separate task.

- [x] 2026-05-26 — Worker-stage constant propagation + structural ids + browse CLI:
  * **Tier-2 / Tier-3 workers now receive the sympy constant** (3-tuple item `(traj_matrix, constant, payload)`).  Without it, `delta_sequence` and other limit-dependent attributes failed inside the worker with `'NoneType' object has no attribute 'evalf'`.  Producer in [hedgehog_scan_mod.py](../dreamer/search/searchers/hedgehog_scan_mod.py) and [tier3_post_process_mod.py](../dreamer/post_process/tier3_post_process_mod.py) pushes `constant_sympy`; workers in [multi_processing.py](../dreamer/utils/multi_processing.py) and [tier3_post_process_mod.py](../dreamer/post_process/tier3_post_process_mod.py) build `TrajectoryAttributesHandler(traj_matrix, constant=constant)`.  Standalone post-process also falls back to `sp.sympify(record["constant"])` when the shard lookup is empty.
  * **shard_id / trajectory_id are now structural** — `shard_id = "{cmf_id}__{16-char hash}"`, `trajectory_id = "{shard_id}__{16-char hash}"`.  Any id is now self-describing (the parent CMF and shard can be recovered with `rsplit("__", 1)` without a separate lookup).  New helper `derive_trajectory_id(...)` centralises the prefix construction.  JSONL filenames are now `<shard_id>.jsonl` (no longer prefixed with the cmf name again).
  * **Browse CLI** at [examples/search_data.py](../examples/search_data.py) — three subcommands (`list-shards <cmf_id>`, `list-trajectories <shard_id>`, `show-trajectory <trajectory_id>`); reads merged per-shard JSONL output.
  * Tests updated for the new id format and 3-tuple worker item shape; 320 / 321 pytest cases pass (the remaining failure — `TestConfigAttributeSelection::test_search_config_default_tier2_is_empty` — is unrelated and pre-existing, due to an uncommitted change to the `TIER2_ATTRIBUTES` default).
  * **Migration:** existing per-shard JSONL files under `EXPORT_SEARCH_RESULTS` are orphaned by the new id format and were deleted before the rerun.

- [x] 2026-05-19 — Created `background/DESIGN.md` — architecture, pipeline stages, module layout, configuration, serialization, decision log
- [x] 2026-05-19 — Created `background/MATH_OBJECTS.md` — mathematical reference for CMF, Integer Lattice, Hyperplane, Shard, Trajectory, δ, PCF, Recurrence Relation
- [x] 2026-05-20 — Implemented Attributes Management System (Task 3): wired `TrajectoryAttributesHandler` as the canonical computation unit across analysis and search stages; introduced MPMC pipeline for async Tier-3 attributes; added DTO serialization (`to_json_line` / `from_dict`) to all 4 DTOs; fixed `ShardDTO` field-ordering bug; added `sample_pairs()` to `SerialSearcher`; rewrote `AnalyzerModV1` and `SearcherModV1`; wired `multi_processing.py` workers with config propagation; 48 tests all passing.
- [x] 2026-05-20 — JSONL import/export + configurable per-stage attribute selection: added `Formats.JSONL` handling in `Exporter`/`Importer`; rewrote `System.__search_stage()` print loop to scan JSONL outputs for the best delta record; removed dead `System.run_search_pipeline()`; introduced central `ATTRIBUTE_REGISTRY` in `dreamer/utils/storage/attribute_registry.py` with `compute_attribute` / `compute_attributes` / `register_attribute`; added `TIER2_EXTRA_ATTRIBUTES`/`TIER3_ATTRIBUTES` to search config and `ANALYSIS_EXTRA_ATTRIBUTES` to analysis config; refactored worker, producer, and analyzer to drive computation from these lists; 71 tests passing.
- [x] 2026-05-20 — Re-run dedup + tier config cleanup: (a) producer now derives `trajectory_id` before constructing the handler, so fully-covered trajectories are skipped without any walk; (b) `_run_shard` skips spawning workers/writer when `TIER2_ATTRIBUTES` is empty (direct file write from main thread); (c) analyzer keys `<constant>.jsonl` by `shard_id` and reuses cached `best_delta`/`identified_pct` on re-runs — no resampling for shards already on file; (d) renamed `TIER3_ATTRIBUTES` → `TIER2_ATTRIBUTES`, default = `()`, removed legacy `TIER2_EXTRA_ATTRIBUTES`; (e) `load_seen_shards()` helper added in `multi_processing.py`; new tests cover the early-skip, direct-write spy, analyzer cache, and Tier-2 patch path.
- [x] 2026-05-21 — Parallelism abstraction + Tier-3 post-process stage: (a) introduced `worker_pool` context manager in `multi_processing.py` that hides all `mp.Queue`/`mp.Process`/sentinel boilerplate behind a single `push(item)` callable; `worker_fn=None` collapses to direct-write on the main thread without changing the producer.  (b) Refactored Search's `_run_shard` to one short body using `worker_pool` — killed the `_run_shard_mpmc`/`_run_shard_direct` split.  (c) Added `PostProcessConfig` with `TIER3_ATTRIBUTES` (default `()`) and wired it through `ConfigManager`.  (d) Added `PostProcessModScheme` and `Tier3PostProcessModV1` (in new `dreamer/post_process/`): scans search-output JSONLs, reconstructs handlers from in-memory CMF lookup (fallback: `EXPORT_CMFS` on disk), computes missing Tier-3 attrs via `worker_pool`, and appends patch records.  (e) `System(post_processor=...)` parameter + auto call from `run()`.  (f) Tests: `TestWorkerPool` covers direct + MPMC + producer-raises cleanup; `TestTier3PostProcess` covers short-circuit / skip / patch / cmf lookup.  (g) New `context/PIPELINE.md` documents the end-to-end flow with file pointers.
- [x] 2026-05-22 — Determinized hyperplane order + constant-free cmf_id + batched writer flush (3 active tasks completed together; share an id-derivation surface).
  * **Task 1 (canonical hyperplanes):** `_extract_cmf_hps` now returns a list sorted by `str(hp.expr)` (canonical form is already produced by `Hyperplane.__post_init__`, so the sort key is stable across runs).  `Shard.encoding[i]` therefore unambiguously labels `cmf.hyperplanes[i]`.  `_serialize_inequalities` (which lex-sorted `[A|b]` rows) is replaced by `_serialize_encoding`, which simply comma-joins the ±1 vector.  `derive_cmf_and_shard_ids` now hashes `(cmf_id, encoding_str)` — no inequality blob involved.
  * **Task 2 (constant-free cmf_id):** the `[self.const]` segment is removed from `Formatter.__init__`'s name composition.  A CMF can host multiple constants over time without spawning new ids; the `EXPORT_CMFS/<const>/...` directory still scopes records by constant.
  * **Task 3 (batched writer flush):** `sys_config.WRITER_BATCH_SIZE = 100` and `sys_config.WRITER_FLUSH_TIMEOUT_SECONDS = 5.0` added.  `_run_writer_loop` flushes every batch_size records or after timeout seconds of queue inactivity (whichever first); a final flush always runs on shutdown.  `worker_pool` direct mode flushes every batch_size pushes; the `with open(...)` exit guarantees the tail flush.
  * **Tests:** `TestSerializeInequalities` → `TestSerializeEncoding` (3 tests rewritten); new `test_encoding_matches_sign_vector`, `test_cmf_name_does_not_contain_constant`, and `test_direct_mode_batches_flushes` (uses `WRITER_BATCH_SIZE=3` to verify per-N flush boundary).  127 tests passing.
  * **Migration:** existing JSONL files under `EXPORT_CMFS` and `EXPORT_SEARCH_RESULTS` are now orphaned (different ids); next run starts fresh, no migration script written.
- [x] 2026-05-22 — `Shard.encoding` (±1 sign vector) + flaky-test fix: `Shard.__init__` now persists its input `encoding` as `self.encoding` (a tuple of ±1, one entry per CMF hyperplane).  `build_shard_dto` reads `shard.encoding` so `ShardDTO.shard_encoding` carries the combinatorial sign label (`+1`=above, `−1`=below) instead of a flattened inequality blob.  `shard_id` still hashes the canonical `[A|b]` system, so it's unaffected.  Test fix: the 4 flaky tests in `TestMergeOnRead` and `TestAnalyzerDedup` now pin `SerialSearcher.sample_pairs` via a new `_freeze_sample_pairs` helper so the test's cache-population loop and the producer/analyzer's internal call see identical pair lists.  125 tests passing, stable across reruns.
- [x] 2026-05-22 — CMF / CMF-family / Shard DTO storage (Atlas-ready): new `dreamer/utils/storage/atlas_writer.py` exposes `build_cmf_family_dto`, `build_cmf_dto`, `build_shard_dto`, `append_dtos_jsonl` (idempotent, dedup-by-id-field), `write_cmf_records`, `write_shard_records`, and `update_cmf_hyperplanes`.  Loading stage in `system.py` writes `cmfs.jsonl` + `cmf_families.jsonl` under `EXPORT_CMFS/<const>/` alongside the existing pickle dump (with empty `cmf_hyperplanes`).  Extraction stage in `extractor.py` writes one `<cmf>__shards.jsonl` per CMF in the same directory **and** backfills the `cmf_hyperplanes` field of the matching CMF record (one canonical row per CMF — in-place rewrite, no patch/merge needed).  `ShardExtractor.hyperplanes` is now exposed as an attribute so the mod can read what was computed.  All writes skip ids already present so reruns don't grow the files.  Shard ids use `derive_cmf_and_shard_ids` (matches the trajectory JSONL filenames).  15 new `TestAtlasWriter` tests cover builders, idempotent append, round-trip via `ShardDTO.from_dict`, the high-level `write_*_records` helpers, and the `update_cmf_hyperplanes` backfill (matching-record / no-match / missing-file).
- [x] 2026-05-21 — Analyzer per-trajectory dedup (replaces the per-shard cache): analyzer now always samples and dedups at the `trajectory_id` level against the same per-shard JSONL the searcher uses (`EXPORT_SEARCH_RESULTS/<const>/<cmf>__<shard_id>.jsonl`).  Cached records with `delta_estimate` + `identified` skip the walk; new ones get a full `TrajectoryDTO` line.  Added `identified` as a Tier-1 field on `TrajectoryDTO` (defaults to `True` for backward-compat).  Old per-shard `<constant>.jsonl` summary file is dropped — aggregations are computed in-memory.  `TestAnalyzerDedup` rewritten: always-samples / skips-walks-for-cached / walks-uncached / writes-per-trajectory / partial-cache-only-walks-missing.  PIPELINE.md §5 + §6 + §11 updated to reflect the unified single-canonical store.

---

## Notes / Open Questions

<!-- Assumptions, blockers, design decisions pending review -->

- **Stopping is now Good-Turing missing-mass (2026-05-30).**  `m̂ = f1/n` estimates P(next sample is a new cell).  Robust to "plateau then spike": a large undiscovered reservoir keeps `f1` high so it won't stop early.  Each generation phase has its own tracker; time/ray budgets are global.  Face-aligned shooting samples the central arrangement's lower-dim faces *heuristically* (integer nullspace of a random hyperplane subset + random offsets), sidestepping the exact face enumeration that proved intractable below.

- **Central-arrangement enumeration — TESTED, naive version does NOT work (2026-05-29).**  The idea: unbounded cells correspond to cells of the *central* arrangement of the linear parts `{d : A_i . d = 0}`, a lower-dimensional problem with no bounded cells; enumerate those cones, take one direction each, shoot one ray for the integer witness.  `examples/demo_central_arrangement.py` tested it on pFq(2,1,-1) (D=3) and it recovered only **18 of 36** unbounded cells.  Root cause (diagnosed): these CMF arrangements are **non-simple** — they have parallel hyperplane families (5 distinct normals among 7 hyperplanes), so reusing `cells.enumerate_cells(A, c=0)` is **not a valid central enumerator**: the duplicate rows make the homogeneous feasibility LP over-count (64 sign-vectors returned, only ~18 genuine cones — exceeds the theoretical max of 22 for 5 planes in R^3, a CBC tolerance artifact on the forced-equal duplicate rows).  Even ignoring that, representative directions hit the same thin-cell problem (in direction space) and origin-rays don't reach every unbounded cell.  **Conclusion: this is NOT a quick win** (my earlier "highest-leverage lever" framing was over-optimistic).  A real implementation would need to dedup parallel hyperplanes, enumerate the projective arrangement correctly, and handle thin cones + reachability — substantial work, deferred.
