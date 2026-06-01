# Exact-method throughput on high-D arrangements — options to weigh

> **Status:** decision pending (Uri to mull). Written 2026-05-27.
> **Context:** triggered by the 4F3 / 7D case, where the exact method
> enumerated ~14,000 candidate cells in a 5-minute budget but did not
> finish, and the question arose: *should we just raise the timeout?*

---

## 1. The situation, precisely

The exact extractor (`dreamer/extraction/v2/`) finds shards in two
conceptual steps, now **interleaved** per cell (Task 3):

1. **Enumerate** the cells of the hyperplane arrangement (reverse
   search over sign vectors). ~20 ms/cell — dominated by the
   stateful-solver feasibility LPs (~2·N per cell).
2. **Validate** each cell: is it *unbounded* (→ a shard)? via an `lrs`
   subprocess on its V-representation; then find an integer interior
   point via MILP. ~15–20 ms/cell — dominated by the **`lrs`
   subprocess spawn**.

### What the ~14,000 number means
At ~20 ms/cell, ~14k cells in 5 min matches **enumeration alone**.
Validation (~15–20 ms/cell more, serialized after each enumeration
step) means most of those 14k were *enumerated but not yet validated*.

### Three facts that make "just raise the timeout" a weak lever
1. **It won't finish.** 4F3/7D has **>100k total cells** (we hit the
   `max_cells=100_000` cap). At ~20 ms/cell that's ~35–40 min just to
   enumerate, before validation. A bigger timeout finds proportionally
   more but never completes.
2. **Most cells are wasted.** The large majority of those 100k are
   **bounded** cells that get discarded — only unbounded cells become
   shards. Exact spends most of its time on things it throws away.
3. **The binding constraint is validation throughput, not wall-clock.**
   The `lrs` subprocess per cell (~15 ms) is the bottleneck.

For comparison, the **heuristic** (vectorised ray shooting) finds
~3,800 unbounded cells in ~1 s. In a measured 8 s exact budget, exact
salvaged 32 shards and added **18** that the heuristic missed (union =
3,807). So exact's marginal value over the heuristic on *this* CMF is
real but modest, and grows slowly with time.

---

## 2. The options

### Option A — Replace the `lrs` subprocess with an in-process recession-cone LP
**Idea.** A non-empty cell `C = {x : sᵢ·(Aᵢ·x + cᵢ) > 0}` is unbounded
**iff** its recession cone `{d : sᵢ·Aᵢ·d ≥ 0 ∀i}` contains a nonzero
direction. That is a single LP we can run on the **same hot-started
stateful solver** already used for enumeration — set the per-cell sign
pattern as bounds, maximise `Σ|dⱼ|` over `{sᵢ·Aᵢ·d ≥ 0, ‖d‖∞ ≤ 1}`;
optimum `> 0` ⇒ unbounded.

- **Cost per cell:** ~0.5 ms (in-process LP) vs ~15 ms (`lrs`
  subprocess) → **~30× faster validation**.
- **Effort:** small. One new method + bound-swap; reuse existing
  `_StatefulFeasibilitySolver` machinery.
- **Pros:** removes the dominant bottleneck with no new infrastructure;
  validation nearly disappears as a cost; deterministic.
- **Cons / risks:** deviates from the original lrs-based spec
  (`SHARD_EXTRACTION_PLAN.md` mandated `lrs` for the unbounded check).
  Mitigation: keep `lrs` available as an opt-in cross-check. Need a
  correctness test (recession-cone LP vs `lrs`) on several
  arrangements.
- **Does NOT fix:** the >100k cell count / bounded-cell waste (facts 1
  and 2). Enumeration becomes the new bottleneck (~20 ms/cell).

### Option B — Decouple validation into a queue + worker processes (the line-61 task)
**Idea.** Enumeration (fast producer) streams encodings into a queue;
separate worker processes pull and validate (`lrs` + MILP) in parallel.
Mirrors the search stage's existing `worker_pool` in
`dreamer/utils/multi_processing.py`.

- **Effort:** medium. Producer/consumer plumbing, graceful shutdown,
  result merging, deadline propagation across processes.
- **Pros:** validation no longer blocks enumeration; parallelises
  across cores; idiomatic (matches search-stage pattern).
- **Cons / risks:** real concurrency/shutdown complexity. **Largely
  redundant if Option A lands** — once validation is ~0.5 ms, there's
  little to offload, and enumeration (also LP-bound) becomes the limit.
  Best value is when validation stays expensive (i.e. if we keep
  `lrs`).
- **Does NOT fix:** the cell-count problem.

### Option C — Both (A then B)
Do the recession-cone LP first; then, if profiling still shows
validation as a meaningful share, parallelise it across workers.

- **Effort:** large (sum of A and B).
- **Pros:** maximum throughput on cells we *do* process.
- **Cons:** most code; B's payoff shrinks after A. Probably only worth
  it if we later also keep heavy per-cell work (e.g. computing extra
  shard attributes during validation).

### Option D — Enumerate *only* unbounded cells (the deepest fix)
**Idea.** Skip bounded cells entirely. The unbounded cells of an affine
arrangement correspond to the cells of the **central arrangement at
infinity** (drop the constant terms `cᵢ`, look at the linear parts
`Aᵢ·x = 0`) — a lower-dimensional (D−1 on the sphere) problem with far
fewer cells. Enumerate those, then lift back.

- **Effort:** large + research-y. New enumeration over the arrangement
  at infinity; careful mapping back to affine unbounded cells and their
  interior points.
- **Pros:** attacks facts 1 **and** 2 head-on — eliminates the wasted
  bounded-cell enumeration, so exact could plausibly *finish* on
  high-D. The only option that makes "exact completes on 7D" realistic.
- **Cons / risks:** the biggest correctness surface; needs its own spec
  and verification against brute force on small cases.

### Option E — Just raise `STRATEGY_TIMEOUT_SECONDS`
- **Effort:** trivial (one config value).
- **Pros:** zero code; finds proportionally more shards; Task 3 already
  keeps them (salvage + union with heuristic).
- **Cons:** doesn't finish (fact 1); most added time is spent on bounded
  cells (fact 2); validation throughput unchanged (fact 3). Marginal
  unbounded-shards-added/minute over the heuristic is modest and
  decreasing.

---

## 3. A note on measuring before committing
Before investing in A–D — or setting a timeout — it's cheap to **run
exact-only with a long budget and compare its unbounded-shard set to
the heuristic's**, quantifying "unique shards added per minute." That
tells us whether exact is even worth pursuing on this CMF, or whether
the heuristic already covers what matters. (Recall: 8 s of exact added
18 cells beyond the heuristic's 3,789.)

---

## 4. Suggested sequencing (one opinion, not a decision)
1. **Measure** exact's marginal yield vs the heuristic (cheap, informs
   everything).
2. **Option A** (recession-cone LP) — biggest single throughput win for
   the least code; keep `lrs` as opt-in.
3. Re-measure. If validation is no longer the limit and we still want
   completeness on 7D → **Option D**.
4. **Option B** only if we deliberately keep heavy per-cell work (lrs
   cross-check, or per-cell attribute computation) that's worth
   parallelising.
5. Treat the **timeout** as a knob set *from data* once throughput is
   fixed — not as the primary fix.

---

## 5. Cross-references
- `context/shard_extraction/EXACT_EXTRACTION_OPTIMIZATION_SPEC.md` — the
  Phase 1 (hot-started solver) + Phase 2 (reverse search) work already
  done.
- `dreamer/extraction/v2/cells.py` — reverse search, stateful solver,
  `iter_cells` streaming generator.
- `dreamer/extraction/v2/lrs_extractor.py` — interleaved
  enumerate→classify→locate; where the `lrs` call (Option A target)
  lives.
- `dreamer/extraction/v2/manager.py` — auto strategy, deadline,
  partial-salvage union with the heuristic.
- `dreamer/utils/multi_processing.py` — `worker_pool` (the pattern
  Option B would follow).
