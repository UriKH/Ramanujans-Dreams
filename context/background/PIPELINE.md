# Pipeline Flow & File Responsibilities

> **Doc type:** Design intent + file map. Describes how the pipeline is
> *intended* to be wired and which file owns each stage. The code is the
> source of truth for current behaviour; treat specific function/line
> references here as breadcrumbs that may have drifted — `grep` to confirm
> before relying on them. Update this file when stages are added/removed
> or a significant refactor changes the file map.

---

## 1. End-to-end picture

```
┌──────────┐   ┌────────────┐   ┌──────────┐   ┌────────┐   ┌──────────────┐
│ Loading  │ → │ Extraction │ → │ Analysis │ → │ Search │ → │ Post-process │
└──────────┘   └────────────┘   └──────────┘   └────────┘   └──────────────┘
   CMFs          shards           ranked          per-shard      per-shard
   + shifts    (Ax<b cells)      shards           JSONLs        JSONL patches
```

| Stage          | Orchestrated by                | Default implementation                 |
|----------------|--------------------------------|----------------------------------------|
| Loading        | `System.__loading_stage()`     | DB / formatter sources                 |
| Extraction     | `System.run()` → extractor     | optional, pluggable                    |
| Analysis       | `System.__analysis_stage()`    | `AnalyzerModV1` (handler-based)        |
| Search         | `System.__search_stage()`      | `SearcherModV1` (handler + worker pool)|
| Post-process   | `System.run()` tail            | `Tier3PostProcessModV1`                |

The user wires concrete classes into the `System` constructor; `System.run()`
calls them in order, threading the outputs of one stage as inputs to the next.

---

## 2. Attribute tiers

This is the model the pipeline is organised around — referenced throughout:

| Tier   | Attributes                                                        | When                             | Where                                 | Config key                       |
|--------|-------------------------------------------------------------------|----------------------------------|---------------------------------------|----------------------------------|
| Tier-1 | `delta`, `identified`, `limit`, `order`, `formula`, p/q vectors   | Always — drives filter/sort      | Main thread, Analysis & Search        | (always on)                      |
| Tier-2 | `eigenvalues`, `spectral_gap`, `gcd_slope`, `convergence_class`   | Async during Search              | Worker subprocesses (when configured) | `config.search.TIER2_ATTRIBUTES` |
| Tier-3 | `asymptotics`, `kamidelta`                                        | After Search, in post-process    | Worker subprocesses (when configured) | `config.post_process.TIER3_ATTRIBUTES` |

All defaults are empty tuples → a vanilla run only computes Tier-1.  Opting in
to any tier is a one-liner in code (or config file).  Every name is resolved
through `ATTRIBUTE_REGISTRY` — a misspelt entry raises `KeyError` loudly.

---

## 3. Loading stage

**File:** [`dreamer/system/system.py`](../dreamer/system/system.py) → `System.__loading_stage`

For each provided source (DB module, JSON path, or `Formatter`), produces a
`CMFData` describing one CMF + its coordinate shift.  Optionally persists the
CMFs to `sys_config.EXPORT_CMFS` so later stages (esp. post-process running
standalone) can rebuild them from disk.

**Output:** `Dict[Constant, List[CMFData]]`.

---

## 4. Extraction stage

**File:** any `ExtractionModScheme` passed to `System(extractor=...)`.

Decomposes each CMF's integer lattice into shards (`Ax < b` cells).  Currently
uses hyperplane enumeration; future work covers symmetry-aware extraction (see
[`roadmap.md`](roadmap.md)).

**Output:** `Dict[Constant, List[Searchable]]` (a shard is a `Searchable`).

---

## 5. Analysis stage

**Files:**

- Module: [`dreamer/analysis/analyzers/serial_scan/analyzer_mod.py`](../dreamer/analysis/analyzers/serial_scan/analyzer_mod.py) (`AnalyzerModV1`)
- Per-shard ID derivation: [`dreamer/utils/storage/trajectory_attributes.py`](../dreamer/utils/storage/trajectory_attributes.py) → `derive_cmf_and_shard_ids`, `_serialize_inequalities` (rows sorted to be enumeration-order-independent)
- DTO factory: same file → `build_trajectory_dto`
- Dedup helper: [`dreamer/utils/multi_processing.py`](../dreamer/utils/multi_processing.py) → `load_seen_trajectories`

For each (constant, shard) pair:

1. **Always sample.**  `SerialSearcher.sample_pairs()` enumerates trajectory
   pairs every run — sampling itself is cheap and the strategy may change
   between runs (different `NUM_TRAJECTORIES_FROM_DIM`, etc.), so the
   analyzer must not skip this step.
2. **Per-trajectory dedup, never per-shard.**  Load the existing per-shard
   JSONL via `load_seen_trajectories(...)`.  For each sampled `(traj, start)`
   pair, derive the deterministic `trajectory_id` (cheap; no symbolic work).
   * If the cached record carries both `delta_estimate` and `identified`,
     **reuse them** — no handler is constructed, no walk happens.
   * Otherwise, build a full `TrajectoryDTO` via `build_trajectory_dto`
     (this triggers the trajectory walk for `delta`, `limit`, etc.) and
     append it as one JSON line to the same shard JSONL.
3. **Filter & rank** — accumulate `best_delta` and `identified_pct` from
   the values seen during this run (cached + freshly computed).  Shards
   passing `IDENTIFY_THRESHOLD` are sorted by best delta descending (dim
   ascending tie-break).

**Output:**
- In memory: `Dict[Constant, List[Searchable]]` — ranked shards per constant.
- On disk: append-only per-trajectory records in the same canonical
  location used by Search:
  `<sys_config.EXPORT_SEARCH_RESULTS>/<constant>/<cmf>__<shard_id>.jsonl`.
  This is the **single canonical store** for per-trajectory data — analyzer,
  search, and post-process all read/write the same files.

The analyzer is **sequential by design**.  Its cost on re-runs is
proportional to the number of *new* trajectories sampled (the cached ones
cost only a dict lookup and a `trajectory_id` hash).  Parallelising over
shards is on the roadmap but not yet needed.

---

## 6. Search stage

**Files:**

- Module: [`dreamer/search/searchers/hedgehog_scan_mod.py`](../dreamer/search/searchers/hedgehog_scan_mod.py) (`SearcherModV1`)
- Per-shard producer: same file, `_produce`
- Worker / writer functions: [`dreamer/utils/multi_processing.py`](../dreamer/utils/multi_processing.py) → `compute_tier2_for_item`, `write_jsonl_line`
- Generic plumbing: [`dreamer/utils/multi_processing.py`](../dreamer/utils/multi_processing.py) → `worker_pool` (context manager — see §8)
- DTO factory & helpers: [`dreamer/utils/storage/trajectory_attributes.py`](../dreamer/utils/storage/trajectory_attributes.py)
- Cross-run dedup helper: same file as worker — `load_seen_trajectories`

**Per-shard producer flow** (`_produce`):

```
for (traj, start) in sample_pairs:
    trajectory_id = stable_hash(cmf_name, shard_encoding, start, direction)  # cheap

    if trajectory_id in seen_trajectories:
        if no Tier-2 attrs are missing:
            continue                                       # ← EARLY SKIP, no handler built
        else:
            handler = TrajectoryAttributesHandler.from_cmf(...)   # symbolic only, no walk
            push((handler.trajectory_matrix(), patch_dict))       # ← PATCH case
    else:
        handler = TrajectoryAttributesHandler.from_cmf(...)
        dto = build_trajectory_dto(handler, ...)          # walks happen here (Tier-1)
        push((handler.trajectory_matrix(), dto))           # ← NEW case
```

The early-skip is the key re-run optimisation: a second run over the same
data does **not** repeat any trajectory walks.

**Dispatch via `worker_pool`** — one call covers both modes; the same
`worker_fn` (which unwraps the producer's tuple) runs in **either** a
subprocess or the main thread, depending on the `parallel` flag:

| `TIER2_ATTRIBUTES` | `parallel` | Effect                                                                                |
|--------------------|-----------|---------------------------------------------------------------------------------------|
| empty (default)    | `False`   | `compute_tier2_for_item` runs on the main thread (a fast no-op when nothing's missing), and `write_jsonl_line` writes inline. No subprocess. |
| non-empty          | `True`    | `compute_tier2_for_item` runs in `NUM_BACKGROUND_WORKERS` subprocesses; a dedicated writer subprocess owns the JSONL. |

**Output files:**
`<sys_config.EXPORT_SEARCH_RESULTS>/<constant>/<cmf>__<shard_id>.jsonl`

These files are the **single canonical per-trajectory store** — the
analyzer writes Tier-1 DTO records here too, so when the search stage
runs, it typically finds the trajectories already on file with all Tier-1
fields present.  The early-skip then bypasses the walk entirely; if
Tier-2 attributes are configured, the partial-coverage path emits a
patch instead of a fresh DTO.

Each line is either a full `TrajectoryDTO` record or a partial *patch*
record (`{trajectory_id, extended_metrics}`).  The file is **append-only**;
readers merge by `trajectory_id` (later records win for conflicting keys;
`extended_metrics` is deep-merged) — see [`dreamer/utils/storage/importer.py`](../dreamer/utils/storage/importer.py) → `_read_jsonl(merge=True)`.

---

## 7. Post-process stage  *(Tier-3)*

**Files:**

- Scheme: [`dreamer/utils/schemes/post_process_scheme.py`](../dreamer/utils/schemes/post_process_scheme.py) → `PostProcessModScheme`
- Module: [`dreamer/post_process/tier3_post_process_mod.py`](../dreamer/post_process/tier3_post_process_mod.py) → `Tier3PostProcessModV1`
- Per-item worker: same file → `compute_tier3_for_item` (module-level so it pickles)
- Config: [`dreamer/configs/post_process.py`](../dreamer/configs/post_process.py)

Runs **once**, after the Search stage finishes, when:
- A `post_processor=` was wired into `System(...)`, **and**
- `config.post_process.TIER3_ATTRIBUTES` is non-empty.

**Flow:**

```
build cmf_lookup {cmf_name → CMF}:
    primary: from the search priorities (every Searchable carries its CMF)
    fallback: load .pkl files under sys_config.EXPORT_CMFS

for each constant → for each <cmf>__<shard_id>.jsonl:
    merged = load_seen_trajectories(file)
    if no record is missing any TIER3 attr:
        continue            # cheap pre-flight; skips worker_pool entirely

    with worker_pool(num_workers=..., worker_fn=compute_tier3_for_item,
                     writer_fn=write_jsonl_line, output_path=file, ...) as push:
        for tid, record in merged.items():
            if all TIER3 attrs present: continue
            cmf = cmf_lookup[record["cmf_id"]]
            start, direction = reconstruct_positions(cmf, record)
            handler = TrajectoryAttributesHandler.from_cmf(cmf, direction, start)
            push((handler.trajectory_matrix(), {"trajectory_id": tid, "extended_metrics": {}}))
    # worker_pool's __exit__ sends sentinels and joins all subprocesses
```

The worker (`compute_tier3_for_item`) computes the missing Tier-3 attributes
and returns the patch dict; the writer subprocess appends one JSON line per
patch.  Subsequent reads of the file see the merged result transparently.

**Why a single Tier-3 module covers a deferred design item:**
- Tier-3 ran nowhere previously (search workers used to default to four
  Tier-2 attributes labelled "Tier-3" in old code — that confusion is now
  resolved in the rename).
- Asymptotics / kamidelta are genuinely expensive; running them only when
  explicitly requested matches how the user described the intent.

---

## 8. Parallelism abstraction — `worker_pool`

**File:** [`dreamer/utils/multi_processing.py`](../dreamer/utils/multi_processing.py) → `worker_pool` (context manager)

A single context manager hides every piece of subprocess and queue glue:

```python
with worker_pool(
    num_workers=4,
    worker_fn=process_item,        # required (use None → identity if there's
                                   #            genuinely no transform)
    writer_fn=write_one_line,      # (item, fout) → None
    output_path="…/out.jsonl",
    config_overrides=config.export_configurations(),
    parallel=True,                 # False → run worker_fn on the main thread,
                                   # no subprocess.  Defaults to True when
                                   # worker_fn is provided.
) as push:
    for item in producer_items():
        push(item)
# On exit: N task-queue sentinels, joins, results-queue sentinel, writer join.
# Even if the producer raises.
```

Crucially, ``worker_fn`` runs in **both** modes — the only difference is
whether it runs in a subprocess or on the main thread.  Producers can
therefore push the same item shape regardless of the configured mode and
let ``worker_fn`` do any unpacking the writer doesn't know how to do
(e.g. the Search stage pushes ``(traj_matrix, payload)`` tuples; the
worker unwraps them before they reach ``write_jsonl_line``).

What it hides:
- `mp.Queue(maxsize=...)` × 2 (defaults to `max(32, 4*num_workers)`)
- `mp.Process(target=worker_loop, ...)` × N
- One dedicated writer `mp.Process`
- `_init_worker(config_overrides)` config propagation
- Per-worker `task_queue.get(timeout=3)` loop with sentinel handling
- Writer's append-open file + `flush()` discipline
- Try/finally cleanup so the caller never has to write it

Both the Search stage and the Post-process stage use this one abstraction.
Adding a new stage now means:
1. Write a module-level `worker_fn(item) -> item` (or None for direct mode).
2. Write a module-level `writer_fn(item, fout) -> None`.
3. In the stage class, call `with worker_pool(...) as push: producer(push)`.

No `mp.Queue`, no sentinels, no joins in stage code.

---

## 9. Storage layer

**Files:**

- DTOs: [`dreamer/utils/storage/dtos.py`](../dreamer/utils/storage/dtos.py) — `TrajectoryDTO`, `ShardDTO`, `CmfDTO`, `CmfFamilyDTO`.  All implement `to_json_line()` + `from_dict()`.
- Attribute registry: [`dreamer/utils/storage/attribute_registry.py`](../dreamer/utils/storage/attribute_registry.py) — `ATTRIBUTE_REGISTRY`, `compute_attribute`, `compute_attributes`, `register_attribute`.
- Handler: [`dreamer/utils/storage/trajectory_attributes.py`](../dreamer/utils/storage/trajectory_attributes.py) — `TrajectoryAttributesHandler` (lazy-cached delta / limit / formula / etc.), `build_trajectory_dto`, `derive_cmf_and_shard_ids`, `_stable_id`, `_position_to_tuple`.
- Importer / Exporter: [`dreamer/utils/storage/importer.py`](../dreamer/utils/storage/importer.py) (with `merge=True` for trajectory JSONL), [`dreamer/utils/storage/exporter.py`](../dreamer/utils/storage/exporter.py), formats in [`dreamer/utils/storage/formats.py`](../dreamer/utils/storage/formats.py).

The merge-on-read model (append-only JSONL + per-trajectory patches keyed on
`trajectory_id`) is what makes both the Search re-run dedup *and* Tier-3 patch
appending work without locking or compaction.

---

## 10. Configuration

**Files:** [`dreamer/configs/`](../dreamer/configs/)

| Section        | File                            | Key fields used in this pipeline                                                         |
|----------------|---------------------------------|------------------------------------------------------------------------------------------|
| `system`       | `system.py`                     | `NUM_BACKGROUND_WORKERS`, `EXPORT_*` paths, `MODULE_ERROR_SHOW_TRACE`                    |
| `analysis`     | `analysis.py`                   | `NUM_TRAJECTORIES_FROM_DIM`, `IDENTIFY_THRESHOLD`                                        |
| `search`       | `search.py`                     | `TIER2_ATTRIBUTES` (default `()`), `NUM_TRAJECTORIES_FROM_DIM`, walking knobs            |
| `post_process` | `post_process.py`               | `TIER3_ATTRIBUTES` (default `()`)                                                        |
| (registry)     | `config_manager.py`             | aggregates all sections; `configure(**overrides)` for per-section updates                |

All sections inherit from `Configurable` (`configurable.py`), which provides
`asdict(self)` export — used by `config.export_configurations()` to propagate
the entire config into every worker subprocess at startup.

---

## 11. Key decisions made along the way

The roadmap's Completed log has the dated rationale.  Highlights:

- **Stable `shard_id`** — `_serialize_inequalities` sorts constraint rows so
  the SHA-256 is independent of hyperplane enumeration order.  This is what
  makes any kind of cross-run dedup possible.
- **Early trajectory-id skip** — derive the id from name+encoding+start+direction
  before building the handler; re-runs cost ~0 per fully-covered trajectory.
- **`worker_pool` collapses to direct mode** — `worker_fn=None` short-circuits
  the entire MPMC pipeline; vanilla runs (no Tier-2/Tier-3 work) never spawn a
  subprocess.
- **Per-trajectory dedup everywhere, never per-shard** — analyzer and search
  both sample trajectories every run and dedup at the `trajectory_id` level
  against a single canonical per-shard JSONL.  A new analyzer with a
  different sampling strategy can be added without invalidating existing
  trajectory data; the new trajectories are computed and appended, the old
  ones are reused.
- **Append-only JSONL + merge-on-read** — patches add data; readers merge.
  Lock-free, crash-safe, incrementally extensible.

---

## 12. Where new work belongs

If you're extending the pipeline, here's a rough decision tree:

- **A new trajectory attribute that's cheap and always wanted** → add it to
  `TrajectoryAttributesHandler` and `TrajectoryDTO`; populate it in
  `build_trajectory_dto`.  No tier change.
- **A new trajectory attribute that's expensive and you want it computed
  in parallel during search** → register it in `ATTRIBUTE_REGISTRY`, add the
  name to `config.search.TIER2_ATTRIBUTES`.  No code change in the searcher.
- **A new attribute that's even more expensive and only sometimes wanted** →
  register it; add to `config.post_process.TIER3_ATTRIBUTES`.  Optionally wire
  a different `post_processor=` if you want a non-default policy.
- **A whole new stage that runs in parallel** → wrap its body in
  `with worker_pool(...) as push:`.  Define module-level `worker_fn` /
  `writer_fn`.  Plug into `System` via its own scheme.
