# Architecture & Design — Ramanujan's Dreams

> **Doc type:** Design intent + decision log. Captures *why* the architecture
> is shaped the way it is. Code is authoritative for current behaviour; this
> file carries rationale that won't be obvious from the source. Update when
> a significant design decision is made or the architecture's shape changes.
> For a higher-level mission statement and invariants, see [`SYSTEM_SPEC.md`](../SYSTEM_SPEC.md).

---

## 1. High-Level Architecture

The system is a **four-stage modular pipeline** that searches for polynomial continued fractions (PCFs) of mathematical constants via Conservative Matrix Fields (CMFs).

```
 Constant + Inspiration Functions
               │
               ▼
  ┌────────────────────────────┐
  │  1. LOADING                 │  Constants → CMFs
  │     dreamer/loading/        │
  └──────────────┬─────────────┘
                 │ Dict[Constant, List[CMFData]]
                 ▼
  ┌────────────────────────────┐
  │  2. EXTRACTION              │  CMFs → bounded convex regions (Shards)
  │     dreamer/extraction/     │
  └──────────────┬─────────────┘
                 │ Dict[Constant, List[Shard]]
                 ▼
  ┌────────────────────────────┐
  │  3. ANALYSIS                │  Shards → filtered & ranked Shards
  │     dreamer/analysis/       │
  └──────────────┬─────────────┘
                 │ Dict[Constant, List[Shard]]  (prioritized)
                 ▼
  ┌────────────────────────────┐
  │  4. SEARCH                  │  Deep search → discovered PCFs
  │     dreamer/search/         │
  └────────────────────────────┘
                 │ Dict[Searchable, DataManager]  (exported to disk)
```

Each stage consumes the previous stage's output unchanged (invariant 5 in `SYSTEM_SPEC.md`). Stages are substitutable — any conforming module can replace the default one without touching `System`.

---

## 2. Pipeline Orchestrator

**File:** [`dreamer/system/system.py`](../dreamer/system/system.py)

`System` drives the full pipeline. Its responsibilities:
- Accept module instances at construction time (extractor, analyzers, searcher)
- Route `Dict[Constant, List[...]]` data between stages
- Export intermediate results to disk (CMFs, shards, priorities, search results)
- Log progress and surface the best results

```python
system = System(
    function_sources=[pFq_source, db_source],
    extractor=ShardExtractorMod,
    analyzers=[AnalyzerModV1],
    searcher=SearcherModV1
)
system.run(constants=["pi"])
```

---

## 3. Stage Details

### 3.1 Loading

**Directory:** `dreamer/loading/`

**Goal:** Produce `Dict[Constant, List[CMFData]]` from any combination of sources.

| Component | Role |
|-----------|------|
| `Formatter` (ABC) | JSON-serializable wrapper around a `ramanujantools` CMF |
| `pFq` | Hypergeometric $_pF_q$ CMF formatter |
| `MeijerG` | Meijer-G CMF formatter |
| `BaseCMF` | Raw CMF formatter for arbitrary CMFs |
| `BasicDBMod` | Loads formatters from a SQLite database |
| `CMFData` | Frozen dataclass wrapping CMF + shift + metadata |

**Serialization contract:** Every `Formatter` must round-trip through JSON — `from_json_obj(to_json_obj(x))` produces an equivalent object.

**Shift:** Each `CMFData` carries a `Position` shift (integer offsets per symbol) that moves the origin of the CMF's lattice. This is how we avoid the origin where matrices may be singular or poles exist.

---

### 3.2 Extraction

**Directory:** `dreamer/extraction/`

**Goal:** Partition the CMF's integer lattice into bounded convex regions (Shards).

**Algorithm:**
1. For each matrix $M_s$ in the CMF: compute zeros of the characteristic polynomial and poles of rational entries — each becomes a hyperplane $\sum_i c_i x_i = k$.
2. Enumerate all $2^H$ sign-vector encodings (above/below each hyperplane).
3. For each encoding: verify the region is non-empty by finding an interior point (via LP).
4. Retain non-empty regions as `Shard` objects with their `Ax < b` inequality system.

| Component | Role |
|-----------|------|
| `Hyperplane` | Single linear constraint $\sum c_i x_i = k$; converts to `(A_row, b)` |
| `Shard` | Bounded convex region; stores `A`, `b`, encoding, interior point |
| `ShardExtractorMod` | Orchestrator — drives the extraction algorithm |
| `ShardExtractor` | Core extraction logic |
| `*Sampler` classes | Generate trajectory directions inside a shard |
| `ShardSamplingOrchestrator` | Coordinates samplers across shards |

**Sampling methods available:**
- `SphereSampler` — random direction on unit sphere
- `RaycastSampler` — raycast from interior along a direction
- `CHRRSampler` — Center-Hit Ray Recursion sampler

---

### 3.3 Analysis

**Directory:** `dreamer/analysis/`

**Goal:** Filter out unlikely shards and rank the rest by discovery promise.

**Algorithm:**
1. Load the per-constant JSONL (`<constant>.jsonl`) into a `{shard_id → cached_record}` map for cross-run dedup.
2. For each shard: if `shard_id` is already in the map, **reuse `best_delta`/`identified_pct` from the cached record** — no resampling, no walks.
3. Otherwise: sample ~$10^d$ trajectories (d = CMF dimension), walk the CMF along each, and compute **Tier-1 attributes** (`delta`, `identified`) via `TrajectoryAttributesHandler`.
4. Discard shards where fewer than `IDENTIFY_THRESHOLD`% of trajectories converge.
5. Rank remaining shards by best $\delta$ found.

The analysis stage computes Tier-1 attributes **only** — heavier work belongs in Search (Tier-2) or the post-process stage (Tier-3).

| Component | Role |
|-----------|------|
| `AnalyzerModV1` | Orchestrator — runs analysis across all shards |
| `Analyzer` | Per-shard analysis logic |
| `SerialSearcher` | Walks trajectories and computes $\delta$ |

**Key config:** `config.analysis.IDENTIFY_THRESHOLD` (default: `-1` = accept all).

When multiple analyzers are provided to `System`, their rankings are merged via a consensus graph (NetworkX-based ranking).

---

### 3.4 Search

**Directory:** `dreamer/search/`

**Goal:** Search prioritized shards more deeply for exact PCFs; record rich per-trajectory attributes.

**Pipeline (one per shard):**

```
Producer (main thread)
  For each (trajectory, start) pair:
    1. Derive trajectory_id (no symbolic work, no trajectory walk).
    2. If id already in seen_trajectories with every desired Tier-2 attr
       present → skip entirely (no handler built, no walk).
    3. Else if id present but Tier-2 attrs missing → build handler (cheap
       symbolic matrix only), emit patch dict for workers.
    4. Else (new) → build full Tier-1 DTO (this triggers the walk).
    5. Hand result to sink.
```

The sink dispatches by configured Tier-2 list:
- **`TIER2_ATTRIBUTES` empty (default)** — main thread writes the DTO straight to the JSONL. No worker or writer subprocess is spawned, eliminating per-shard fork overhead.
- **`TIER2_ATTRIBUTES` non-empty** — full MPMC pipeline: bounded task queue → `NUM_BACKGROUND_WORKERS` worker processes that compute the missing Tier-2 attributes → results queue → dedicated writer process that owns the JSONL file.

**Three cases per trajectory in the producer:**
- **Complete** — every configured Tier-2 attribute already present: skip immediately, before any handler is built.
- **Partial** — seen but missing some configured attributes: build a patch dict `{"trajectory_id": ..., "extended_metrics": {}}` for the worker to fill.
- **New** — not yet seen: build full `TrajectoryDTO` (walks happen here) and ship the trajectory matrix to the worker.

| Component | Role |
|-----------|------|
| `SearcherModV1` | Default orchestrator — runs MPMC pipeline per shard |
| `SerialSearcher` | Per-trajectory walk + $\delta$ computation |
| `GeneticSearcher` | Genetic algorithm variant (experimental) |
| `TrajectoryDTO` | Per-trajectory result DTO with `extended_metrics` for all tiers |

See §3.5 for the attribute tier definitions and JSONL persistence model.

---

### 3.5 Attribute Tier System

Trajectory attributes are split into three tiers by compute cost and stage:

| Tier | Attributes | When computed | Where | Config key |
|------|-----------|--------------|-------|------------|
| **Tier 1** | `delta`, `identified`, `limit`, `order`, `formula`, p/q vectors | Always — drives filter/sort | Main thread, both Analysis and Search | (always on) |
| **Tier 2** | `eigenvalues`, `spectral_gap`, `gcd_slope`, `convergence_class` (configurable) | Asynchronously in background workers, **Search stage only** | Child processes (skipped when empty) | `TIER2_ATTRIBUTES` (default `()`) |
| **Tier 3** | `asymptotics`, `kamidelta` | Post-process stage (deferred — not yet implemented) | Main thread, separate pipeline pass | TBD |

**Attribute Registry:** All non-Tier-1 attributes are registered in `dreamer/utils/storage/attribute_registry.py` (`ATTRIBUTE_REGISTRY`). Config lists are validated against this registry at run time (a misspelled name raises `KeyError` loudly).

**JSONL append-only / merge-on-read (Search outputs):**

Search results are written as JSONL (one JSON object per line). The file is **append-only** — no line is ever modified or deleted. When new attributes are added to `TIER2_ATTRIBUTES` after trajectories have already been written, a re-run appends **patch records** containing only `trajectory_id` + the new `extended_metrics` entries. Readers merge all records with the same `trajectory_id` (later records win for conflicting keys; `extended_metrics` is deep-merged). This is an event-sourcing / CRDT pattern — crash-safe, lock-free, and incrementally extensible.

```
<cmf_name>__<shard_id>.jsonl
  line 1: {"trajectory_id": "abc", "delta_estimate": 0.9, ..., "extended_metrics": {"gcd_slope": 1.2}}   ← base record
  line 2: {"trajectory_id": "abc", "extended_metrics": {"kamidelta": 0.7}}                                ← patch record
  merged:  {"trajectory_id": "abc", "delta_estimate": 0.9, ..., "extended_metrics": {"gcd_slope": 1.2, "kamidelta": 0.7}}
```

`Importer._read_jsonl(path, merge=True)` performs this merge; `merge=False` (default) returns raw lines.

**Shard ID stability:** The output filename contains the `shard_id`, a SHA-256 of the canonical `Ax < b` system. Rows of `A` are sorted lexicographically before hashing so the ID is independent of hyperplane enumeration order between runs.

---

## 4. Module/Scheme Pattern

Every stage follows the same structural pattern:

```
dreamer/<stage>/
├── <plural>/<name>/
│   ├── <name>_mod.py       # Orchestrator — implements execute()
│   └── config.py           # Stage-specific config
└── methods/
    └── <method_name>.py    # Algorithmic implementation
```

**Abstract base classes** (`dreamer/utils/schemes/`):
- `ExtractionModScheme` — implemented by `ShardExtractorMod`
- `AnalyzerModScheme` — implemented by `AnalyzerModV1`
- `SearcherModScheme` — implemented by `SearcherModV1`, `GeneticModV1`
- `DBModScheme` — implemented by `BasicDBMod`

Any new module must inherit the relevant scheme and implement `execute()`. Register it in the stage's `__init__.py`.

---

## 5. Configuration System

**Directory:** `dreamer/configs/`

All configuration is accessed through a global `config` object:

```python
from dreamer import config

config.analysis.IDENTIFY_THRESHOLD = 0.5
config.search.DEPTH_FROM_TRAJECTORY_LEN = lambda n: 10 * n
config.system.EXPORT_SEARCH_RESULTS = "/path/to/output"
```

Configs are dataclasses, one per stage:

| File | Covers |
|------|--------|
| `analysis.py` | Threshold, what attributes to compute, trajectory count |
| `extraction.py` | Sampling settings, parallelism |
| `search.py` | Walk depth, trajectory density, export format |
| `system.py` | Export paths, DB usage mode |
| `database.py` | DB path, usage (retrieve / store / both) |
| `logging.py` | Log level, output destination |

---

## 6. Data Transfer Objects (DTOs)

**File:** [`dreamer/utils/storage/dtos.py`](../dreamer/utils/storage/dtos.py)

Frozen dataclasses designed for persistent storage and eventual DB migration:

| DTO | Represents | Key fields |
|-----|------------|------------|
| `CmfFamilyDTO` | A CMF family (e.g., 4F3) | `family_id`, `global_family_id`, `matrix_definitions` |
| `CmfDTO` | A specific CMF instance | `cmf_id`, `family_id`, `cmf_hyperplanes`, `found_constants` |
| `ShardDTO` | A shard within a CMF | `shard_id`, `cmf_id`, `shard_encoding`, `interior_point`, volume/defect metrics |
| `TrajectoryDTO` | A trajectory result | `trajectory_id`, start/direction, recurrence relation, $\delta$, p/q vectors, `extended_metrics` |

`extended_metrics: Dict` in `TrajectoryDTO` is an intentional open field for future attributes without requiring a schema migration.

---

## 7. Serialization Strategy

| Object | Format | Why |
|--------|--------|-----|
| `Formatter` (pFq, MeijerG) | JSON | Human-readable; round-trip required |
| `Shard` | Base64-encoded pickle, wrapped in JSON | CMF matrices contain complex symbolic objects not directly JSON-serializable |
| `DataManager` / `SearchData` | Pickle or JSON (configurable) | Large result sets; JSON for portability |
| `TrajectoryDTO` | JSONL (planned) | Append-friendly; easy DB migration |

Serialization format for search results is controlled by `config.system.EXPORT_SEARCHABLES_FORMAT`.

---

## 8. Storage Layout (on disk)

Default export structure under `config.system.EXPORT_*` paths:
```
<export_root>/
├── cmfs/          # CMFData objects per constant
├── shards/        # Shard objects after extraction
├── priorities/    # Ranked shards after analysis
└── results/
    └── <const_name>/
        └── <cmf_name>__<shard_id>.jsonl   # TrajectoryDTOs + patch records per shard (append-only JSONL)
```

The `shard_id` is a 16-hex-char SHA-256 of the shard's canonical `Ax < b` system (rows sorted); see §3.5.

---

## 9. External Dependencies

| Dependency | Role | Notes |
|------------|------|-------|
| `ramanujantools` | CMF engine: `CMF`, `Position`, `Matrix`, `Limit`, `pFq`, `MeijerG` | All CMF operations go through this library |
| `LIReC` | Integer relation detection and constant identification | Linux-only; Windows requires WSL |
| `sympy` | Symbolic math: matrix expressions, hyperplane equations | Used for exact symbolic manipulation |
| `mpmath` | Arbitrary-precision arithmetic (≥100 digits) | All numerical verification; Python `float` is forbidden |
| `numpy` | Linear algebra: `Ax < b` shard checks, hyperplane vectors | Used in extraction and sampling |
| `networkx` | Graph-based consensus ranking of shards | Used when multiple analyzers are combined |

---

## 10. Known Limitations

| Area | Limitation |
|------|-----------|
| **LIReC on Windows** | `cysignals`/`fpylll` are Linux-only; tests with LIReC require WSL |
| **Parallelism** | Extraction parallelism is commented out; search uses basic chunking |
| **Trajectory sampling uniformity** | Degrades in high dimensions — thin cones may be undersampled |
| **Database** | SQLite is local-only; no shared/remote DB yet |
| **Search depth** | Max depth 1500; some PCFs require deeper walks |
| **Ascent logic** | Not yet implemented (PCF → CMF → higher identity) |
| **Proof generation** | System produces conjectures, not proofs |

---

## 11. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2024 | Use `ramanujantools` as CMF engine | Avoid duplication; leverage tested group library |
| 2024 | SQLite for local DB | Simplicity; no server needed for single-user runs |
| 2024 | Modular stage architecture with abstract schemes | Allow independent development of new analyzers/searchers |
| 2025 | `IDENTIFY_THRESHOLD = -1` means "accept all" | Disables filtering for exploratory runs |
| 2026-04 | Fix `Constant.value_mpmath` infinite recursion | `@cached_property` was calling itself; use `_explicit_mpmath` backing field |
| 2026-04 | Fix `MeijerG.__init__` super() argument order | `use_inv_t` was being passed as `shifts` positional arg |
| 2026-04 | Fix `SearcherModV1.execute()` empty return | `dms` dict was never populated with search results |
| 2026-05 | Base64-encoded pickle for Shard JSON | CMF matrices contain complex sympy objects not directly JSON-serializable |
| 2026-05 | Three-tier attribute system for trajectories | Separates cheap always-on attrs (Tier 1) from configurable synchronous extras (Tier 2) and expensive async attrs (Tier 3); allows incremental attribute addition without recomputing everything |
| 2026-05 | Append-only JSONL + merge-on-read for search results | Patch records let new attributes be added to existing trajectories across runs without rewriting files; crash-safe and lock-free |
| 2026-05 | Sort `Ax < b` rows before hashing for shard_id | Row order from hyperplane enumeration is non-deterministic between runs; sorting gives a stable canonical ID so the same shard always maps to the same output file |

---

*See also:* [`MATH_OBJECTS.md`](MATH_OBJECTS.md) for the mathematical objects, [`SYSTEM_SPEC.md`](../SYSTEM_SPEC.md) for invariants and priorities.
