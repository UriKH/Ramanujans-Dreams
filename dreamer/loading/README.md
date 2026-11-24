# Loading stage — `dreamer/loading/`

> **Goal:** turn each target constant into one or more **CMFs to search**.
> **Input:** the `function_sources` you pass to `System` (formatters, database
> modules, or export paths) + the run's constants.
> **Output:** `Dict[Constant, List[CMFData]]` — for every constant, the list of
> CMFs (each a `ramanujantools` CMF plus an integer coordinate *shift*) in which
> the pipeline will look for formulas.

For where this stage sits in the whole pipeline, see the
[main README](../../README.md#system-structure).

---

## What this stage does

A **CMF** (Continued Matrix Field) is the mathematical object this whole system
searches. The loading stage is the only place that *creates* CMFs; every later
stage just consumes them. You never build a CMF matrix by hand — instead you
name an **inspiration function** (e.g. a hypergeometric ${}_pF_q$), and a
**formatter** turns it into the corresponding CMF.

Three kinds of source can be mixed freely in `function_sources`:

| Source kind | What it is | Example |
|-------------|------------|---------|
| **Formatter** | A live inspiration function turned into a CMF. | `pFq(log(2), 2, 1, -1)` |
| **Database module** | A `DBModScheme` that retrieves/stores CMF descriptions in a DB. | `BasicDBMod(json_path=...)` |
| **Path string** | A directory of previously-exported CMFs to reload (no pickles). | `"./CMFs/pi"` |

A `CMFData` bundles the CMF, an integer (or exact `sp.Rational`) **shift** that
moves the search's start point, and metadata. The same `CMFData` can be attached
to several constants at once, so a CMF shared by `pi` and `log(2)` is loaded once.

### Formatters and the no-pickle contract

`Formatter` (in `funcs/formatter.py`) is the bridge between a live CMF and a
**JSON-serialisable** description of it. Every formatter:

- normalises the constant(s) and validates the shifts (a `float` or irrational
  shift is rejected — shifts must be exact integers or `sp.Rational`);
- builds a deterministic `cmf_name` that **excludes the constant**, so the same
  CMF keeps a stable id even when you later search it for a different constant;
- can round-trip to/from JSON, which is what lets the pipeline export CMFs and
  reload them later **without pickle files**.

Concrete formatters live in `funcs/` — currently `pFq` (hypergeometric),
`MeijerG`, and `BaseCMF` (wrap an arbitrary `ramanujantools` CMF directly).

### Walk type of a CMF (`use_inv_t`)

Every formatter takes `use_inv_t`, the walk type of that CMF:

| `use_inv_t` | Walk read by RD |
|---|---|
| `None` (default) | follows the global `config.search.DEFAULT_USES_INV_T` **at walk time** |
| `True` | type 1 — the inverse transpose `W^-T` of the walk (RD's mathematics is built on it) |
| `False` | type 2 — the walk `W` itself |

An explicit `True` / `False` overrides the global setting for that CMF only. `None` is
saved to the exported JSON as `null`, so a reloaded CMF keeps following the global.
The resolved walk type is stored in every result row (`walk_type`) and is part of the
row's fingerprints. Resolution rule: `dreamer/utils/walk_type.py`.

### Type-2 requests of pFq and MeijerG are walked as type 1

RD's mathematics is built on type-1 walks. A **type-2** walk of a CMF is the type-1 walk of
the CMF of its **formal adjoint**, and for `pFq` and `MeijerG` that is again a CMF of the
same family. So when a `pFq` / `MeijerG` formatter resolves to walk type 2, `to_cmf()`
returns the adjoint CMF, walked as type 1 (`funcs/walk_type_conversion.py`):

| Family | start | direction | z |
|---|---|---|---|
| `pFq(p, q, z)` | `x → 1 − x`, `y → 2 − y` | `d → −d` | `(−1)^(p+q+1) z` (same z for p = q + 1; `−z` for 1F1, 2F2, 3F1) |
| `MeijerG(m, n, p, q, z)` | `a → 1 − a`, `b → −b` | `d → −d` | `(−1)^(p+q) z` (`−z` for odd p + q, e.g. 𝒢 G^{0,2}_{2,1}, γ G^{3,2}_{2,3}) |

- z, the shifts (`s → −s`), every selected start point and selected trajectory are
  converted. The map is its own inverse, so a walked ray maps back exactly.
- The converted CMF has the adjoint CMF's `cmf_name`, so it shares shards and stored
  rows with type-1 runs of that CMF. Each row it writes also stores the ray in the
  requested coordinates: `"type2": {"cmf", "start_point", "direction"}`.
- A converted seed whose type-1 walk would hit a pole or a singular step is **reported and
  skipped**, not fatal: a warning event `type2_seed_unwalkable` (screen + run monitor) names
  the type-2 seed, the bad steps and the smallest walkable shift `k`, and the run continues
  with the other seeds and shards. RD never moves a seed on its own. With
  `only_selected=True` and no walkable seed left, the CMF yields no shards
  (event `type2_no_walkable_seed`).
- The exported CMF JSON keeps the type-2 request (it is converted again on reload).
- Other families (`BaseCMF`) have no such map and are walked natively as type 2.
- There is no separate switch: the walk type (`use_inv_t` / `DEFAULT_USES_INV_T`) decides.

---

## Directory contents

| Path | What it contains |
|------|------------------|
| `funcs/formatter.py` | `Formatter` ABC — the CMF ⇄ JSON bridge and self-registry. |
| `funcs/pFq_fmt.py` | `pFq` formatter — hypergeometric ${}_pF_q$ inspiration functions. |
| `funcs/meijerG_fmt.py` | `MeijerG` formatter — Meijer-G inspiration functions. |
| `funcs/base_cmf.py` | `BaseCMF` — wrap an existing `ramanujantools` CMF with no special structure. |
| `funcs/walk_type_conversion.py` | Type 2 → type 1 conversion of pFq / MeijerG requests (formal adjoint), pole check, `Type2Source`. |
| `databases/db_v1/` | A reference database module: `db.py` (the store), `db_mod.py` (`BasicDBMod` / `DBModScheme`), `config.py`. |
| `config.py` | Loading-stage annotation constants (DB command/data tags). |
| `errors.py` | Loading-specific exceptions. |

---

## Key configurations

Loading has no dedicated config *category*; it is driven by **what you pass to
`System(function_sources=...)`** and by the database config:

- `config.database.USAGE` — retrieve / store / store-then-retrieve mode for DB sources.
- `config.system.EXPORT_CMFS` — directory the loaded CMFs are exported to (the
  JSON artefact that later enables extractor-free reloads).

See the [configuration index](../configs/README.md) for the full list.

---

## Extending this stage

**Add a new inspiration function** → subclass `Formatter` in a new file under
`funcs/` and implement its CMF construction + JSON round-trip; subclasses
auto-register, so `Formatter.from_json_obj` can rebuild them. Use
`funcs/pFq_fmt.py` as the reference.

**Add a new CMF source** (e.g. a different database backend) → subclass
`DBModScheme` and implement `execute(constants)` to return
`Dict[Constant, List[CMFData]]`. Pass an instance in `function_sources`.
