# Test Audit Report (2026-04-16)

Bottom line: pool-migration regressions were fixed in the affected tests by switching stubs from `Pool.map/starmap` assumptions to `create_pool()` + `imap_unordered(...)` behavior, and the targeted files now pass (`17 passed`) with the required repository-wide coverage command also passing (`164 passed`).

## Touched Modules (Detailed Review)

| Touched module | Changes made | Coverage evidence | Challenge rubric (/5) | Regression evidence |
|---|---|---|---:|---|
| `tests/test_extraction_initial_points.py` | Replaced outdated `mp.Pool` monkeypatching with `initial_points.create_pool` monkeypatching, added an `imap_unordered`-compatible sequential pool stub, and made iterator stub length-aware for `SmartTQDM(total=...)`. | Included in targeted run (`17 passed`) and full suite run (`164 passed`). | 4 | `test_compute_mapping_selects_closest_point_per_signature`, `test_compute_mapping_tie_breaks_lexicographically` now explicitly guard pool-API compatibility. |
| `tests/test_search_genetic.py` | Updated parallel-path pool double to `imap_unordered` semantics and added `DummySpace.compute_trajectory_data_from_tup` so the new tuple-based evaluation path is exercised correctly. | Included in targeted run (`17 passed`) and full suite run (`164 passed`). | 4 | `test_genetic_search_uses_parallel_pool_when_enabled` fails under old map-based stub and now validates the pool-backed branch. |
| `tests/test_shard_mapping.py` | Converted direct `compute_mapping` call setup to deterministic `initial_points.create_pool` monkeypatching with an `imap_unordered` + `__len__` iterator stub; added explicit assumption/failure docstring. | Included in targeted run (`17 passed`) and full suite run (`164 passed`). | 4 | `TestShardMaps.test_compute_mapping` now traps pool API drift that breaks shard aggregation flow. |

Challenge rubric breakdown (pool-migration test updates):
- Failure-path coverage: yes (pool API mismatch conditions are represented by deterministic doubles that would fail on old call patterns).
- Boundary stress: partial (iterator-length/progress boundary is covered; no large-scale multiprocessing stress in unit tests).
- Known-answer / invariant: yes (stable shard counts and deterministic best/closest selections remain asserted).
- Stochastic robustness: not applicable (tests use deterministic samplers/pool doubles).
- Regression trap: yes (tests encode `create_pool` + `imap_unordered` assumptions and fail if code or tests revert to old API wiring).

## Non-Touched Modules (Repository-Wide Summary)

| Area | Status from latest run | Risk / follow-up |
|---|---|---|
| `dreamer/` runtime modules (non-test code) | No code edits in this cycle; repository-wide test + coverage command completed successfully with no collection failures. | Low for this cycle; keep watching multiprocessing integration paths in future refactors. |
| Remaining `tests/` files not edited in this task | Executed as part of full suite and all passed. | Low. |

## Executed Test Evidence

Commands executed in this cycle:

```bash
python -m pytest -q tests/test_extraction_initial_points.py tests/test_search_genetic.py tests/test_shard_mapping.py
python -m pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing
```

Observed outcomes:
- Targeted run: `17 passed`, `1 warning`.
- Full suite + coverage run: `164 passed`, `1 warning`.

## Coverage Command Output Snapshot

Required command status:
- `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` -> **executed successfully**.

Coverage highlights from this run:
- `dreamer/extraction/utils/initial_points.py`: `89%` line coverage (`124` stmts, `10` missed), branch partials present.
- `dreamer/search/methods/genetic.py`: `68%` line coverage (`278` stmts, `73` missed), branch partials present.
- Overall project coverage: `55%` line coverage.

## Notes / Remaining Risks

1. These updates intentionally use deterministic in-process pool doubles for reliability and speed; they validate API wiring, not OS-level process behavior.
2. The `LIReC` SQLAlchemy deprecation warning is still present and unrelated to this task.
3. If deeper multiprocessing integration assurance is required, add an opt-in integration test that runs against a real `create_pool()` under controlled resources.
