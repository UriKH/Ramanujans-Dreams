# Test Audit Report (2026-04-18)

Bottom line: extraction mapping test doubles were aligned with the current packed-task worker adaptor contract; targeted failing tests now pass (`9 passed`), and the repository-wide coverage command is green again (`167 passed`).

## Touched Modules (Detailed Review)

| Touched module | Changes made | Coverage evidence | Challenge rubric (/5) | Regression evidence |
|---|---|---|---:|---|
| `tests/test_extraction_initial_points.py` | Updated `_DummyPool.imap_unordered` to pass each packed task as a single argument to the mapped callable, matching `compute_mapping`'s `__worker_wrapper_adaptor(filter_func, args)` contract. | Included in targeted run (`9 passed`) and full suite (`167 passed`). | 4 | `test_compute_mapping_selects_closest_point_per_signature` and `test_compute_mapping_tie_breaks_lexicographically` now validate behavior through the real adaptor path and would fail if tuple unpacking is reintroduced. |
| `tests/test_shard_mapping.py` | Updated `_DummyPool.imap_unordered` to use `func(task)` instead of tuple-unpacked invocation, preserving deterministic aggregation while matching production pool semantics. | Included in targeted run (`9 passed`) and full suite (`167 passed`). | 4 | `TestShardMaps.test_compute_mapping` now guards against API drift between pool stubs and packed-task adaptor usage. |

Challenge rubric breakdown (this task):
- Failure-path coverage: yes (stub/adapter signature mismatch is directly trapped).
- Boundary stress: partial (covers minimal deterministic task batches and merge path).
- Known-answer / invariant: yes (nearest-point and deterministic tie-break invariants remain asserted).
- Stochastic robustness: not applicable (deterministic test doubles).
- Regression trap: yes (tests fail if pool adaptor call shape drifts again).

## Non-Touched Modules (Repository-Wide Summary)

| Area | Status from latest run | Risk / follow-up |
|---|---|---|
| `dreamer/extraction/utils/initial_points.py` and dependent tests | Previously failing pool-adapter tests are now green after aligning test doubles with packed-task invocation. | Low for this cycle; keep test doubles in sync if adapter signature changes. |
| Remaining non-touched runtime modules | Full repository test and coverage run passed with no new failures. | Low. |

## Executed Test Evidence

Commands executed in this cycle:

```bash
python -m pytest -q tests/test_extraction_initial_points.py tests/test_shard_mapping.py
python -m pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing
```

Observed outcomes:
- Targeted run: `9 passed`, `1 warning`.
- Full suite + coverage run: `167 passed`, `1 warning`.

## Coverage Command Output Snapshot

Required command status:
- `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` -> **executed successfully**.

Coverage highlights from this run:
- `dreamer/extraction/utils/initial_points.py`: `89%` line coverage (`126` stmts, `10` missed), branch partials present.
- `dreamer/utils/logger.py`: `78%` line coverage (`237` stmts, `42` missed), branch partials present.
- Overall project coverage: `55%` line coverage.

## Notes / Remaining Risks

1. The workspace is currently in a large pre-existing dirty state; this report only evaluates files touched for this task.
2. The LIReC SQLAlchemy deprecation warning persists and is unrelated.
