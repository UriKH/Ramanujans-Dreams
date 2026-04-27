# Global Test Report (2026-04-24)

Bottom line: after adding DataManager searchable-space context and JSON export/import support for Shards, the full discovered suite is green (`173 passed`, `1 warning`) across `20` test modules.

## Suite Inventory and Outcome

| Metric | Value |
|---|---:|
| Collected tests | 173 |
| Test modules | 20 |
| Passed | 173 |
| Failed | 0 |
| Warnings | 1 |
| Runtime (full+coverage) | 61.73s |

## Test Module Status (All Collected Modules)

| Test module | Collected | Passed | Failed | Status |
|---|---:|---:|---:|---|
| `tests/test_constant.py` | 18 | 18 | 0 | PASS |
| `tests/test_db_and_config.py` | 9 | 9 | 0 | PASS |
| `tests/test_extraction_conditioner.py` | 5 | 5 | 0 | PASS |
| `tests/test_extraction_initial_points.py` | 8 | 8 | 0 | PASS |
| `tests/test_extraction_sampler_pipeline.py` | 9 | 9 | 0 | PASS |
| `tests/test_extractor.py` | 3 | 3 | 0 | PASS |
| `tests/test_extractor_mod.py` | 1 | 1 | 0 | PASS |
| `tests/test_formatters.py` | 20 | 20 | 0 | PASS |
| `tests/test_hyperplanes.py` | 31 | 31 | 0 | PASS |
| `tests/test_logger.py` | 11 | 11 | 0 | PASS |
| `tests/test_sampler_shard_sampler.py` | 4 | 4 | 0 | PASS |
| `tests/test_sampler_sphere.py` | 6 | 6 | 0 | PASS |
| `tests/test_sanity.py` | 8 | 8 | 0 | PASS |
| `tests/test_search_genetic.py` | 8 | 8 | 0 | PASS |
| `tests/test_shard.py` | 21 | 21 | 0 | PASS |
| `tests/test_shard_mapping.py` | 1 | 1 | 0 | PASS |
| `tests/test_storage_objects.py` | 2 | 2 | 0 | PASS |
| `tests/test_system_logger_integration.py` | 1 | 1 | 0 | PASS |
| `tests/test_system_priorities_io.py` | 5 | 5 | 0 | PASS |
| `tests/test_tqdm_config.py` | 2 | 2 | 0 | PASS |

## Failure Details (Exact)

- No failing tests in the latest full-suite execution.

## Coverage Snapshot and Policy Alignment (`COVERAGE_POLICY.md`)

Required execution command status:
- `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` -> **executed** (via `python -m pytest ...`).

Measured coverage from this run:

| Scope | Line coverage | Branch coverage | Policy target | Status |
|---|---:|---:|---|---|
| Overall project (`dreamer`) | 62.55% (2245/3589) | 37.40% (466/1246) | 80%+ line, 65%+ branch | Below target |
| Critical path: `dreamer/extraction` | 56.75% (656/1156) | 35.62% (161/452) | 85%+ line, 75%+ branch | Below target |
| Critical path: `dreamer/search` | 69.15% (269/389) | 48.08% (75/156) | 85%+ line, 75%+ branch | Below target |

Notable uncovered/low-coverage cases from term-missing output:
- `dreamer/extraction/samplers/chrr_sampler.py`: 0%
- `dreamer/extraction/samplers/raycaster.py`: 8%
- `dreamer/analysis/errors.py`: 0%
- `dreamer/search/methods/hedgehog_scan.py`: 22%

## Warning Summary

- `MovedIn20Warning` from `LIReC/db/models.py` (`declarative_base()` deprecation under SQLAlchemy 2.0); warning is pre-existing and unrelated to current pass/fail status.

## Executed Evidence

Commands run in this audit cycle:

```bash
python -m pytest -q tests/test_storage_objects.py tests/test_shard.py tests/test_system_priorities_io.py tests/test_search_genetic.py
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing
python -m pytest tests/ --collect-only -q
python -m coverage json -o /tmp/global_cov.json
```

Observed outcomes:
- Targeted regression run: `36 passed`, `1 warning`.
- Full suite run: `173 passed`, `1 warning in 33.41s`.
- Full suite + coverage run: `173 passed`, `1 warning in 61.73s`.
- Collection check: `173` tests discovered in `20` modules.

## Risk Notes / Follow-up

1. Runtime/test behavior is green; remaining risk is coverage debt against project and critical-path thresholds.
2. Highest-priority follow-up targets are still `raycaster`, `chrr_sampler`, and `hedgehog_scan`.
3. This report reflects the workspace state at execution time.

