# Global Test Report (2026-04-24)

Bottom line: the full discovered suite (`168` tests across `19` test modules) was executed and is now green (`168 passed`, `1 warning`).

## Suite Inventory and Outcome

| Metric | Value |
|---|---:|
| Collected tests | 168 |
| Test modules | 19 |
| Passed | 168 |
| Failed | 0 |
| Warnings | 1 |
| Runtime | 46.60s |

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
| `tests/test_shard.py` | 20 | 20 | 0 | PASS |
| `tests/test_shard_mapping.py` | 1 | 1 | 0 | PASS |
| `tests/test_system_logger_integration.py` | 1 | 1 | 0 | PASS |
| `tests/test_system_priorities_io.py` | 3 | 3 | 0 | PASS |
| `tests/test_tqdm_config.py` | 2 | 2 | 0 | PASS |

## Failure Details (Exact)

- No failing tests in the latest full-suite execution.

## Coverage Snapshot and Policy Alignment (`COVERAGE_POLICY.md`)

Required execution command status:
- `pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing` -> **executed** (via `python -m pytest ...`).

Measured coverage from this run:

| Scope | Line coverage | Branch coverage | Policy target | Status |
|---|---:|---:|---|---|
| Overall project (`dreamer`) | 61.67% (2167/3514) | 36.49% (443/1214) | 80%+ line, 65%+ branch | Below target |
| Critical path: `dreamer/extraction` | 56.36% (642/1139) | 35.49% (159/448) | 85%+ line, 75%+ branch | Below target |
| Critical path: `dreamer/search` | 69.15% (269/389) | 48.08% (75/156) | 85%+ line, 75%+ branch | Below target |

Notable low-coverage files from pytest term-missing output:
- `dreamer/extraction/samplers/chrr_sampler.py`: 0%
- `dreamer/extraction/samplers/raycaster.py`: 8%
- `dreamer/analysis/errors.py`: 0%
- `dreamer/search/methods/hedgehog_scan.py`: 22%

## Warning Summary

- `MovedIn20Warning` from `LIReC/db/models.py` (`declarative_base()` deprecation under SQLAlchemy 2.0); warning is pre-existing and unrelated to current pass/fail status.

## Executed Evidence

Commands run in this audit cycle:

```bash
python -m pytest -q tests/test_extraction_initial_points.py
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=dreamer --cov-branch --cov-report=term-missing
python -m coverage json -o /tmp/global_cov.json
python -m pytest tests/ --collect-only -q
```

Observed outcomes:
- Targeted run (`tests/test_extraction_initial_points.py`): `8 passed`, `1 warning`.
- Full suite run: `168 passed`, `1 warning in 26.98s`.
- Full suite + coverage run: `168 passed`, `1 warning in 46.60s`.
- Collection check: `168` tests discovered in `19` modules.

## Risk Notes / Follow-up

1. Test status is green, but coverage policy thresholds are still unmet globally and in both critical paths (`dreamer/extraction`, `dreamer/search`).
2. Low-coverage extraction/search modules should be prioritized for follow-up tests, especially `raycaster`, `chrr_sampler`, and `hedgehog_scan`.
3. This report reflects the workspace state at execution time.

