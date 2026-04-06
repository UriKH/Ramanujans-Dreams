# Code Review Checklist

Use this checklist before declaring any code development task **done**.

## Correctness

- [ ] All tests pass (`pytest tests/ -v`).
- [ ] New public functions have tests (see `COVERAGE_POLICY.md`).
- [ ] Mathematical results verified numerically to 100+ digits.
- [ ] Edge cases handled: $n=0$, $n=1$, empty input, single-element input.

## Guardrails

- [ ] Public API inputs are validated (types, ranges, dimensions).
- [ ] No use of Python `float` for mathematical computation — `mpmath.mpf` or `sympy.Rational` only.
- [ ] `mpmath.mp.dps` set explicitly and high enough (≥ 2× needed digits).
- [ ] Assertions on mathematical invariants (determinant, symmetry, recurrence order).

## Sanity Checks

- [ ] Known-answer test: computes a well-known constant and matches reference.
- [ ] Roundtrip test: forward + inverse operations compose to identity (where applicable).
- [ ] Scale test: tested at realistic $N$, not just toy inputs.
- [ ] Cross-validated against `ramanujantools` output (where applicable).

## Code Quality

- [ ] Docstring on every new public function.
- [ ] No commented-out code or debug prints left behind.
- [ ] Consistent naming conventions with existing codebase.
- [ ] No unnecessary dependencies added.

## Performance (if applicable)

- [ ] Profiled — actual bottleneck identified before optimizing.
- [ ] Before/after benchmarks included with wall-clock times.
- [ ] Tested at intended operational scale.
- [ ] No regression in existing benchmarks (>20% slowdown needs justification).

## Security

- [ ] No hardcoded credentials or tokens.
- [ ] No `eval()` or `exec()` on untrusted input.
- [ ] File paths sanitized if user-provided.
