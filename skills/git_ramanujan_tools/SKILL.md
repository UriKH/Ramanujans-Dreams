---
name: git_ramanujan_tools
description: "Instructions for navigating the RamanujanMachine repositories, CMF structures, and integrating with existing tools. USE FOR: using ramanujantools library, understanding CMF/PCF/LinearRecurrence APIs, navigating the RamanujanMachine GitHub organization repos, integrating with existing discovery algorithms."
---
# Ramanujan Tools

The Ramanujan Machine project has multiple repositories under `https://github.com/RamanujanMachine/`:

| Repository | Purpose |
|------------|---------|
| `ramanujantools` | **Primary library.** Symbolic and numeric tools: CMF, PCF, LinearRecurrence, asymptotics, solvers. Install via `pip install ramanujantools`. |
| `RamanujanMachine` | Original discovery algorithms: MITM-RF, ESMA, polynomial domain enumeration. |
| `euler2ai` | Formula harvesting from arXiv + unification via coboundary equivalences. |
| `ASyMOB` | Algebraic Symbolic Mathematical Operations Benchmark. |
| `LIReC` | Library of Integer Relations and Constants. |
| `Blind-Delta-Algorithm` | Alternative search algorithm. |

## Rules of Repository Interaction
1. **Never update or commit to a remote repository without EXPLICIT permission.** You may clone, read, and pull locally for experiments.
2. Use the `ramanujantools` library as the primary tool for CMF/PCF/recurrence work before writing ad-hoc scripts.
3. When building new evaluation methods, validate them against the existing library's output on shared benchmarks.

## `ramanujantools` Library Architecture

### Package Structure
```
ramanujantools/
├── cmf/           # Conservative Matrix Fields (multi-dimensional)
│   ├── cmf.py          # Core CMF class
│   ├── known_cmfs.py   # Pre-defined CMFs for e, π, ζ(3), hypergeometric, etc.
│   ├── ffbar.py         # FFbar CMF construction
│   ├── d_finite.py      # D-Finite base class
│   ├── pfq.py           # Hypergeometric pFq → CMF
│   └── meijer_g.py      # Meijer G-function → CMF
├── pcf/            # Polynomial Continued Fractions
├── asymptotics/    # Growth rate and convergence analysis
├── solvers/        # Equation solvers
├── flint_core/     # Fast numeric/symbolic matrix operations (via FLINT)
├── utils/          # Utilities, batching
├── matrix.py       # Symbolic matrix with polynomial entries
├── linear_recurrence.py  # P-recursive sequence class
├── position.py     # Multi-dimensional position/coordinate class
├── limit.py        # Limit computation from matrix walks
└── generic_polynomial.py # Generic polynomial utilities
```

### Core Classes

**`CMF`** — Conservative Matrix Field (multi-dimensional):
```python
from ramanujantools.cmf import CMF
from ramanujantools.cmf.known_cmfs import e, pi, zeta3, hypergeometric_derived_2F1

cmf = e()  # 2D CMF for e
cmf.axes()  # {x, y}
cmf.dim()   # 2
cmf.rank()  # 2 (matrix size)

# Walk along a trajectory
cmf.walk({x: 1, y: 0}, [100, 200, 500], {x: 0, y: 0})

# Compute limit (convergent value)
cmf.limit({x: 1, y: 0}, [100, 200], {x: 0, y: 0})

# Irrationality measure
cmf.delta({x: 1, y: 0}, 500, {x: 0, y: 0})

# Higher-dimensional CMF
cmf3d = hypergeometric_derived_2F1()  # 3D: axes a, b, c
cmf3d.dim()  # 3
```

**`PCF`** — Polynomial Continued Fraction:
- Created from polynomial $a_n, b_n$ or extracted from a CMF trajectory.

**`LinearRecurrence`** — P-recursive sequence:
- Represents sequences satisfying polynomial-coefficient linear recurrences.

**`Matrix`** — Symbolic matrix with SymPy entries:
- Supports substitution, coboundary transforms, factoring, inversion.

**`Position`** — Multi-dimensional coordinate:
- Used for CMF starting points, trajectories, and substitutions.

### Discovery Algorithms (Original `RamanujanMachine` Repo)

| Algorithm | Class | Strategy |
|-----------|-------|----------|
| MITM-RF | `EfficientGCFEnumerator` | Enumerate polynomial coefficient space, hash GCF values at 10-digit precision, refine matches to 100+ digits at depth 1000 |
| ESMA | `SignedRcfEnumeration` | Generate sign-change patterns, extract minimal LFSR via Berlekamp-Massey, filter by complexity |

These are two specific algorithms among potentially many. They're useful reference implementations but the algorithmic space is much broader.

### Key Utility Classes (Original Repo)
- **`MobiusTransform`** (`ramanujan/utils/mobius.py`): 2×2 matrix representation of $f(x) = (ax+b)/(cx+d)$. Supports composition via multiplication, reciprocal, inverse, normalization via GCD.
- **`GeneralizedContinuedFraction`** (`ramanujan/utils/mobius.py`): Accumulates a chain of Möbius transforms. Key method: `evaluate()` returns the accumulated convergent as `mpf`.
- **`LHSHashTable`** (`ramanujan/LHSHashTable.py`): Two-tier lookup (Bloom filter → pickled dict) storing rational expressions for LHS matching.
- **`CachedSeries`** (`ramanujan/CachedSeries.py`): Lazy-caching polynomial series evaluation.
- **`AbstractPolyDomains`** (`ramanujan/poly_domains/`): Define the polynomial coefficient search space. Custom domains exist for Catalan, Zeta(3), Zeta(5), Zeta(7).

### Configuration Constants (`ramanujan/constants.py`)
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `g_N_initial_search_terms` | 32 | Phase 1 GCF depth |
| `g_N_verify_terms` | 1000 | Refinement depth |
| `g_N_initial_key_length` | 10 | Digits matched in phase 1 |
| `g_N_verify_compare_length` | 100 | Digits verified in phase 2 |
| `g_N_verify_dps` | 2000 | mpmath decimal places for verification |
