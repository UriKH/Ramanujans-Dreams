# Context: Ramanujan's Dreams - Shard Extraction Module (`extraction.py`)

You are acting as an expert Software Engineer and Computational Geometer. We are building the extraction module for a pipeline that explores high-dimensional Conservative Matrix Fields (CMFs). 

## 1. The Mathematical Problem
**Inputs:** * A set of $1 \le N \le 100$ hyperplanes in a space of $3 \le D \le 15$ dimensions. 
* The hyperplanes have integer coefficients and are highly sparse.
* Because of the sparsity and integer coefficients, the geometric space is **highly degenerate** (many hyperplanes intersect at the exact same lower-dimensional boundaries).

**Goal:** Find exactly one point with **strictly integer coordinates** inside every **unbounded cell** (shard) formed by this hyperplane arrangement. Bounded cells should be ignored.

## 2. The Chosen Architecture: Strategy Pattern
Because high-dimensional exact exact computational geometry is prone to exponential time complexity, we are implementing a **Strategy Pattern** via an `ExtractionManager` class. The user will pass a `strategy` argument (`"auto"`, `"exact"`, or `"heuristic"`).

Please implement the following architecture:

### Strategy A: The `LrslibExtractor` (Exact Method)
Standard libraries like `cddlib` crash on our degenerate data. We must use **Lexicographic Reverse Search (`lrslib`)**. 
* **Crucial Constraint:** Do NOT use Python C-API wrappers (like `pyrs`) to avoid C-compiler dependency issues for our users. 
* **Implementation:** 1. Write a Python `subprocess` wrapper. 
    2. The python code should write the hyperplanes' H-representation (inequalities) to a temporary text file.
    3. Call a standalone `lrs` binary via CLI to compute the V-representation.
    4. Parse the `lrs` text output. Filter the cells: if a cell contains at least one **ray** in the output, it is unbounded. If it only contains vertices, discard it.
    5. Take the bounding inequalities of the unbounded cells and pass them to an MILP solver (e.g., `PySCIPOpt` or `scipy.optimize.milp`).
    6. Set the MILP objective to $0$ (feasibility only) and constrain all variables to be integers to extract the specific coordinate.

### Strategy B: The `RayShootingExtractor` (Heuristic Method)
Bypass exact enumeration entirely to generate a fast, partial sample.
* **Implementation:** 1. Generate thousands of rays originating from the origin, moving outward in integer directions.
    2. Track which hyperplanes the ray crosses. Once the ray crosses its final hyperplane and escapes to infinity, the cell is unbounded.
    3. Scale the ray's vector until it hits a valid integer coordinate satisfying the cell's inequalities.

### Strategy C: The `"auto"` mode (The Fallback)
* **Implementation:** 1. Attempt to run the `LrslibExtractor`.
    2. Wrap the execution in a strict timeout mechanism (e.g., 1 hour maximum).
    3. If the exact enumeration times out (due to an astronomically complex 15D space), catch the exception, log a warning, and automatically execute the `RayShootingExtractor` to ensure the pipeline never freezes.

## 3. Requested Deliverables
Please write the Python code for `extraction.py` including:
1. The abstract base class `BaseExtractor`.
2. The `ExtractionManager` that handles the strategy routing and the timeout logic for `"auto"`.
3. The skeleton for `LrslibExtractor`, focusing heavily on the `subprocess` file I/O handling (using `tempfile` to avoid disk clutter) and the regex/parsing logic expected from an `lrs` output.
4. The skeleton for `RayShootingExtractor`.
5. Ensure the code is strictly typed and documented.

Please provide the code structure.