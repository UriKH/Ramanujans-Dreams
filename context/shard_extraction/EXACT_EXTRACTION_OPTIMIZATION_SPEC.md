# Context: Ramanujan's Dreams - Exact Method Optimization (`cells.py`)

You are acting as an expert Software Engineer and Computational Geometer. We are optimizing the exact shard extraction module (`cells.py`) for a pipeline that explores high-dimensional Conservative Matrix Fields (CMFs). 

## 1. The Bottlenecks and Bugs (The Problem)
1. **The Performance Bottleneck:** Currently, `cells.py` uses `scipy.optimize.linprog`. Building the SciPy LP matrix from scratch tens of thousands of times during a BFS walk causes massive overhead and 5-minute timeouts.
2. **The Mathematical Bug:** The current `make_unbounded_checker` uses a strict interior ray formulation (`b_ub=-1e-5`). This wrongly flags infinite "tubes" or "slabs" (where recession rays are parallel to boundaries) as bounded, artificially destroying over 90% of our unbounded shards.

## 2. The Solution: "Hot-Started" Stateful Solvers
We will replace `scipy` with a stateful solver library like `python-mip` (CBC backend). We will build two static LP models in memory **exactly once**. We will navigate the geometry purely by flipping the bounds of the variables and calling `model.optimize()`.

### Solver A: The Feasibility Checker (The BFS Walk)
This determines if a sign-vector geometrically exists with a non-zero volume.
* **Initialization (Static):**
  * Define continuous variables $\vec{x} \in \mathbb{R}^D$ (unbounded).
  * Define continuous variables $\vec{y} \in \mathbb{R}^N$ (auxiliary).
  * Add $N$ static constraints: $\vec{A}_i \cdot \vec{x} + c_i - y_i = 0$.
* **Bound Swapping (Dynamic):**
  * When checking a cell, if $s_i = +1$, set the bounds of $y_i$ to $[10^{-5}, \infty)$.
  * If $s_i = -1$, set the bounds of $y_i$ to $(-\infty, -10^{-5}]$.
  * Call `optimize()`. If FEASIBLE, the cell exists.
  * *CRITICAL:* Revert the bounds of $y_i$ before checking the next neighbor!

### Solver B: The Unbounded Checker (Stiemke's Theorem)
This perfectly checks unboundedness using the Dual Space, bypassing the "Tube Paradox" bug entirely.
* **Initialization (Static):**
  * Define continuous variables $\vec{w} \in \mathbb{R}^N$.
  * Add $D$ static constraints forcing the weighted sum of the normals to 0: 
    $\sum_{i=1}^N \vec{A}_{i, d} \cdot w_i = 0$ (for each dimension $d$).
* **Bound Swapping (Dynamic):**
  * If $s_i = +1$, set the bounds of $w_i$ to $[1, \infty)$.
  * If $s_i = -1$, set the bounds of $w_i$ to $(-\infty, -1]$.
  * Call `optimize()`. 
  * If FEASIBLE $\rightarrow$ The cell is BOUNDED.
  * If INFEASIBLE $\rightarrow$ The cell is UNBOUNDED.

## 3. Phase 2: Avis-Fukuda Reverse Search (Future Direction)
While the Stateful Solver will massively speed up the BFS, it still relies on a `seen` set in memory. If extreme dimensions ($11 \le D \le 15$) cause memory exhaustion, we must eventually migrate to **Memoryless Reverse Search (Avis-Fukuda)**. 
* By defining a strict min-index objective function, every valid cell gets exactly one "parent", forming a spanning tree.
* This requires zero memory (no `seen` set) and allows perfect linear parallelization via `multiprocessing.Pool` across CPU cores. Keep the current code structure modular so this generator can be easily swapped in later.

## 4. Requested Deliverables
Please rewrite `cells.py` to implement **Solver A and Solver B** using `python-mip` (or `PySCIPOpt` if configured).
1. Initialize the stateful models inside the `enumerate_cells` loop (or a wrapper class).
2. Set `model.verbose = 0` to suppress console spam.
3. Replace the `_interior_slack` function with Solver A's bound-swapping logic.
4. Replace `make_unbounded_checker` with Solver B's bound-swapping logic.
5. Return the accurately filtered list of sign-vectors.