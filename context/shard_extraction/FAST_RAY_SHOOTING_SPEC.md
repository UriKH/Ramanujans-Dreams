# Context: Ramanujan's Dreams - Fast Algebraic Ray-Shooting Extractor

You are acting as an expert Software Engineer and Computational Geometer. We are upgrading the heuristic shard extraction module (`ray_extractor.py`) for a pipeline that explores high-dimensional Conservative Matrix Fields (CMFs).

## 1. The Bottleneck and The Goal
**The Problem:** In dimensions $D \ge 7$, the current extraction pipeline is too slow. The bottleneck is the "setup and teardown" overhead of calling `scipy.optimize` solvers iteratively for thousands of shards. 
**The Goal:** We need to completely rewrite the `RayShootingExtractor` to use a **100% solver-free, purely algebraic, vectorized NumPy approach**. There should be no while-loops for scaling, no rounding of real coordinates, and zero calls to `scipy`.

## 2. The Algebraic Ray-Shooting Strategy
Instead of stepping along a ray iteratively, we can calculate the exact integer coordinate inside an unbounded cell algebraically. 

Given $N$ hyperplanes defined by $A \mathbf{x} + c = 0$ (where $A$ is an $N \times D$ integer matrix and $c$ is an $N$-length integer vector):

1. **The Integer Ray:** Generate a matrix $V$ of random integer direction vectors (rays originating from the origin). Because $\vec{v}$ is made of integers, any scalar multiple $t\vec{v}$ where $t$ is an integer is guaranteed to be a perfect integer coordinate.
2. **The Intersection Time:** A ray $\vec{v}$ crosses hyperplane $i$ at a continuous scalar "time" $t_i$: 
   $$t_i = \frac{-c_i}{\vec{A}_i \cdot \vec{v}}$$
3. **The Escape Boundary:** The ray enters its final, unbounded cell only after crossing the very last hyperplane. The "escape" scalar is:
   $$t_{escape} = \max(t_1, t_2, \dots, t_N)$$
4. **The Integer Point:** To guarantee we are strictly inside the unbounded shard and have perfect integer coordinates, we floor the maximum time and add a buffer (e.g., $+1$):
   $$t_{final} = \lfloor t_{escape} \rfloor + 1$$
   The final integer point inside the unbounded cell is simply $t_{final}\vec{v}$.

## 3. Implementation Instructions (Vectorized NumPy)
Please implement this math inside the `RayShootingExtractor.extract` method (or a helper method called by it). Use the following highly optimized matrix operations:

* **Matrix Shapes:** * Let $V$ be the ray matrix of shape `(num_rays, D)`.
  * Let $A$ be the hyperplane normals of shape `(N, D)`.
  * Let $c$ be the offsets of shape `(N,)`.
* **Dot Products:** Calculate the dot products for all rays and all hyperplanes simultaneously: `M = V @ A.T` (Shape: `(num_rays, N)`).
* **Safe Division:** Calculate the crossing times `T = -c / M`. 
  * **CRITICAL EDGE CASE:** If a ray is parallel to a hyperplane, the dot product in `M` is exactly `0`. You MUST use `np.divide(..., out=np.zeros_like(...), where=M!=0)` to avoid Divide-by-Zero errors. Parallel rays never cross the hyperplane, so a crossing time of `0` or `-inf` is safely ignored when taking the maximum.
* **Filter Degeneracy:** If an entire ray lies exactly on a hyperplane (`M == 0` AND `c == 0`), discard that ray.
* **Extraction:** Compute `t_escape = np.max(T, axis=1)`. Then `t_final = np.floor(t_escape) + 1`.
* **Final Points:** The valid integer points are `Points = t_final[:, np.newaxis] * V`.
* **Sign Vector Mapping:** For each unique final point, multiply `Points @ A.T + c` to find its $+1/-1$ sign vector representation. Return a dictionary mapping the unique `SignTuple` to its corresponding integer `np.ndarray` point.

## 4. Requested Deliverables
Please rewrite the `RayShootingExtractor` class inside `ray_extractor.py` to utilize this exact vectorized mathematical strategy. 
* Keep the existing base class signatures (`BaseExtractor`, `ShardMapping`, etc.).
* Ensure the code is strictly typed and heavily commented.