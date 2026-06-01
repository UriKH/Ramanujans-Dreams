# SPECIFICATION: CMF Shard Symmetry Reduction (Canonical Teleportation)

## Challenge Definitions
### Context
Each CMF has intrinsic hyperplanes in it which define cells in the space. Each unbounded cell is called a shard and is the mathematical object we care about.  
Specific CMFs, and more concretely CMFs which were generated from the pFq contiguous relations, poses some symmetries we can utilize to reduce the number of shards we care about as some CMFs have astronomically large number of shards.

**Provided:** a set of shards and an interior point (along with the set of hyperplanes).

### Symmetry definition:

Let us denote an interior of a shard by: (x_1, x_2, ... x_p, y_1, y_2, ... , y_q).  
Two shards are considered symmetric if there exists a point in one of them such that applying a permutation of the x_i values and y_i values generates a point which land in the other shard.

### Goal
find the set of shards which encapsulates all shards. Meaning a set of shards which are not symmetric between one another and there is no other shard which is not symmetric to any of them and not in the set.

**Notes** about scale and implementation consideration:  
* The number of shards could be up to several millions.
* If relevant, symmetry application could happen during the shard extraction phase and not just after.


## 1. Architectural Objective & Extensibility
Implement symmetry reduction for high-dimensional CMF shards. 
**CRITICAL CONTEXT:** Previous attempts to enforce symmetry by injecting structural hyperplanes (e.g., $x_1 \ge x_2$) into the solver or ray-generator failed due to "Sampling Bias" and "Tree Disconnection."

**The New Strategy:** "Canonical Teleportation." Algorithms explore the unconstrained mathematical space, but we mathematically "teleport" the discovered interior points into a fundamental domain by applying a symmetry transformation, then filter duplicates using these canonical signatures.

**Why teleportation is a valid orbit invariant (and the one subtlety).**
For $pFq$ the transform is: sort the $x$-block and $y$-block coordinates
*independently* (each is its own $S_p$ / $S_q$ factor — never mix the two
blocks). Two shards are symmetric iff a coordinate permutation maps an
interior point of one onto the other; since sorting is invariant under
permutation, **all** interior points of a whole orbit teleport to the same
canonical point, and hence to the same canonical sign signature. The
symmetry walls $x_i = x_j$ are *not* arrangement hyperplanes, so a cell can
straddle one — but a cell that genuinely straddles $x_i = x_j$ is itself
*invariant* under that swap (it contains a wall point, which the swap
fixes), so its interior points still teleport consistently.

> **CRITICAL — use a strictly-interior point, not an LP vertex.** The
> fingerprint is only sound when computed from a point that is *strictly
> inside every facet* (e.g. the slack-maximising / Chebyshev-centre point).
> A plain feasibility LP (CBC with a zero objective) returns a **vertex**,
> which lies *on* arrangement hyperplanes and frequently on a wall
> $x_i = x_j$; teleporting such a degenerate point gives an inconsistent
> signature that silently **mis-counts orbits** (measured on $pFq(3,1)$:
> 44 instead of the true 50; on $pFq(2,2)$: 102 instead of 100). Always
> fingerprint with the deep-interior point.

This means **no symmetry-group enumeration is needed** — teleportation is
$O(D \log D)$ per point and fully vectorisable, so it scales to large
families ($6F5$ etc.).

**Extensibility Requirement:** Symmetry is specific to the CMF family. The $pFq$ family uses $S_p \times S_q$ (independent sorting of $x$ and $y$ coordinates), but future CMFs will have different symmetries. Therefore, the canonical transformation logic **must not** be hardcoded into the extraction algorithms. It is abstracted behind a `SymmetryStrategy` class (vectorised `apply(points) -> canonical_points`) built by `symmetry_for_cmf(cmf, shift)`; the extractors receive an injected strategy and never reference the CMF family directly. For $pFq$, `BlockSortSymmetry` sorts each block, grouping coordinates by **equal fractional shift** (only coordinates whose shifts share a fractional part may be swapped — otherwise the swap would not map the shifted integer lattice onto itself; mirrors the legacy `initial_points.__same_shift_indices`).

---

## 2. The Heuristic Method (`ray_extractor.py`)
Because the ray extractor is fully vectorized via NumPy, we generate points globally, teleport them all simultaneously via the CMF's symmetry definition, and deduplicate before instantiating heavy Python objects.

### Implementation Steps:
1. Generate the random ray matrix and algebraically compute the raw point intersections as usual (no domain constraints).
2. For the resulting matrix of points `P` (Shape: $N \times D$):
   - Pass `P` to the CMF's canonical transformation method (e.g., `cmf.apply_symmetry(P)`).
   - For $pFq$, this method will internally split `P` into $p$ and $q$ columns, apply `np.sort` descending along `axis=1` to each, and recombine them to return `Canonical_P`.
3. Multiply `Canonical_P` by the hyperplane matrix $A$ to get the Canonical Sign Vectors.
4. Use `np.unique(..., axis=0)` on the sign vectors to instantly filter out symmetric duplicates.
5. Generate `ShardDTO` objects *only* for the unique canonical shards.

### Conceptual Snippet:
```python
# P is the (N x D) matrix of calculated points
P_canonical = cmf.apply_symmetry(P) # CMF dictates its own symmetry

# Get canonical sign vectors and filter
canonical_signs = np.sign(P_canonical @ A.T)
unique_signs, unique_indices = np.unique(canonical_signs, axis=0, return_index=True)

# Proceed with unique_points
unique_points = P_canonical[unique_indices]
```

---

## 3. The Exact Method (`cells.py` / Legacy Track)
The Exact method must dynamically alter its search algorithm based on whether symmetry reduction is active, balancing RAM efficiency with CPU preservation.

### Branch A: Symmetry Reduction is OFF (`IGNORE_DUPLICATE_SEARCHABLES = False`)
- **Action:** Retain the current **Avis-Fukuda Reverse Search**.
- **Reasoning:** It is memoryless and highly efficient for exhaustive enumeration when stateful tracking is not required.

### Branch B: Symmetry Reduction is ON (`IGNORE_DUPLICATE_SEARCHABLES = True`)
- **Action:** Switch to a **Breadth-First Search (BFS)** with dynamic pruning (`cells.iter_cells_canonical`).
- **Reasoning:** Avis-Fukuda relies on a continuous spanning tree and will silently drop valid branches if a symmetric cell is dynamically skipped. BFS safely drops symmetric branches on the fly without breaking the search topology. Completeness (exactly one representative per orbit, none missed) holds because the arrangement is invariant under the group: a group element mapping a pruned cell $r'$ to the kept representative $r$ also maps any neighbour of $r'$ to a neighbour of $r$, so every orbit adjacent to $r$'s orbit is reachable from $r$ alone. Verified against brute force ($pFq(2,1)\to28$, $(3,1)\to50$, $(2,2)\to100$). Memory is $O(\#\text{orbits})$ — the whole point, since $\#\text{orbits}\ll\#\text{cells}$.

### BFS Dynamic Pruning Logic:
1. **Initialize:** Maintain a `canonical_seen = set()`.
2. **Step:** The BFS flips a sign and proposes a new cell boundary.
3. **Interior point:** Run the **slack-maximising LP** (no integer constraints) to test feasibility *and* obtain a **strictly-interior (deep) point**. A bare feasibility LP returning a vertex is **not** acceptable here — see the CRITICAL note in §1.
4. **Teleport:** If feasible, pass the deep-interior point to the injected `symmetry.apply` (e.g. block-sort for $pFq$).
5. **Fingerprint:** Multiply this canonical point by $A$ (add $c$) and take the sign vector. This is the `Canonical Signature`.
6. **Prune or Proceed:**
   - If `Canonical Signature` is IN `canonical_seen`: Discard the cell. **Do not queue its neighbors.**
   - If `Canonical Signature` is NOT IN `canonical_seen`: Add it to the set, emit the cell (its *actual* sign vector is the representative), queue the cell's neighbors, and proceed to the unbounded-classification + integer-point MILP as in Branch A.

### Conceptual Snippet (BFS Worker Loop):
```python
if symmetry is not None:
    interior_pt = solve_max_slack_lp(cell_sign_vector)   # deep interior, NOT a vertex
    if interior_pt is not None:                          # feasible cell
        canonical_pt = symmetry.apply(interior_pt[None, :])[0]
        canonical_sign = tuple(np.sign(A @ canonical_pt + c).astype(int))

        if canonical_sign in canonical_seen:
            continue                                     # prune this orbit branch
        canonical_seen.add(canonical_sign)
        queue.extend(get_neighbors(cell_sign_vector))
        # emit cell_sign_vector; classify unbounded; MILP integer point...
```

## 4. Final Checklist for the Agent
- [x] Symmetry logic abstracted behind `SymmetryStrategy` (`v2/symmetry.py`), built by `symmetry_for_cmf` — not hardcoded in the extractors (extensibility requirement).
- [x] `ray_extractor.py` teleports witnesses (`symmetry.apply`) before signing, deduping by canonical signature. Witnesses are generic integer interior points, so this is exact (verified ratio 1.000 through $pFq(4,3)$).
- [x] `cells.py` routes to the canonical BFS (`iter_cells_canonical`) when a symmetry is set, Avis-Fukuda (`iter_cells`) otherwise.
- [x] BFS canonical pruning happens *after* the slack-maximising LP (feasibility + deep-interior point), but *before* heavy MILP integer extraction.
- [x] **Exact** uses a strictly-interior (slack-maximising) point for the fingerprint — never an LP vertex (the original bug; see §1 CRITICAL).