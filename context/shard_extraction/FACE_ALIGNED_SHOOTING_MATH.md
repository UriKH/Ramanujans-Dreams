# Face-Aligned Ray Shooting — Mathematical Background

> **Audience.** This document assumes you understand what shards and
> hyperplanes are in the context of this project, but assumes **no** prior
> background in computational geometry.  All concepts are built from scratch.

---

## 1. What are we actually looking for?

Recall the setup.  A CMF defines a collection of `N` hyperplanes in `R^D`
(D-dimensional real space).  Each hyperplane `i` is a flat, (D-1)-dimensional
surface defined by a linear equation:

```
A_i · x + c_i = 0
```

where `A_i` is an integer vector (the "normal" to the surface) and `c_i` is an
integer offset.  These `N` hyperplanes cut `R^D` into open regions called
**cells**.  A **shard** is a cell that extends to infinity (an *unbounded* cell).
We need one integer point inside each shard to use as a starting point for the
trajectory walk.

Every cell is labelled by a **sign vector** `s ∈ {-1, +1}^N`:

```
s_i = +1  means x is on the positive side of hyperplane i  (A_i · x + c_i > 0)
s_i = -1  means x is on the negative side of hyperplane i  (A_i · x + c_i < 0)
```

---

## 2. Generic origin ray shooting — what it does and why it misses some shards

The main heuristic shoots a random **integer direction** `v` from the origin.
As you travel along the ray `t · v` (for increasing scalar `t`), you cross
hyperplanes at specific times:

```
t_i = -c_i / (A_i · v)        (when A_i · v ≠ 0)
```

After the last crossing `t_escape = max_i t_i`, you are in a cell whose sign
vector is fixed for all `t > t_escape`.  The witness point is the first integer
step past that:

```
witness = (floor(t_escape) + 1) · v
```

This always lands in an **unbounded** cell, because the origin-ray construction
guarantees it goes "outward" past all hyperplanes.

**Why does this miss some shards?**  The key is a concept called the
**recession cone** of a cell.

---

## 3. Recession cones — what makes a cell unbounded?

A cell is unbounded if you can walk in some direction `d` forever without
leaving it.  The set of all such directions is called the **recession cone**:

```
K(s) = { d ∈ R^D : s_i · (A_i · d) ≥ 0 for all i }
```

Think of it this way: if you stand inside a cell and walk in direction `d`, you
never cross hyperplane `i` as long as your movement is "compatible" with the
side you're already on.  The recession cone collects all such safe directions.

Two very different shapes of recession cone can occur:

### Case A: Full-dimensional recession cone

The cone is a solid wedge in `R^D` — it has non-zero volume on the unit sphere.
If you shoot a random ray from the origin, there is a *positive probability* of
hitting a direction that lands in this cell.  These cells are what generic origin
shooting finds.

```
Example in 2D: the quadrant { x > 0, y > 0 }.  The recession cone contains all
directions (v_x, v_y) with v_x > 0 and v_y > 0 — a quarter of all directions.
A random ray hits it 25% of the time.
```

### Case B: Lower-dimensional recession cone (tubes and slabs)

The cone is a flat subspace — a line, a plane, etc.  Its volume on the unit
sphere is **zero**.  A random ray from the origin has probability **exactly 0**
of hitting it, regardless of how many rays you shoot.

```
Example in 2D: the vertical strip { 0 < x < 1 }.  The only directions you can
walk forever without leaving are straight up (0, +1) and straight down (0, -1).
That is a one-dimensional subspace — a line — inside a two-dimensional space.
A random direction is almost surely not (0, ±1), so random rays never land here.
```

These cells are **structurally unreachable** by origin ray shooting.  They are
the "tube" and "slab" cells that P2 is designed to find.

---

## 4. What is a "nullspace direction"?

Before explaining P2, we need one concept from linear algebra: the **nullspace**
of a matrix.

Given a set `S` of hyperplanes, stack their normal vectors into a matrix `A_S`
(one row per hyperplane in S).  The **nullspace** of `A_S` is the set of all
vectors `v` such that:

```
A_S · v = 0        (equivalently, A_i · v = 0 for every i in S)
```

In words: `v` is perpendicular to *every* normal in S, so if you travel along
`v` you move *parallel* to every hyperplane in S simultaneously.  You never
cross any of the S hyperplanes no matter how far you go.

### Why does this matter for unbounded cells?

If a cell's recession cone is the nullspace of some subset of hyperplanes, then
a direction `v` in that nullspace is a valid **recession direction** for that
cell.  A ray shot along `v` moves parallel to the S hyperplanes, but still
crosses the remaining hyperplanes — so it eventually escapes them all and lands
in a definite cell.

---

## 5. The P2 algorithm, step by step

Face-aligned shooting exploits the nullspace observation to reach the
structurally-missed cells.

### Step 1: Sample a random subset of hyperplanes

Pick a random subset `S` of the `N` hyperplanes (subset size chosen uniformly
from 1 to D-1).

### Step 2: Compute the integer nullspace of A_S

Find all directions `v` such that `A_i · v = 0` for every `i ∈ S`.  This is
a standard linear algebra computation (we use the exact sympy solver to get
integer directions rather than floating-point ones, which avoids rounding errors
when we later compute sign vectors).

The nullspace has dimension `D - rank(A_S)`.  For a subset of k independent
hyperplanes in D dimensions, the nullspace has dimension `D - k`.

### Step 3: Shoot from random offsets along each nullspace direction

For each direction `v` in the nullspace, shoot a ray from several random integer
**start points** `p` (not from the origin):

```
witness = p + t_final · v
where t_final = floor(max_i { -(A_i·p + c_i) / (A_i·v) for i ∉ S }) + 1
```

The formula is the same escape-time calculation as the generic method, but:
- The ray starts at `p`, not the origin
- Hyperplanes in S have `A_i · v = 0` (they are parallel to the ray), so they
  don't contribute a crossing time

### Step 4: Collect the witness and its cell

Compute the sign vector of the witness (which side of each hyperplane is it on?)
and add it to the shard map if it's new.

---

## 6. The crucial insight: why sweeping offsets finds new cells

When you shoot from offset `p` along `v` (where `v` is in the nullspace of
`A_S`):

- For hyperplanes **not** in S: `A_i · v ≠ 0`, so the ray crosses them.  The
  sign of the witness on those hyperplanes is determined by the **direction** `v`
  alone — it doesn't change with `p`.

- For hyperplanes **in** S: `A_i · v = 0`, so the ray never crosses them.  The
  sign of the witness on those hyperplanes is determined by the **starting
  point** `p`:

  ```
  sign_i = sign(A_i · p + c_i)    for i ∈ S
  ```

  This changes when you move `p` across those hyperplanes.

So by fixing `v` and sweeping many different `p`, you enumerate all the
different cells that share the recession direction `v` — the "tube" cells that
are distinguished only by which side of the S hyperplanes they lie on.  These
are exactly the cells whose recession cone is contained in (or aligned with) the
nullspace of `A_S`.

A concrete picture in 2D:

```
     y
     |
     |          The strip { 0 < x < 3 } has recession direction v = (0, 1)
   3 |..........  (the y-axis direction, nullspace of the hyperplane x = 0 and x = 3).
     |          
     |   STRIP   A ray from the origin (0,0) along (0,1):
   0 |..........    never crosses x=0 or x=3 (parallel)
     |              starts at x=0, which is ON the hyperplane → invalid
     |          
     +----------> x  Generic rays from origin: almost never have x=0, so they land
     0         3       in x > 3 or x < 0 — never in 0 < x < 3.
                  
                  Face-aligned: pick S = {hyperplane x=0}, nullspace v = (0,1).
                  Start at p = (1, 0) (inside the strip in x).
                  Shoot along (0, 1): the witness lands in the strip.  ✓
```

---

## 7. Why the probability argument works

For a generic CMF arrangement, the cells with lower-dimensional recession cones
form a **measure-zero** subset of directions on the unit sphere.  That means:

- Any specific random direction hits them with probability 0 (no matter how many
  rays you shoot from the origin)
- But they are still real, unbounded, integer-containing cells that the pipeline
  needs

Face-aligned shooting is not smarter about *which* cell to target — it is
smarter about *how it generates candidate directions*.  By aligning the ray with
a hyperplane subset's nullspace, it concentrates probability mass exactly on the
directions that can reach the missed cells.

---

## 8. What "integer nullspace" means and why exactness matters

We compute the nullspace using exact rational arithmetic (sympy), then scale to
the smallest integer representative.  Why not just use floating-point SVD?

When we compute a sign vector, we evaluate `sign(A_i · w + c_i)` where `w` is
our witness and all values are integers.  If `v` is only *approximately* in the
nullspace (e.g. `A_i · v = 1e-14`), then the witness `w = p + t_final · v` will
have floating-point coordinates, and the sign could be wrong — we might classify
the witness as inside the wrong cell.

By using exact integer arithmetic throughout (integer `v`, integer `p`, integer
`t_final`), we guarantee `A_i · w` is an exact integer, and the sign vector is
correct by construction.

---

## 9. Known limitations of the P2 approach

**Sampling coverage:** P2 samples a random subset of hyperplanes each iteration.
For a given lower-dimensional face (intersection of exactly k hyperplanes), the
probability of hitting exactly that face in one iteration is `C(N,k)^{-1}`.
With large N this can be small, so some tube cells may still be missed after a
finite number of subsets.  More `face_subsets` and `face_offsets` = better
coverage.

**Subset distribution:** The current implementation samples subset sizes
uniformly from 1 to D-1.  This is a heuristic — the "right" distribution
depends on the arrangement structure, which we don't know in advance.

**No completeness guarantee:** Unlike the exact extractor (which enumerates
every non-empty cell), P2 is still a probabilistic heuristic.  It provably
reaches some cells that generic shooting cannot, but makes no claim about
completeness.

---

## 10. Summary

| Method | What cells it reaches | Mechanism |
|---|---|---|
| Generic origin shooting | Cells with full-dimensional recession cones | Random directions from origin |
| Face-aligned shooting (P2) | Cells with lower-dimensional recession cones | Random nullspace directions from random offsets |
| Exact extractor | Every non-empty cell | Exhaustive reverse search + MILP |

P2 complements the generic phase by systematically targeting the structural gap
— cells that no amount of generic rays could ever find.  Together, the two
phases cover more of the shard space with no additional solver cost.
