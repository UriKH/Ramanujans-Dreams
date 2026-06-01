# Mathematical Objects — Ramanujan's Dreams

> **Doc type:** Reference (math vocabulary). Stable definitions of the
> objects the pipeline operates on. Update only when a definition itself
> changes or a new object is introduced — not when the code that uses
> them changes.
> For architecture and code structure, see [`DESIGN.md`](DESIGN.md).

---

## 1. Mathematical Constants

**Code:** `dreamer/utils/constants/constant.py`

A **constant** is a target mathematical value (e.g., $\pi$, $e$, $\ln 2$, $\zeta(3)$) that the system tries to express as a polynomial continued fraction.

Each constant has two representations:
- **Symbolic** (`sympy.Expr`): exact algebraic form, used for expression manipulation.
- **Numerical** (`mpmath.mpf`): arbitrary-precision value at ≥100 decimal digits, used for identification and verification.

Constants are globally registered: `Constant.registry["pi"]` always returns the same object. Two constants with the same name are definitionally the same constant (invariant 4 in `SYSTEM_SPEC.md`).

**Pre-defined constants:** $\pi$, $e$, $\ln 2$, $\zeta(3)$, and their variants are in `dreamer/utils/constants/ready_made_consts.py`.

---

## 2. CMF — Conservative Matrix Field

**Code:** `ramanujantools` library; wrapped in `dreamer/loading/funcs/`

A **Conservative Matrix Field** is the central algebraic structure the system searches inside.

A CMF of dimension $d$ assigns a matrix $M_s(\mathbf{x})$ to each symbol $s \in \{x_1, \ldots, x_d\}$, where the matrices depend on the integer-coordinate point $\mathbf{x} \in \mathbb{Z}^d$. The matrices satisfy a **commutativity constraint**: for any two points $\mathbf{x}$ and any two paths between them in the lattice, the product of matrices along both paths is the same. This path-independence is what makes the field "conservative."

**Walking a CMF:** Starting from a point $\mathbf{x}_0$ and moving in a direction $\mathbf{v}$, the matrix product
$$
P_N = M_{s_1}(\mathbf{x}_0) \cdot M_{s_2}(\mathbf{x}_0 + \mathbf{v}) \cdots M_{s_k}(\mathbf{x}_0 + N\mathbf{v})
$$
produces a sequence of convergents. If this sequence converges to a constant $c$, then the trajectory $(\mathbf{x}_0, \mathbf{v})$ **identifies** $c$.

**Implementations supported:**
- Hypergeometric $_pF_q$ (most common): `dreamer/loading/funcs/pFq_fmt.py`
- Meijer-G function: `dreamer/loading/funcs/meijerG_fmt.py`
- Raw CMF matrix definition: `dreamer/loading/funcs/base_cmf.py`

**CMFData** is the system's wrapper around a CMF object. It is a frozen dataclass that adds:
- `shift`: integer offsets per symbol (moves the lattice origin away from singularities)
- `use_inv_t`: whether to walk using the inverse-transpose of each matrix
- `cmf_name`: a human-readable label for grouping results

---

## 3. Integer Lattice

The CMF is defined on the **integer lattice** $\mathbb{Z}^d$, where $d$ is the number of symbols in the CMF (its dimension). Each point $\mathbf{x} = (x_1, \ldots, x_d) \in \mathbb{Z}^d$ is a valid starting position for a trajectory walk.

A **Position** (from `ramanujantools`) maps symbols to integer coordinates:
```python
Position({a: 3, b: 7})  # Point (a=3, b=7) in the lattice
```

Not all lattice points are usable: some produce singular or undefined matrices (poles, zeros of characteristic polynomials). The extraction stage identifies and partitions around these singular loci.

---

## 4. Hyperplane

**Code:** `dreamer/extraction/hyperplanes.py`

A **hyperplane** is a linear constraint of the form
$$
c_1 x_1 + c_2 x_2 + \cdots + c_d x_d = k, \quad c_i, k \in \mathbb{Z}
$$
derived from the CMF's matrix structure. Hyperplanes arise from two sources:
1. **Zeros of characteristic polynomials:** eigenvalues of $M_s(\mathbf{x})$ vanish on a hyperplane.
2. **Poles of rational entries:** matrix entries blow up on a hyperplane.

These hyperplanes are the boundaries between qualitatively different regions of the CMF's behavior — trajectories that cross them may fail to converge or converge to a different value.

The hyperplane divides the lattice into two open half-spaces:
- **Above** ($c_1 x_1 + \cdots > k$): encoded as $+1$
- **Below** ($c_1 x_1 + \cdots < k$): encoded as $-1$

In matrix form, the constraint $Ax < b$ (strict inequality) excludes the boundary itself.

---

## 5. Shard

**Code:** `dreamer/extraction/shard.py`

A **shard** is a maximal bounded convex region of the integer lattice where the CMF behaves consistently — no hyperplane boundary is crossed.

Formally, given $H$ hyperplanes with coefficient vectors $\mathbf{a}_1, \ldots, \mathbf{a}_H$ and constants $k_1, \ldots, k_H$, a shard is defined by a sign-vector encoding $\sigma \in \{-1, +1\}^H$:
$$
\sigma_i (\mathbf{a}_i \cdot \mathbf{x} - k_i) > 0 \quad \text{for all } i = 1, \ldots, H
$$

This is written compactly as the strict linear inequality system $A\mathbf{x} < \mathbf{b}$.

**Key properties:**
- **Bounded** — the shard is a bounded convex polytope.
- **Strict inequalities** — the boundary ($A\mathbf{x} = \mathbf{b}$) is excluded (invariant 3 in `SYSTEM_SPEC.md`).
- **Interior point** — each shard stores a known-interior point, used as the base for trajectory sampling.
- **Encoding** — the sign vector $\sigma \in \{-1, +1\}^H$ uniquely identifies the shard among all $2^H$ candidates.
- **Atomic unit of work** — analysis and search process shards independently. They can be parallelized.

**In code**, a `Shard` object stores:
- `A: np.ndarray`, `b: np.ndarray` — the inequality system
- `encoding: List[int]` — sign vector
- `interior_point: Position` — guaranteed inside point
- `shift: Position` — lattice origin offset
- `cmf: CMF`, `const: Constant`, `cmf_name: str` — what is being searched

---

## 6. Trajectory

**Code:** `dreamer/utils/storage/storage_objects.py` (`SearchVector`)

A **trajectory** is a ray through the integer lattice, defined by a **start point** and a **direction vector**:
$$
\mathbf{x}(n) = \mathbf{x}_0 + n \cdot \mathbf{v}, \quad n = 0, 1, 2, \ldots
$$
where $\mathbf{x}_0, \mathbf{v} \in \mathbb{Z}^d$.

Walking the CMF along this trajectory produces the convergent sequence. A trajectory is **valid** for a shard if the entire ray stays inside the shard (i.e., $A\mathbf{v} \leq \mathbf{0}$, so the direction never exits the shard as $n \to \infty$).

A trajectory **identifies** a constant $c$ if the matrix product
$$
\lim_{N \to \infty} P_N = \begin{pmatrix} p_\infty \\ q_\infty \end{pmatrix}
$$
satisfies $p_\infty / q_\infty \to c$ at a measurable convergence rate.

In code, `SearchVector` stores the `(start, trajectory)` pair (both as `Position` objects).

---

## 7. Convergence Rate δ

**δ** (delta) measures how fast a trajectory converges to a constant. It is defined (informally) as the number of new decimal digits gained per step of the walk.

A higher δ means faster convergence. Trajectories with δ close to 0 converge too slowly to be useful. Trajectories with large δ are excellent candidates for PCFs.

The analysis stage filters shards by δ (via `IDENTIFY_THRESHOLD`) and ranks surviving shards by their best δ. The search stage focuses on the highest-δ shards.

δ is computed using LIReC's constant identification infrastructure or directly from the ratio of successive convergent pairs.

---

## 8. PCF — Polynomial Continued Fraction

**Code:** `ramanujantools` library

A **Polynomial Continued Fraction** is the end product of a successful search. It has the form:
$$
a_0 + \cfrac{b_1}{a_1 + \cfrac{b_2}{a_2 + \cfrac{b_3}{a_3 + \cdots}}}
$$
where $a_n$ and $b_n$ are polynomials in $n$ with integer (or rational) coefficients.

A PCF is extracted from a trajectory by identifying the **p-vector** and **q-vector** — the numerator and denominator sequences of the convergents. LIReC's integer relation detection finds exact polynomial expressions for these sequences.

**Characterization of a PCF:**
- **Convergence rate δ** — number of digits gained per term.
- **Irrationality measure** — related to δ; constrains how well the constant can be approximated rationally.
- **Recurrence relation** — the linear recurrence satisfied by p and q.
- **Recurrence order** — degree of the recurrence.

The system produces PCFs as conjectures (verified to 100+ digits). Formal proofs require separate mathematical work.

---

## 9. Recurrence Relation

A **recurrence relation** is the algebraic description of how consecutive convergents of a PCF relate to each other. For a PCF with numerator sequence $\{p_n\}$, the recurrence has the form:
$$
p_{n+k} = r_{k-1}(n) \cdot p_{n+k-1} + \cdots + r_0(n) \cdot p_n
$$
where $r_i(n)$ are polynomials in $n$ and $k$ is the **recurrence order**.

The recurrence relation is the most compact representation of a discovered formula. It is stored in `TrajectoryDTO.recurrence_relation` (as a string) and `TrajectoryDTO.recurrence_order`.

---

## 10. Object Relationships

```
Constant
    │ searched inside
    ▼
CMF (pFq / MeijerG / raw)
    │ partitioned by Hyperplanes into
    ▼
Shard  (bounded convex region, Ax < b)
    │ sampled to produce
    ▼
Trajectory  (start point + direction vector)
    │ walked along CMF to produce
    ▼
Convergents → δ (convergence rate)
    │ if δ > threshold → extract
    ▼
PCF (p-vector, q-vector, recurrence relation)
```

---

*See also:* [`DESIGN.md`](DESIGN.md) for code structure, [`SYSTEM_SPEC.md`](../SYSTEM_SPEC.md) for invariants and pipeline details.
