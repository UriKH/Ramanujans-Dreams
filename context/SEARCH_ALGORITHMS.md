# Search Algorithms

Search and analysis algorithms used in the search and analysis stages of the pipeline (see [PIPLINE.md](context/background/PIPELINE.md)).

### The Setup

Provided a searchable object such as a Shard, we want to:
- In the context of analysis: check if a constant is found in a shard.
- In the context of search: search for the best delta value found in the Shard.

Definition of Shard and trajectory in: [MATH_OBJECTS.md](context/background/MATH_OBJECTS.md) 

Shard implementation in: [shard.py](dreamer/extraction/shard.py)

## Small Angle Search

0. Transform the A matrix of the shard into flatland space exactly as done in [HyperSpaceConditioner](dreamer/extraction/samplers/conditioner.py) class.
1. Sample an integer coordinate trajectory in the flatland space using an LP solver and reduce the coordinates to be of GCD=1. 
2. Compute attributes for the trajectory. In the real shard space (A(x+vt) < b) - a transformation is needed.
3. If operating in anslysis mode and the trajectory was not identified or when operating in search mode: pertubate the trajectory coordinates by adding +1/-1 ,to each of the coordinates separately and reduce trajectory coordinates to GCD=1.
4. For each pertubation, if the pertubated trajectory stays inside the shard: identify and compute the delta. 
If no pertubation stays inside the shard: double trajectory length (do not reduce to GCD=1 coordinate) and go back to step 2.
Otherwise: continue to stage 2 with the trajectory which provided the best delta. If the best trajectory was a pertubation apply reduction to GCD=1 coordinates, if it was the starting trajectory do not reduce.

Notes:
- This algorithm could be used as both analysis and search algorithm depending on the maximum search depth.
- All attribute computations are done in the original Shard space while the perubation and trajectory sampling occurs in the flatland spcae.

**Implemented: Full (search).**  Method `SmallAngleSearch` + error `NoInitialIdentification` in
[small_angle_scan.py](dreamer/search/methods/small_angle/small_angle_scan.py); flatland geometry
wrapper `FlatlandGeometry` in [flatland.py](dreamer/search/methods/small_angle/flatland.py);
search module `SmallAngleSearchMod` in
[small_angle_mod.py](dreamer/search/searchers/small_angle_mod.py).  GCD vector reduction helper
`reduce_to_primitive` in [fast_gcd.py](dreamer/extraction/utils/fast_gcd.py).  Config knobs
`SA_MAX_DEPTH`, `SA_IMPROVE_THRESHOLD`, `SA_PATIENCE`, `SA_MAX_DOUBLINGS`, `SA_RESERVOIR_SIZE` in
[search.py](dreamer/configs/search.py).

Implementation notes / decisions (search-only scope; analysis-mode branch deferred):
- **Initial seed (reservoir, not LP):** the existing shard sampler draws ~`SA_RESERVOIR_SIZE`
  candidate trajectories; they are tried in **ascending L2-norm** order (start close to the
  origin) and the first that *identifies* the constant seeds the climb.  If none identify,
  `NoInitialIdentification` is raised and caught by the module (logged, skip to next constant).
- **Per-constant:** a multi-constant shard runs the whole algorithm once per identified constant.
- **Perturbations are GCD-reduced** ⇒ the climb explores primitive *directions/angles*; magnitude
  growth happens only via the length-doubling branch (when no perturbation stays inside the cone).
- **Termination:** `SA_MAX_DEPTH` iterations, with early stop after `SA_PATIENCE` iterations
  without δ gain ≥ `SA_IMPROVE_THRESHOLD`.

## Genetic Search

Algorithm basic implementation is provided in [genetic.py](context/resources/code/algos/genetic.py).
More details in: [resources](context/resources/).

**Implemented: Full (search).**  Method `GeneticSearch` + error `NoInitialPopulation` in
[genetic_scan.py](dreamer/search/methods/genetic_search/genetic_scan.py);
search module `GeneticSearchModV2` in
[genetic_search_mod.py](dreamer/search/searchers/genetic_search_mod.py).
Shared geometry / evaluator in
[dreamer/search/methods/flatland/](dreamer/search/methods/flatland/).
Config knobs `GA_*` (existing) in [search.py](dreamer/configs/search.py).

Implementation notes / decisions (search-only scope):
- **Flatland genomes** — populations are integer vectors `z` in the
  LLL-reduced basis from `FlatlandGeometry`; `geom.is_inside(z)` enforces
  shard membership exactly.
- **No GCD reduction** — raw integer coords so that the `2*z ± 1` refinement
  move and magnitude growth remain meaningful (reference behaviour).
- **Operators faithful to reference** — single-point crossover, `random.choice(elites)`
  parent selection, refine_prob 0.7 (child-1) / 0.3 (child-2) asymmetry,
  δ-invalid resampling from the shard sampler.
- **Early-stop** on `GA_MAX_NO_IMPROVEMENT_COUNT_RETRY` unchanged-best generations
  (user decision; deliberate deviation from the fixed-generation reference).
- **Per-constant** — one GA run per identified constant; walk-reuse via `handler_cache`.

## Simulated Annealing

Algorithm basic implementation is provided in [annealing.py](context/resources/code/algos/annealing.py).
More details in: [resources](context/resources/).

**Implemented: Full (search).**  Method `SimulatedAnnealingSearch` + error
`NoInitialIdentification` in
[annealing_scan.py](dreamer/search/methods/annealing/annealing_scan.py);
search module `SimulatedAnnealingMod` in
[annealing_mod.py](dreamer/search/searchers/annealing_mod.py).
Shared geometry / evaluator in
[dreamer/search/methods/flatland/](dreamer/search/methods/flatland/).
Config knobs `ANNEAL_*` in [search.py](dreamer/configs/search.py).

Implementation notes / decisions (search-only scope):
- **Flatland genomes** — current position is an integer vector `z`; neighbours
  are raw ±1 unit steps (`geom.perturbations(reduce=False)`), filtered by
  `geom.is_inside` and excluding tabu entries.
- **No GCD reduction** — raw coords; `2*z` doubling stays meaningful.
- **Tabu list** — bounded recent-position list of size `ANNEAL_TABU_SIZE`
  (default 70, matching reference `14*5`).
- **Temperature schedule** — `linear` (T0/(k+1)) or `log` (T0/log(k+1)),
  advancing only on accepted moves (reference semantics).
- **Adaptive scaling** — on rejection, `z = 2*z`; after `ANNEAL_MAX_DOUBLINGS`
  consecutive doublings, reset and reseed a fresh direction (reference's
  dead "give up" branch fixed to its evident intent).
- **Per-constant** — one SA run per identified constant; walk-reuse via `handler_cache`.


## Gradient Ascent + variants

- A per-constant search method that **ascends** δ (larger δ is better) over the continuous
  trajectory-direction *angle*.  δ is continuous and generally smooth in the angle
  (non-differentiable only at a finite set of points), so a gradient method over a real-valued
  direction space is well-posed.
- The step is **semi-discrete**: the optimizer holds a real-valued direction; each update is
  *realized* as the angle-best integer trajectory whose L2 norm does not exceed `GRAD_MAX_NORM`
  (e.g. 100), then walked / evaluated.
- The gradient is estimated by **forward differences in angle space** (rotate the direction by a
  small angle toward each coordinate axis, realize + evaluate).
- ML-optimizer variants are selectable via `GRAD_VARIANT`: `vanilla`, `momentum`, `rmsprop`,
  `adam` (strategy pattern).
- **Number of steps** is a manual knob (`GRAD_MAX_STEPS`).
- **Convergence stop:** the ascent terminates when the gradient is too small (`GRAD_GRAD_TOL`),
  the snapped step cannot move, patience is exhausted, or the step budget is spent — it never
  spins on a local optimum.
- **Non-identified trajectory handling (three-stage):** (1) *skip* the offending probe and keep
  the last identified direction; (2) after `GRAD_SKIP_LIMIT` (=3) unproductive steps,
  length-double to escape the dead region; (3) after `GRAD_MAX_DOUBLINGS` doublings, *diffract*
  off the wall by drawing a random in-cone direction from the last identified trajectory; if all
  `GRAD_DIFFRACT_TRIES` diffraction attempts fail, raise `SearchStalled` (caught + logged later).

**Implemented: Full (search).**  Method `GradientAscentSearch` + errors `NoInitialIdentification`,
`SearchStalled` in
[grad_ascent_scan.py](dreamer/search/methods/gradient_ascent/grad_ascent_scan.py); optimizer
strategies (`VanillaGrad`/`Momentum`/`RMSprop`/`Adam` + `optimizer_for`) in
[optimizers.py](dreamer/search/methods/gradient_ascent/optimizers.py); lattice realization
(`snap_to_trajectory`, `rotate_toward`) in
[lattice.py](dreamer/search/methods/gradient_ascent/lattice.py); search module
`GradientAscentMod` in
[gradient_ascent_mod.py](dreamer/search/searchers/gradient_ascent_mod.py).  Shared geometry /
evaluator reused from [dreamer/search/methods/flatland/](dreamer/search/methods/flatland/).
Config knobs `GRAD_*` in [search.py](dreamer/configs/search.py).
