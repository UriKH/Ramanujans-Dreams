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

Needed work: Adapt and upgrade current implementation into the system pipline format.

**Implemented: Partial**

## Simulated Annealing

Algorithm basic implementation is provided in [annealing.py](context/resources/code/algos/annealing.py).
More details in: [resources](context/resources/).

Needed work: Adapt and upgrade current implementation into the system pipline format.

**Implemented: Partial**