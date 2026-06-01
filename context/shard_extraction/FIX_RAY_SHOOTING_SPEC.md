# Context: Ramanujan's Dreams - Ray Extractor Vectorization Fix

You are acting as an expert Software Engineer. The previous implementation of `RayShootingExtractor` in `ray_extractor.py` is mathematically correct but computationally too slow because it is missing strict NumPy vectorization and ray-normalization. 

## 1. The Bottleneck
The current code likely iterates over rays using a Python loop or generates integer points that are massively large, which bottlenecks downstream processes. We need to process all rays simultaneously in a single NumPy matrix and reduce them to their primitive vectors.

## 2. Required Fixes

Please update `ray_extractor.py` with the following strict requirements:

### Fix A: Increase Default Ray Count
Change the default value of `num_rays` in the class `__init__` from `4096` to `100_000`. The vectorized math will handle this instantly.

### Fix B: The Primitive Ray Reduction (GCD)
When generating the initial random direction matrix `V` (Shape: `num_rays x D`), you MUST divide every ray by the Greatest Common Divisor (GCD) of its coordinates. This prevents "Fat Rays" and ensures the integer points are as close to the origin as possible.
* **Implementation:** ```python
  V = rng.integers(-self.max_coord, self.max_coord + 1, size=(self.num_rays, d), dtype=np.int64)
  # Reduce rays to primitive vectors
  gcds = np.gcd.reduce(V, axis=1, keepdims=True)
  gcds[gcds == 0] = 1  # Prevent divide-by-zero for the origin ray
  V = V // gcds