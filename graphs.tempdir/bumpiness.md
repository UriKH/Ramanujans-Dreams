# Shard bumpiness (δ non-smoothness)

* **relative_nugget** = nugget / sill of the empirical semivariogram of δ over angular direction-distance — ≈1 → needle/bumpy (no spatial structure), ≈0 → smooth.  Density-robust (pairs binned by distance).
* **median_delta_seq_TV** = median over trajectories of the total variation of the stored δ-sequence (convergence wobble); needs `delta_sequence` in TIER3_ATTRIBUTES, else nan.

| cmf | constant | shard | n_trajectories | relative_nugget | nugget | sill | initial_slope | median_delta_seq_TV | n_TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pFq_3_2_1__0_0_0_0_0 | zeta-2 | 1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,1,1 | 51 | 0.07708 | 0.002344 | 0.0304 | 0.1636 | nan | 0 |
