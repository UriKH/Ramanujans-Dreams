# Shard bumpiness (δ non-smoothness)

* **relative_nugget** = nugget / sill of the empirical semivariogram of δ over angular direction-distance — ≈1 → needle/bumpy (no spatial structure), ≈0 → smooth.  Density-robust (pairs binned by distance).
* **median_delta_seq_TV** = median over trajectories of the total variation of the stored δ-sequence (convergence wobble); needs `delta_sequence` in TIER3_ATTRIBUTES, else nan.

| cmf | constant | shard | n_trajectories | relative_nugget | nugget | sill | initial_slope | median_delta_seq_TV | n_TV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pFq_2_1_-1__0_0_0 | log-2 | -1,-1,-1,1,1,1,-1 | 39 | 0.05395 | 0.00243 | 0.04503 | 0.1112 | nan | 0 |
| pFq_2_1_-1__0_0_0 | log-2 | -1,-1,-1,1,1,1,1 | 65 | 0.06776 | 0.002604 | 0.03842 | 0.122 | nan | 0 |
| pFq_2_1_-1__0_0_0 | log-2 | -1,1,1,-1,1,1,-1 | 45 | 0.04867 | 0.00153 | 0.03143 | 0.1915 | nan | 0 |
| pFq_2_1_-1__0_0_0 | log-2 | 1,-1,-1,1,-1,-1,1 | 40 | 0.01493 | 0.0006018 | 0.04031 | 0.161 | nan | 0 |
