### CMFs to search

#### Types

D - for dual walk (with inverse transpose)
F - forward walk (no inverse transpose)

Line format: (constant, CMF, known best start and trajectory, search type)

**Note:** The order does not matter and is there only for convinience!

**Note:** Every run searches **all shards** of the CMF, never only the shard of the known start point. The known start point and trajectory (when given) are only an extra seed: the shard they lie in is included and its search starts from that trajectory. A line with no start point or trajectory searches the whole CMF the same way, without a seed.

#### Table

1. $L(2,\chi-3)$, ${}_4F_3(\frac{1}{4})$, $(\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, \frac{1}{2}; \frac{3}{2}, \frac{3}{2}, \frac{3}{2})$, $(5, 6, 7, 8; 17, 18, 8)$, D.
2. $\zeta(3)$, $G_{5, 5}^{5, 2}(1)$, $(1, 1,0,0,2;0,0,0,0,0)$, $(11, 12, -16, -17, 10; 0, 1, 2, -1, -2)$, D.
3. $\zeta(5)$, ${}_8F_7(1)$, $(5, 2, 2, 2, 2, 2, 2, 2; 4, 4, 4, 4, 4, 4, 4)$, $(41, 17, 16, 15, 14, 13, 12, 11; 24, 25, 26, 27, 28, 29, 30)$, D.
4. Catalan's $G$, ${}_4F_3(1)$, $(4, 4, -\frac{1}{2}, 0; \frac{7}{2}, \frac{7}{2}, 4)$, $(11, 9, -2, -11; -1, 0, 2)$, D.
5. $L(2,\chi-3)-\frac{2}{3}\zeta(2)$, $G_{4,4}^{4,2}(1)$, $(-1, 0, 2, \frac{8}{3}; 0, 0, 0, \frac{2}{3})$, $(-10, -9, 14, 6; 0, 1,2,-2)$, D.
6. $L(2,\chi-3)+\frac{2}{3}\zeta(2)$, $G_{4,4}^{4,2}(1)$, $(-1, 0, 2, \frac{7}{3}; 0, 0, 0, \frac{1}{3})$, $(-10, -9, 14, 6; 0, 1,2,-2)$, D.
7. Euler's $\gamma$, $G_{2,3}^{3,2}(1)$, $(0, 0; 0, 0, 0)$, $(0,0; 1,1,1)$, D.
8. Euler's $\gamma$, $G_{3,4}^{4,3}(1)$, $(0, 0, 0; 0, 0, 0, 0)$, $(0,0,0; 1,1,1,1)$, D.
9. $\log(\frac{3}{2})$, ${}_2F_1(-\frac{1}{2})$, $(2, 2; 4)$, $(1,1; 2)$, F.
10. $\log(\frac{4}{3})$, ${}_2F_1(-\frac{1}{3})$, $(2, 2; 4)$, $(1,1; 2)$, F.
11. $\log(\frac{5}{4})$, ${}_2F_1(-\frac{1}{4})$, $(1, 1; 2)$, $(1,1; 2)$, F.
12. $\log(\frac{6}{5})$, ${}_2F_1(-\frac{1}{5})$, $(1, 1; 2)$, $(1,1; 2)$, F.
13. $\log(\frac{5}{3})$, ${}_2F_1(-\frac{2}{3})$, $(1, 1; 2)$, $(1,1; 2)$, F.
14. $\log(\frac{7}{5})$, ${}_2F_1(-\frac{2}{5})$, $(1, 1; 2)$, $(1,1; 2)$, F.
15. $\log(5)$, ${}_2F_1(-4)$, $(1, 1; 2)$, $(1,1; 2)$, F.
16. $\log(7)$, ${}_2F_1(-6)$, $(1, 1; 2)$, $(1,1; 2)$, F.
17. $\log(2)$, ${}_3F_2(\frac{1}{2})$, $(1, 1, 1; 2, 1)$, $(7, 5, 6; 2, 1)$, F.
18. $\log(2)$, ${}_4F_3(\frac{1}{2})$, F.
19. $\zeta(3)+\zeta(5)$, ${}_8F_7(1)$, F.
20. $\zeta(2)-\gamma^2$, $G_{3,4}^{4,3}(1)$, $(0, 0, 0; 0, 0, 0, 0)$, $(0, 0, 0; 1, 1, 1, 1)$, D.
21. $\gamma+\log(z)$, $G_{2,3}^{3,2}(z)$, D. Note: $z$ is an integer. Might be cool to try for $z=2$.
22. Euler-Gompertz $\mathcal{G}$, $G_{2,1}^{0,2}(1)$, $(1,1; 0)$, $(1,1; 0)$, D.
23. $\frac{\pi}{\sqrt 3}$, ${}_3F_2(-\frac{1}{3})$, $(\frac{1}{2}, 1, \frac{1}{2}; \frac{1}{2}, \frac{3}{2})$, $(3, 3, 2; 0, 6)$, F.
24. $L(2,\chi-3)$, ${}_3F_2(4)$, $(0, 0, \frac{1}{2}; 1, 1)$, $(-1, -1, 0; 0, 0)$, F.


### Run specification

#### General specifications

- Ensure analysis and search depth is set to 1000.
- Trajectory length should be no longer than 100 where the trajectory length is computed as the sum of all entries.
- For each run create plots and preform postprocess to depth 1500, only if identified and for the 5 highest delta in the shard. Compute the delta sequence and create shard summary graphs.
- Run name must be informative - including the constant name in short (e.g. pi, gamma, L2chi, log3o2 - for log of 3 over 2), and the CMF name (e.g. 4F3-1-2 for ${}_4F_3(\frac{1}{2})$).

#### Shard extraction

For CMFs of 7 or more dimensions allow for up to 10 minutes of finding shards.

#### Memory and cores

Denote dimesnion with D (for both pFq and MeijerG, $D = p + q$). We assume the search space of a shard has the full dimension D (no trajectory constraint removes a direction).

**Cores: request $2D + 1$ cores, run with $2D$ workers** (`TOTAL_CORES = 2D`) and look-ahead on (`ANNEAL_LOOKAHEAD = 'auto'`, the default).
- Each Simulated Annealing step walks the $2D$ neighbours of the current trajectory (±1 in each direction), so $2D$ workers walk a whole step at once. The search budget per seed (500 accepted moves, at most 10,000 steps) does not depend on D.
- With look-ahead, workers that would sit idle already walk the most likely next step. Measured: 3-D runs 1.2–1.4x faster, 5-D runs 1.3x faster, with identical results.
- More workers than $2D$ barely help: in 5-D, 16 workers instead of 10 gave 4% more speed for 60% more cost. Without look-ahead, more than about $2D$ workers give nothing.
- The extra core is for the main process (it runs the SA loop, the watchdog threads and the monitor). Measured on Zeus: 0.11 cores on average, 0.24 at the 95th percentile, short peaks of 0.9. One core is enough; RD does not reserve it by itself, so it must be requested.
- Shard extraction and analysis also run on the $2D$ workers. They are spread evenly over the workers, so fewer cores make them take longer but cost the same.

| D | Workers (`TOTAL_CORES`) | Cores to request |
|---|---|---|
| 3 | 6 | 7 |
| 5 | 10 | 11 |
| 7 | 14 | 15 |
| 8 | 16 | 17 |
| 10 | 20 | 21 |
| 15 | 30 | 31 |

**Memory** (not billed; kept as before):
- For CMF of $D \leq 3$: ask for 16 GB of RAM
- For CMF of $4 \leq D \leq 5$: ask for 32 GB of RAM
- For CMF of $6 \leq D \leq 10$: ask for 64 GB of RAM
- For CMF of $11 \leq D$: ask for 128 GB of RAM

Measured: a 3-D run with 8 workers peaked at 2.2 GB on Zeus.

#### Performance on Zeus

Calibration job 4852776 (31-10-26, `zeus_new_q`, node n088, Intel Xeon Platinum 8358 @ 2.6 GHz, 2 threads per core, node load about 46 of 128): the ${}_2F_1(-\frac{1}{2})$ $\log(\frac{3}{2})$ run (shard extraction, analysis and SA) with 8 workers and look-ahead.

| Machine | Wall time | Busy cores (average) |
|---|---|---|
| Laptop (18 cores), mean of 3 runs | 126.5 s (125.8, 124.7, 129.0) | 7.1 |
| Zeus | 433.5 s (405 s of computation + about 28 s Python start) | 6.4 |

**Zeus is about 3.4x slower than the laptop** for the same run and the same number of workers. Use this factor to plan walltimes: Zeus time ≈ 3.4 × laptop time.

Estimates with this factor (laptop timings x 3.4):
- A 3-D run takes about 7 minutes.
- A 5-D run (for example ${}_3F_2(\frac{1}{2})$, about 180 shards) takes about 35–40 hours, about 6 NIS each. The 72 h walltime leaves room for about 2x more work than estimated.

#### Analysis

- Ensure analysis does not create more than 500,000 trajectories.
- Continue to search in all shards that include trajectories which are identified (use minimal identification threshold: $10^-3$)

#### Search 

Use Simulated Annealing search method.


#### Walltime

By dimesions (D):
- For CMF of $D \leq 3$: 24h via zeus_all_q
- For CMF of $4 \leq D \leq 5$: 72h via zeus_new_q
- For CMF of $6 \leq D \leq 10$: 168h via zeus_long_q
- For CMF of $11 \leq D$: 168h via zeus_long_q


### File Structure

- The working directory is: `/home/uri.kasher/work/New-Directions-In-Irationality-runs` in the Zeus HPC server.
- In the working directoy create a folder for each constant.
- For each of the constats create a subfolder for the specific run formatted as: `<CMF>_<type: D\F>_<search method>_<date>` (e.g. `4F3-1-2_D_SA_31-10-26` for CMF ${}_4F_3(\frac{1}{2})$ and Simulated Annealing method).
- In each run create a markdown file explicitly stating the run goal and the general idea (e.g. Running a search for constant X in CMF Y using known start point ... and trajectory ..., The run type is D. Using search method Simulated Annealing).
- Ensure that the run is from the speicifc folder such that the monitor server will connect to the run files in the specific run direcotry so a user could see it remotely.


**Global Important Note**

If you are unable to use any of these specifications due to any reason, report back and do not run.