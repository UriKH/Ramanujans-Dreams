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
22. Euler-Gompertz $\mathcal{G}$, $G_{2,1}^{0,1}(1)$, $(1,1; 0)$, $(1,1; 0)$, D.
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

Denote dimesnion with D:
- For CMF of $D \leq 3$: ask for 16 GB of RAM, 16 cores
- For CMF of $4 \leq D \leq 5$: ask for 32 GB of RAM, 32 cores
- For CMF of $6 \leq D \leq 10$: ask for 64 GB of RAM, 64 cores
- For CMF of $11 \leq D$: ask for 128 GB of RAM, 128 cores

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