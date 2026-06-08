"""Diagnostic harness for the DiscreteMCMCSampler.

Companion to ``tests/testing_tool.py`` (the raycast harness).  It reuses the same
synthetic shard archetypes (fat baseline / needle / pancake) but renders the four
diagnostics the sampling plan specifies for the discrete walk:

1. PCA variance plot (UNIT vectors) -> directions span the cone's angular dimension.
2. Boundary-slack histogram         -> points sit safely inside every facet.
3. L2 norm vs. quota timeline       -> the funnel/PID keep norms in the useful band.
4. Nearest-neighbour angle          -> repulsion enforced angular uniformity.

PCA is computed on **unit directions** ``v/||v||`` (not raw coordinates): raw-coordinate
PCA is dominated by radial norm spread and misreports a multi-dimensional angular cloud
as a 1D corridor.  See ``context/sampling_trajectories/SAMPLING_MATH.md`` Section 9.
"""

import csv
import time

import matplotlib.pyplot as plt

from dreamer.utils.rand import np
from dreamer.extraction.samplers.discrete_raycaster import DiscreteMCMCSampler
from tests.testing_tool import TestHarness


def _unit_pca_spectrum(rays):
    """Explained-variance ratio of the harvested **unit directions**.

    :param rays: ``(n, d)`` harvested integer vectors.
    :return: ``(variance_ratio, effective_dim_90)`` — the per-component variance ratio
        of ``v/||v||`` and the number of components needed to reach 90% variance.
    """
    lengths = np.linalg.norm(rays, axis=1)
    units = rays / lengths[:, None]
    centered = units - units.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    var = sv ** 2
    var_ratio = var / var.sum() if var.sum() > 0 else var
    eff_dim = int(np.searchsorted(np.cumsum(var_ratio), 0.90) + 1)
    return var_ratio, eff_dim


class DiscreteTestHarness(TestHarness):
    """Evaluation framework for the discrete lattice sampler.

    Inherits the synthetic ``A_prime`` generators from :class:`TestHarness` and adds a
    dashboard tailored to the four discrete-walk diagnostics.
    """

    def __init__(self):
        """Bind the harness to :class:`DiscreteMCMCSampler`."""
        super().__init__(engine_class=DiscreteMCMCSampler)

    def render_dashboard(self, scenario_name):
        """Render the 4-pane discrete-sampler diagnostic dashboard for one scenario.

        :param scenario_name: key of a stored run in ``self.results``.
        """
        data = self.results.get(scenario_name)
        if not data or "error" in data or len(data["rays"]) == 0:
            print(f"No valid data to plot for {scenario_name}.")
            return

        rays = np.asarray(data["rays"], dtype=np.float64)
        lengths = np.linalg.norm(rays, axis=1)

        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # 1. PCA variance — on UNIT directions (angular spread, not radial stretch).
        var_ratio, eff_dim = _unit_pca_spectrum(rays)
        axs[0, 0].bar(np.arange(1, len(var_ratio) + 1), var_ratio, color="#00FFFF", edgecolor="black")
        axs[0, 0].set_title(
            f"1. Unit-direction PCA — {scenario_name}\n(eff dim @90% = {eff_dim}/{rays.shape[1]})"
        )
        axs[0, 0].set_xlabel("principal component")
        axs[0, 0].set_ylabel("variance ratio")

        # 2. Boundary slack — distance to each facet (-B z must be > 0 inside).
        engine = data.get("engine")
        if engine is not None and engine.B.shape[0] > 0:
            # Map harvested originals back to flatland is non-trivial; instead measure
            # slack directly on the harvested integer vectors against A_prime rows.
            slacks = -(engine.A_prime @ rays.T).T.flatten()
            axs[0, 1].hist(slacks, bins=50, color="#00FF00", edgecolor="black")
            axs[0, 1].axvline(0.0, color="red", linestyle="--", label="boundary")
            axs[0, 1].legend()
        axs[0, 1].set_title(f"2. Boundary-slack histogram — {scenario_name}")
        axs[0, 1].set_xlabel("slack ( >0 = inside )")

        # 3. L2 norm vs. quota timeline — harvest order is the discovery order.
        axs[1, 0].plot(np.arange(1, len(lengths) + 1), lengths, color="#FFAA00", linewidth=1.5)
        axs[1, 0].set_title(f"3. L2 norm vs. harvest index — {scenario_name}")
        axs[1, 0].set_xlabel("harvest index (quota fill)")
        axs[1, 0].set_ylabel("L2 norm")

        # 4. Nearest-neighbour angle histogram — angular uniformity.
        u_rays = rays / lengths[:, np.newaxis]
        cos_sim = np.clip(u_rays @ u_rays.T, -1.0, 1.0)
        np.fill_diagonal(cos_sim, -1.0)
        min_angles = np.degrees(np.arccos(np.max(cos_sim, axis=1)))
        axs[1, 1].hist(min_angles, bins=50, color="#FF00FF", edgecolor="black")
        axs[1, 1].set_title(f"4. Nearest-neighbour angle — {scenario_name}")
        axs[1, 1].set_xlabel("min angle to neighbour (deg)")

        for ax in axs.flat:
            ax.grid(color="grey", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_one(self, scenario_name, target_quota=10000, seeding=True):
        """Run a single scenario and stash the engine alongside the harvested rays.

        :param scenario_name: archetype key understood by ``_generate_matrix``.
        :param target_quota: number of primitive vectors requested.
        :param seeding: whether to seed the synthetic-matrix RNG for reproducibility.
        """
        A_prime = self._generate_matrix(scenario_name, seeding)
        engine = self.EngineClass(A_prime)
        start = time.time()
        rays = engine.harvest(target_quota)
        elapsed = time.time() - start
        self.results[scenario_name] = {
            "rays": rays,
            "time": elapsed,
            "yield": len(rays),
            "d_flat": engine.d_flat,
            "engine": engine,
        }
        print(f"[{scenario_name}] yielded {len(rays)} vectors in {elapsed:.2f}s (d_flat={engine.d_flat})")
        return self.results[scenario_name]

    def summary_row(self, scenario_name):
        """Compute a flat dict of diagnostic metrics for one stored run.

        :param scenario_name: key of a stored run in ``self.results``.
        :return: dict of summary metrics (empty-ish row if the run yielded too few points).
        """
        data = self.results.get(scenario_name, {})
        rays = np.asarray(data.get("rays", []), dtype=np.float64)
        engine = data.get("engine")
        row = {
            "scenario": scenario_name,
            "d_orig": getattr(engine, "d_orig", ""),
            "d_flat": data.get("d_flat", ""),
            "yield": data.get("yield", 0),
            "time_s": round(data.get("time", 0.0), 3),
        }
        if rays.shape[0] < 3:
            return row

        lengths = np.linalg.norm(rays, axis=1)
        var_ratio, eff_dim = _unit_pca_spectrum(rays)
        u = rays / lengths[:, None]
        cos_sim = np.clip(u @ u.T, -1.0, 1.0)
        np.fill_diagonal(cos_sim, -1.0)
        nn = np.degrees(np.arccos(cos_sim.max(axis=1)))
        slacks = -(engine.A_prime @ rays.T) if engine is not None else np.array([[0.0]])

        row.update({
            "unit_pca_pc1": round(float(var_ratio[0]), 4),
            "unit_pca_effdim90": eff_dim,
            "norm_min": round(float(lengths.min()), 2),
            "norm_median": round(float(np.median(lengths)), 2),
            "norm_max": round(float(lengths.max()), 2),
            "slack_min": round(float(slacks.min()), 4),
            "slack_median": round(float(np.median(slacks)), 4),
            "nn_angle_min_deg": round(float(nn.min()), 2),
            "nn_angle_median_deg": round(float(np.median(nn)), 2),
        })
        return row

    def export_csv(self, path, scenario_names=None):
        """Write a one-row-per-scenario diagnostic summary CSV (shareable).

        :param path: output CSV file path.
        :param scenario_names: scenarios to include; defaults to all stored runs.
        """
        names = scenario_names if scenario_names is not None else list(self.results.keys())
        rows = [self.summary_row(n) for n in names]
        fields = [
            "scenario", "d_orig", "d_flat", "yield", "time_s",
            "unit_pca_pc1", "unit_pca_effdim90",
            "norm_min", "norm_median", "norm_max",
            "slack_min", "slack_median",
            "nn_angle_min_deg", "nn_angle_median_deg",
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        print(f"Wrote diagnostic summary CSV: {path}")


def run_diagnostic_dashboard(target_quota=10_000, scenario_name="15D_Pancake"):
    """Run one scenario through the discrete sampler and render its dashboard.

    :param target_quota: number of primitive vectors to request.
    :param scenario_name: archetype key (e.g. ``"10D_Fat_Baseline"``).
    """
    print("=" * 60)
    print(" DISCRETE MCMC SAMPLER — DIAGNOSTIC HARNESS")
    print("=" * 60)
    harness = DiscreteTestHarness()
    data = harness.run_one(scenario_name, target_quota)
    if "error" not in data and len(data["rays"]) > 0:
        harness.render_dashboard(scenario_name)


def run_gauntlet_csv(out_path="discrete_sampler_diagnostics.csv", target_quota=2_000):
    """Run all archetypes, render dashboards, and write a shareable summary CSV.

    :param out_path: path for the summary CSV.
    :param target_quota: quota requested per scenario.
    """
    harness = DiscreteTestHarness()
    scenarios = ["10D_Fat_Baseline", "15D_Needle", "15D_Pancake"]
    for name in scenarios:
        data = harness.run_one(name, target_quota)
        if "error" not in data and len(data["rays"]) > 0:
            harness.render_dashboard(name)
    harness.export_csv(out_path, scenarios)


if __name__ == "__main__":
    run_gauntlet_csv()
