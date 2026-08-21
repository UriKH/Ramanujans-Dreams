from dreamer import System, config
from dreamer import analysis, search, extraction, post_process
from dreamer.loading import pFq
from dreamer import log, pi, zeta
import sympy as sp

# Because of pickling format we need to define these functions here
def trajectory_compute_func(d):
    """Number of search trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**d, 10)``).
    """
    return max(5 * 10 ** d, 10)


def trajectory_compute_func_analysis(d):
    """Number of analysis trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**(d-1), 10)``).
    """
    return max(10 ** (d - 6), 10)


if __name__ == '__main__':
    config.configure(
        system={
            'EXPORT_CMFS': './CMFs',                                # export CMF as objects to directory: ./CMFs
            'EXPORT_ANALYSIS_PRIORITIES': './analysis priorities',  # export shards found in analysis into: ./analysis priorities
            'EXPORT_SEARCH_RESULTS': './search results',            # export the search results into: ./search results
            'PATH_TO_SEARCHABLES': './spaces',                       # export all shard to this directory: ./spaces
            'EXPORT_ANALYSIS_PRIORITIES_FORMAT': 'json',
            'EXPORT_SEARCHABLES_FORMAT': 'json',
            'EXPORT_SEARCH_RESULTS_FORMAT': 'json'
        },
        analysis={
            # ignore shards with less than 0.1% identified trajectories as converge to the constant
            'IDENTIFY_THRESHOLD': 1e-3,
            # number of trajectories to be auto-generated in analysis
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func_analysis,
            'STORE_TRAJECTORIES_SEPARATELY': True,
            'SAMPLING_METHOD': 'pt'
        },
        extraction={
            # In this case this indicates usage of pFq symmetries utilization to reduce the number of shards
            'IGNORE_DUPLICATE_SEARCHABLES': True,
            #   'auto'      -- try exact (lrs + MILP), fall back to heuristic on timeout (DEFAULT)
            #   'exact'     -- lrs + MILP only; raises on failure
            #   'heuristic' -- ray-shooting only (Best for high dimensional CMFs)
            #   'legacy'    -- brute-force lattice scan
            'STRATEGY': 'heuristic',
            # Under 'auto': exact extractor gets EXACT_TIMEOUT_SECONDS before
            # falling back; heuristic then gets HEURISTIC_TIMEOUT_SECONDS.
            # Under 'exact'/'heuristic' alone, only the matching knob applies.
            'EXACT_TIMEOUT_SECONDS': 300.0,
            'HEURISTIC_TIMEOUT_SECONDS': 300.0,
            'LOAD_SHARD_CACHE': True,
        },
        search={
            'DEFAULT_USES_INV_T': True,
            'MAX_TRAJECTORY_LENGTH': 100,
            'ANNEAL_TMIN': 1e-4,
            'SAMPLING_METHOD': 'pt',
            'TIER2_ATTRIBUTES': (),
            'MAX_CONSTANT_RESOLUTION': 200_000
        },
        logging={
            'GENERATE_LOGS': True
        },
        post_process={
            # Each entry: bare attribute name, or (attribute, predicate). Predicate may be:
            #   'if_identified' / 'if_has_degree_2'            -- named, handler-only
            #   'max_degree below N' / 'max_degree above N'    -- recurrence polynomial degree
            #   'top N highest <metric> in shard|cmf'          -- shard/CMF-scoped, gate-only
            #   'top N lowest  <metric> in shard|cmf'
            # <metric> read from stored JSONL: delta, convergence_rate, asymptotic_digits_per_step,
            # spectral_gap, gcd_slope, precision_at. (To rank on a Tier-2/3 metric, store it first.)
            # Example:
            #   'TIER3_ATTRIBUTES': (
            #       ('asymptotics', 'top 3 highest convergence_rate in cmf'),
            #       ('delta_sequence', 'top 10 highest delta in shard'),
            #       ('relation', 'max_degree below 4'),
            #   ),
            'TIER3_ATTRIBUTES': ()
        },
        # Post-process graphing (writes under system.EXPORT_GRAPHS; all off by default).
        graph={
            'PLOT_BEST_DELTA_SEQUENCE': True,   # d-sequence of the best trajectory per (CMF, constant)
            'PLOT_DELTA_HISTOGRAMS': True,      # d histograms per shard and per CMF
            'WRITE_BUMPINESS_TABLE': True,      # per-shard d non-smoothness (semivariogram + d-seq TV)
            'DELTA_SEQUENCE_DEPTH': 1000,
        },
    )

    System(
        function_sources=[pFq(zeta(5), 6, 5, 1,
                selected_start_points=[(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2)],
                selected_trajectories=[(11, 12, 13, 14, 15, 16, 30, 29, 28, 27, 26)],
                only_selected=True)],
        extractor=extraction.extractor.ShardExtractorMod,
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SimulatedAnnealingMod,
        # searcher=search.HybridSPSAMod,
        post_processor=post_process.Tier3PostProcessModV1,
    ).run(constants=[zeta(5)])
