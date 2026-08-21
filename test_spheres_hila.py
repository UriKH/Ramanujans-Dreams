
from ramanujantools.cmf import CMF, pFq as rt_pFq
import sympy as sp
from dreamer import System, config, analysis, search, extraction, post_process, zeta
from dreamer.loading import BaseCMF, pFq



# Because of pickling format we need to define these functions here
def trajectory_compute_func(d):
    """Number of search trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**d, 10)``).
    """
    return max(10 ** (d+1), 10)


def trajectory_compute_func_analysis(d):
    """Number of analysis trajectories to sample for a CMF of dimension ``d``.

    :param d: CMF dimensionality.
    :return: Trajectory count (``max(10**(d-1), 10)``).
    """
    return max(5 * 10 ** (d - 1), 10)


if __name__ == '__main__':
    config.configure(
        system={
            # export CMF as objects to directory: ./CMFs
            'EXPORT_CMFS': './CMFs',
            # export shards found in analysis into: ./analysis priorities
            'EXPORT_ANALYSIS_PRIORITIES': './analysis priorities',
            # export the search results into: ./search results
            'EXPORT_SEARCH_RESULTS': './search results',
            # export all shard to this directory: ./spaces
            'PATH_TO_SEARCHABLES': './spaces',
            'EXPORT_ANALYSIS_PRIORITIES_FORMAT': 'json',
            'EXPORT_SEARCHABLES_FORMAT': 'json',
            'EXPORT_SEARCH_RESULTS_FORMAT': 'json',
            'OPTIMIZATION_OBJECTIVE': 'delta',
        },
        analysis={
            # ignore shards with less than 0.1% identified trajectories as converge to the constant
            'IDENTIFY_THRESHOLD': 1e-3,
            # number of trajectories to be auto-generated in analysis
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func_analysis,
            'STORE_TRAJECTORIES_SEPARATELY': True,
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
            'EXACT_TIMEOUT_SECONDS': 60.0,
            'HEURISTIC_TIMEOUT_SECONDS': 200.0,
            'LOAD_SHARD_CACHE': True,
            'SAMPLING_METHOD': 'pt',
            'TRAJECTORY_CONSTRAINTS': {'x0': 12, 'x1': 14, 'y1': 28}
        },
        search={
            # number of trajectories to be auto-generated in search if needed by the module
            'NUM_TRAJECTORIES_FROM_DIM': trajectory_compute_func,
            'DEFAULT_USES_INV_T': False,
            'MAX_TRAJECTORY_LENGTH': 50,
            'SAMPLING_METHOD': 'pt'
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
            # δ-sequence of the best trajectory per (CMF, constant)
            'PLOT_BEST_DELTA_SEQUENCE': True,
            'PLOT_DELTA_HISTOGRAMS': True,      # δ histograms per shard and per CMF
            # per-shard δ non-smoothness (semivariogram + δ-seq TV)
            'WRITE_BUMPINESS_TABLE': True,
            'DELTA_SEQUENCE_DEPTH': 1000,
        },
    )

    x0, x1, x2 = sp.symbols('x:3')
    y0, y1 = sp.symbols('y:2')
    cmf = rt_pFq(3, 2, 1)

    # substitution = {x0: 12 * x0, y1: 28 * x0, x2: 14 * x0}
    # ccmf = CMF({x0: cmf.matrices[x0].subs(substitution).simplify(), x1: cmf.matrices[x1].subs(
    #     substitution).simplify(), y0: cmf.matrices[y0].subs(substitution).simplify()}, validate=False)
    ccmf = cmf

    System(
        function_sources=[pFq(zeta(2), 3, 2, 1, selected_start_points=[(3, 3, 3, 6, 6)], selected_trajectories=[(12, 13, 14, 24, 28)], only_selected=True)],
        extractor=extraction.extractor.ShardExtractorMod,
        analyzers=[analysis.AnalyzerModV1],
        searcher=search.SearcherModV1,
        post_processor=post_process.Tier3PostProcessModV1,
    ).run(constants=[zeta(2)])
