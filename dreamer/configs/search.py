from dataclasses import dataclass, field
from typing import Callable
from .configurable import Configurable
from typing import Tuple
import math


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


def depth_from_len(traj_len, dim) -> int:
    return min(round(1500 / max(traj_len / math.sqrt(dim), 1)), 1500)

def ga_generations(dim: int) -> int:
    return 15 + 4 * dim

def ga_population(dim: int) -> int:
    return 20 + 2 * dim ** 2


@dataclass
class SearchConfig(Configurable):
    PARALLEL_SEARCH: bool = field(default=True, metadata={"description": "Enable parallel trajectory evaluation where available."})
    SEARCH_VECTOR_CHUNK: int = field(
        default=1,
        metadata={"description": "Number of search vectors batched per parallel chunk."},
    )
    NUM_TRAJECTORIES_FROM_DIM: Callable = field(
        default=traj_from_dim,
        metadata={"description": "Callable mapping CMF dimension to target number of trajectories."},
    )
    DEPTH_FROM_TRAJECTORY_LEN: Callable = field(
        default=depth_from_len,
        metadata={"description": "Callable mapping trajectory length/dimension to maximum walk depth."},
    )
    DEPTH_CONVERGENCE_THRESHOLD: Tuple[float, ...] = field(
        default=(0.9, 0.95, 1.0),
        metadata={"description": "Convergence quality checkpoints used while selecting candidate depths."},
    )
    DEFAULT_USES_INV_T: bool = field(
        default=True,
        metadata={"description": "Default toggle for using inverse-transformed trajectories in search."},
    )

    # ============================== Delta calculation and validation settings ==============================
    LIMIT_DIFF_ERROR_BOUND: float = field(
        default=1e-10,
        metadata={"description": "Maximum absolute limit mismatch accepted when validating convergents."},
    )
    MIN_ESTIMATE_DENOMINATOR: int = field(
        default=1e6,
        metadata={"description": "Minimum denominator magnitude required for a reliable rational estimate."},
    )
    CACHE_ACCEPTANCE_THRESHOLD: float = field(
        default=1e-12,
        metadata={"description": "Tolerance for accepting cached p/q vectors as equivalent."},
    )
    IDENTIFY_CHECK_THRESHOLD: float = field(
        default=1e-10,
        metadata={"description": "Tolerance used when deciding whether a searched trajectory identifies the constant."},
    )

    COMPUTE_EIGEN_VALUES: bool = field(
        default=False,
        metadata={"description": "Compute eigenvalue diagnostics for trajectory matrices in search results."},
    )
    COMPUTE_GCD_SLOPE: bool = field(
        default=False,
        metadata={"description": "Compute gcd-slope diagnostics for search trajectories."},
    )
    COMPUTE_LIMIT: bool = field(
        default=False,
        metadata={"description": "Compute explicit limit approximations during search evaluation."},
    )

    # ============================== Attribute selection (new DTO pipeline) ==============================
    # Names listed here are resolved through
    # ``dreamer.utils.storage.attribute_registry.ATTRIBUTE_REGISTRY``.
    # Misspelled entries raise KeyError loudly.
    #
    # Tier model:
    #   Tier-1 — core DTO fields (delta, identified, limit, order, recurrence_relation,
    #            p/q vectors).  Always computed in the main thread.
    #   Tier-2 — async extras computed in background worker processes during search.
    #            Default is empty so a vanilla run does no extra work beyond Tier-1.
    #   Tier-3 — expensive post-process attributes (asymptotics, kamidelta).  Not
    #            yet implemented; will run as a separate pipeline pass.
    TIER2_ATTRIBUTES: Tuple[str, ...] = field(
        default=(
            # ("eigenvalues", "if_identified"), ("eigenvalue_errors", "if_identified"), ("spectral_gap", "if_identified"),
            # ("companion_coboundary_rank", "if_identified"), ("asymptotics", "if_identified"),
            # ("convergence_class", "if_identified"), ("kamidelta", "if_identified"), ("gcd_slope", "if_identified")
            ("asymptotics", "if_identified")
        ),
        metadata={"description": "Background-worker attributes computed asynchronously during search. Empty disables the worker/writer subprocesses entirely."},
    )

    # ============================== Genetic search settings ==============================
    # Number of evolutionary generations to run.
    GA_GENERATIONS: Callable[[int], int] | int = field(
        default=ga_generations,
        metadata={"description": "Genetic algorithm generation schedule as callable or fixed integer."},
    )
    # Number of individuals in each generation.
    GA_POPULATION_SIZE: Callable[[int], int] | int = field(
        default=ga_population,
        metadata={"description": "Genetic algorithm population-size schedule as callable or fixed integer."},
    )
    GA_ELITE_FRACTION: float = field(
        default=0.2,
        metadata={"description": "Fraction of top individuals carried unchanged between generations."},
    )
    GA_MUTATION_PROB: float = field(
        default=0.3,
        metadata={"description": "Per-child probability of applying mutation in genetic search."},
    )
    GA_MUTATION_STEP: int = field(
        default=1,
        metadata={"description": "Maximum coordinate perturbation magnitude for mutation steps."},
    )
    GA_CROSSOVER_PROB: float = field(
        default=0.5,
        metadata={"description": "Probability of crossover versus cloning during offspring creation."},
    )
    GA_MAX_RETRIES: int = field(
        default=3,
        metadata={"description": "Maximum retries when trajectory evaluation fails or produces invalid states."},
    )
    GA_REFINE_PROB: float = field(
        default=0.5,
        metadata={"description": "Probability of entering local-refinement mutation mode."},
    )
    GA_REFINE_COORD_PROB: float = field(
        default=0.5,
        metadata={"description": "Per-coordinate probability for refinement perturbations."},
    )
    GA_MAX_NO_IMPROVEMENT_COUNT_RETRY: int = field(
        default=5,
        metadata={"description": "Retry budget before stopping when no GA improvement is observed."},
    )

    # ============================== Raycaster settings ==============================
    MAX_TRAJECTORY_LENGTH: int = field(
        default=100,
        metadata={"description": "Upper bound for absolute trajectory coordinate values during search."},
    )

    MAX_SEARCH_RADIUS: int = field(
        default=10_000,
        metadata={"description": "Upper bound for search radius used to sample trajectories."},
    )

    CONSTANT_NO_DIGITS_HIGH_RES: int = field(
        default=50_000,
        metadata={"description": "Number of digits to use for high-resolution constant values."},
    )

    CONSTANT_NO_DIGITS_LOW_RES: int = field(
        default=1000, #600,
        metadata={"description": "Number of digits to use for low-resolution constant values."},
    )


search_config: SearchConfig = SearchConfig()
