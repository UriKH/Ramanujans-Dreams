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

    MAX_TRAJECTORY_LENGTH: int = field(
        default=100,
        metadata={"description": "Upper bound for absolute trajectory coordinate values during search."},
    )


search_config: SearchConfig = SearchConfig()
