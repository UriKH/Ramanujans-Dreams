from dataclasses import dataclass, field
from .configurable import Configurable
from typing import Callable


def traj_from_dim(dim: int) -> int:
    return 10 ** dim


@dataclass
class AnalysisConfig(Configurable):
    """
    Stage analysis configurations
    """
    # ============================= Parallelism and efficiency =============================
    USE_CACHING: bool = field(
        default=True,
        metadata={"description": "Enable LRU caches used by analysis computations."},
    )

    NUM_TRAJECTORIES_FROM_DIM: Callable = field(
        default=traj_from_dim,
        metadata={"description": "Callable that maps searchable dimension to number of sampled trajectories."},
    )
    IDENTIFY_THRESHOLD: float = field(
        default=-1,
        metadata={"description": "Minimum identified-trajectory ratio required to keep a shard; -1 disables filtering."},
    )

    # ============================= Printing and error management =============================
    PRINT_FOR_EVERY_SEARCHABLE: bool = field(
        default=True,
        metadata={"description": "Log per-searchable analysis summaries during analyzer execution."},
    )
    SHOW_START_POINT: bool = field(
        default=True,
        metadata={"description": "Include searchable start points in analysis logs."},
    )
    SHOW_SEARCHABLE: bool = field(
        default=False,
        metadata={"description": "Include full searchable object dumps in analysis logs."},
    )

    # ============================= Analysis features =============================
    USE_LIReC: bool = field(
        default=True,
        metadata={"description": "Use LIReC constant-identification routines instead of fallback heuristics."},
    )
    ANALYZE_LIMIT: bool = field(
        default=False,
        metadata={"description": "Compute and store explicit limit estimates during analysis."},
    )
    ANALYZE_EIGEN_VALUES: bool = field(
        default=False,
        metadata={"description": "Compute trajectory matrix eigenvalues for each sampled search vector."},
    )
    ANALYZE_GCD_SLOPE: bool = field(
        default=False,
        metadata={"description": "Compute gcd-slope diagnostics for sampled trajectory values."},
    )


analysis_config: AnalysisConfig = AnalysisConfig()
