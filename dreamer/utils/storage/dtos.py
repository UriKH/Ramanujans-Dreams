from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List, Any


@dataclass(frozen=True)
class CmfFamilyDTO:
    family_id: str  # e.g. 4F3
    global_family_id: str # e.g. pFq
    matrix_definitions: Dict[str, str]
    dimensions: int


@dataclass(frozen=True)
class CmfDTO:
    cmf_id: str
    family_id: str
    cmf_hyperplanes: List[str]
    coordinate_shift: Tuple[int | str, ...]


@dataclass(frozen=True)
class ShardDTO:
    shard_id: str
    cmf_id: str
    shard_encoding: Tuple[int, ...]
    dimensionality: int
    interior_point: Optional[Tuple[int, ...]] = None
    volume_estimate: Optional[float] = None
    orthogonality_defect: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryDTO:
    trajectory_id: str
    cmf_id: str
    shard_id: str

    # Raw Parameters (Using tuples instead of SymPy vectors)
    start_point: Tuple[int | str, ...]
    direction: Tuple[int | str, ...]

    # Base data
    recurrence_relation: str
    recurrence_order: int
    limit_value: float
    delta_estimate: float
    p_vector: Tuple[int | str, ...]
    q_vector: Tuple[int | str, ...]

    extended_metrics: Dict[str, Any] = field(default_factory=dict, hash=False)
