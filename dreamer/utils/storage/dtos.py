"""
Data Transfer Objects (DTOs) for the CMF Atlas pipeline.

DTOs are immutable snapshots of pipeline entities (CMF families, CMFs, shards,
trajectories) intended for incremental storage in JSONL files and eventual
migration into a relational database.

Design rules:
  - All fields are JSON-serializable primitives or collections of primitives.
  - Tuple-typed fields are serialized as JSON arrays; ``from_dict`` converts them
    back to tuples so round-trips are lossless.
  - ``extended_metrics`` on TrajectoryDTO is an intentionally open dict for
    asynchronous workers to populate without schema changes.
  - ``frozen=True`` prevents accidental field reassignment; mutable dict fields
    (``extended_metrics``) can still be updated in place by background workers.
"""

import json
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# CMF family
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CmfFamilyDTO:
    """Top-level CMF family record (e.g. the whole pFq family)."""
    family_id: str                          # e.g. "4F3"
    global_family_id: str                   # e.g. "pFq"
    matrix_definitions: Dict[str, str]      # symbol name → symbolic matrix str
    dimensions: int

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "CmfFamilyDTO":
        return cls(
            family_id=d["family_id"],
            global_family_id=d["global_family_id"],
            matrix_definitions=d["matrix_definitions"],
            dimensions=d["dimensions"],
        )


# ---------------------------------------------------------------------------
# CMF instance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CmfDTO:
    """A single CMF instance within a family."""
    cmf_id: str
    family_id: str
    cmf_hyperplanes: List[str]
    coordinate_shift: Tuple[int | str, ...]
    found_constants: List[str]

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "CmfDTO":
        return cls(
            cmf_id=d["cmf_id"],
            family_id=d["family_id"],
            cmf_hyperplanes=d["cmf_hyperplanes"],
            coordinate_shift=tuple(d["coordinate_shift"]),
            found_constants=d["found_constants"],
        )


# ---------------------------------------------------------------------------
# Shard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShardDTO:
    """A bounded convex region of a CMF's integer lattice (Ax < b)."""
    shard_id: str
    cmf_id: str
    shard_encoding: Tuple[int, ...]         # sign-vector encoding of the shard
    dimensionality: int
    found_constants: List[str]
    # --- optional fields (computed lazily or not yet available) ---
    interior_point: Optional[Tuple[int, ...]] = None
    volume_estimate: Optional[float] = None
    orthogonality_defect: Optional[float] = None

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ShardDTO":
        return cls(
            shard_id=d["shard_id"],
            cmf_id=d["cmf_id"],
            shard_encoding=tuple(d["shard_encoding"]),
            dimensionality=d["dimensionality"],
            found_constants=d["found_constants"],
            interior_point=tuple(d["interior_point"]) if d.get("interior_point") is not None else None,
            volume_estimate=d.get("volume_estimate"),
            orthogonality_defect=d.get("orthogonality_defect"),
        )


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajectoryDTO:
    """
    A single trajectory through a shard, with its Tier-2 base attributes.

    ``extended_metrics`` is populated asynchronously by background workers
    (Tier-3 and Tier-4 attributes).  Even though the dataclass is frozen,
    the dict itself is mutable — workers can do ``dto.extended_metrics[k] = v``
    without breaking the frozen contract.
    """
    trajectory_id: str
    cmf_id: str
    shard_id: str

    # Raw parameters (tuples instead of Position objects for JSON compatibility)
    start_point: Tuple[int | str, ...]
    direction: Tuple[int | str, ...]

    # Tier-2 base attributes
    recurrence_relation: str
    recurrence_order: int
    limit_value: float
    delta_estimate: float
    p_vector: Tuple[int | str, ...]
    q_vector: Tuple[int | str, ...]

    # Open extension field for Tier-3/4 attributes added by background workers
    extended_metrics: Dict[str, Any] = field(default_factory=dict, hash=False)

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryDTO":
        """Reconstruct from a JSON-parsed dict. Converts JSON lists back to tuples."""
        return cls(
            trajectory_id=d["trajectory_id"],
            cmf_id=d["cmf_id"],
            shard_id=d["shard_id"],
            start_point=tuple(d["start_point"]),
            direction=tuple(d["direction"]),
            recurrence_relation=d["recurrence_relation"],
            recurrence_order=d["recurrence_order"],
            limit_value=d["limit_value"],
            delta_estimate=d["delta_estimate"],
            p_vector=tuple(d.get("p_vector", ())),
            q_vector=tuple(d.get("q_vector", ())),
            extended_metrics=d.get("extended_metrics", {}),
        )
