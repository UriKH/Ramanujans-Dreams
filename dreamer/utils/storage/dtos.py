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
    A single trajectory through a shard, with its Tier-1 base attributes.

    ``extended_metrics`` is populated asynchronously by Tier-2 background
    workers (and later by the Tier-3 post-process stage).  Even though the
    dataclass is frozen, the dict itself is mutable — workers can do
    ``dto.extended_metrics[k] = v`` without breaking the frozen contract.
    """
    trajectory_id: str
    cmf_id: str
    shard_id: str

    # Raw parameters (tuples instead of Position objects for JSON compatibility)
    start_point: Tuple[int | str, ...]
    direction: Tuple[int | str, ...]

    # Tier-1 base attributes — always present in any DTO record.
    recurrence_relation: str
    recurrence_order: int
    limit_value: float

    # Per-constant attributes — dicts keyed by constant name so that one
    # trajectory record covers all constants searched in this shard.
    # ``delta_estimate``: irrationality measure δ per constant.
    # ``p_vector`` / ``q_vector``: LIReC projection vectors per constant
    #   (None entry = constant not identified for this trajectory).
    # ``identified``: whether LIReC found a convergent p/q for this constant.
    delta_estimate: Dict[str, float]
    p_vector: Optional[Dict[str, Optional[Tuple[int | str, ...]]]]
    q_vector: Optional[Dict[str, Optional[Tuple[int | str, ...]]]]
    identified: Dict[str, bool] = field(default_factory=dict)

    # Walk-style flag: 1 → ``inv().T`` applied after walking the trajectory
    # matrix (the dual recurrence); 2 → walked directly.
    walk_type: int = 1

    # Open extension field for Tier-2+ attributes added by background workers
    extended_metrics: Dict[str, Any] = field(default_factory=dict, hash=False)

    def to_json_line(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryDTO":
        """Reconstruct from a JSON-parsed dict."""
        # p_vector / q_vector: dict of {const: tuple-or-None} or None
        def _restore_pq(raw) -> Optional[Dict[str, Optional[tuple]]]:
            if raw is None:
                return None
            if isinstance(raw, dict):
                return {k: (tuple(v) if v is not None else None) for k, v in raw.items()}
            return None  # unexpected format — discard gracefully

        return cls(
            trajectory_id=d["trajectory_id"],
            cmf_id=d["cmf_id"],
            shard_id=d["shard_id"],
            start_point=tuple(d["start_point"]),
            direction=tuple(d["direction"]),
            recurrence_relation=d["recurrence_relation"],
            recurrence_order=d["recurrence_order"],
            limit_value=d["limit_value"],
            delta_estimate=d.get("delta_estimate") or {},
            p_vector=_restore_pq(d.get("p_vector")),
            q_vector=_restore_pq(d.get("q_vector")),
            identified=d.get("identified") or {},
            walk_type=int(d.get("walk_type", 1)),
            extended_metrics=d.get("extended_metrics", {}),
        )
