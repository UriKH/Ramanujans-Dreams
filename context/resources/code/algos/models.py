from dataclasses import dataclass
from typing import Any, List, Dict
from ramanujantools import Position


@dataclass
class DataTemplate:
    """
    running a single scan.
    data is typically a list of dicts {"trajectory": Position, "delta": float}
    produced by search_traj_sa / search_traj_ga.
    """
    name: str
    limit_expr: Any
    initial_point: Position
    data: List[Dict[str, Any]]
