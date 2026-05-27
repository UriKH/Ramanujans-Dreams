from dataclasses import dataclass, field
from .configurable import Configurable


@dataclass
class ExtractionConfig(Configurable):
    """
    Extraction stage configurations
    """
    PARALLELIZE: bool = field(
        default=True,
        metadata={"description": "Enable parallel shard extraction routines."},
    )
    INIT_POINT_MAX_COORD: int = field(
        default=2,
        metadata={"description": "Maximum coordinate magnitude when searching for shard interior points."},
    )
    IGNORE_DUPLICATE_SEARCHABLES: bool = field(
        default=True,
        metadata={"description": "Skip duplicate searchables detected during extraction."},
    )
    STRATEGY: str = field(
        default="auto",
        metadata={
            "description": (
                "Shard-extraction strategy.  'auto' / 'exact' / 'heuristic' route "
                "through the v2 ExtractionManager (lrs + MILP, with ray-shooting "
                "fallback); 'legacy' falls back to the brute-force lattice scan "
                "in dreamer.extraction.utils.initial_points."
            )
        },
    )
    STRATEGY_TIMEOUT_SECONDS: float = field(
        default=3600.0,
        metadata={
            "description": (
                "Wall-clock cap (seconds) on the exact strategy when "
                "STRATEGY='auto'; on timeout the ExtractionManager falls back "
                "to the heuristic ray-shooter."
            )
        },
    )


extraction_config: ExtractionConfig = ExtractionConfig()
