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


extraction_config: ExtractionConfig = ExtractionConfig()
