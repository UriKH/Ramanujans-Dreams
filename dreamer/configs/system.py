from .configurable import Configurable
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SystemConfig(Configurable):
    # ============================== Arguments ==============================
    CONSTANTS: List[str] | str = field(
        default_factory=list,
        metadata={"description": "Registered constants to search for when no runtime constants are provided."},
    )

    # ============================== Printing and errors ==============================
    MODULE_ERROR_SHOW_TRACE: bool = field(
        default=True,
        metadata={"description": "Whether stage-level module errors include full stack traces in logs."},
    )
    TQDM_CONFIG: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"description": "Progress-bar rendering options shared by SmartTQDM calls."},
    )
    LOGGING_BUFFER_SIZE: int = field(
        default=150,
        metadata={"description": "Width used by buffered log banners and separators."},
    )

    # ============================== constant mapping ==============================
    USE_LIReC: bool = field(
        default=True,
        metadata={"description": "Enable LIReC-backed identification when available."},
    )

    def __post_init__(self):
        self.TQDM_CONFIG = {
            'bar_format': '{desc:<40}' + ' ' * 5 + '{bar} | {elapsed} {rate_fmt} ({percentage:.2f}%)',
            'ncols': 100
        }

    DEFAULT_DIR_SUFFIX: str = field(
        default='tempdir',
        metadata={"description": "Suffix that marks temporary export directories eligible for cleanup."},
    )
    EXPORT_CMFS: Optional[str] = field(
        default=None,
        metadata={"description": "Optional directory path for exporting loaded CMFs before extraction."},
    )
    EXPORT_ANALYSIS_PRIORITIES: Optional[str] = field(
        default=None,
        metadata={"description": "Optional directory path for persisting analyzed shard priorities."},
    )
    EXPORT_SEARCH_RESULTS: str = field(
        default='search_results.tempdir',
        metadata={"description": "Directory used by search stage to save discovered results and metadata."},
    )
    PATH_TO_SEARCHABLES: str = field(
        default='searchables.tempdir',
        metadata={"description": "Default import directory for precomputed searchable shards."},
    )


sys_config: SystemConfig = SystemConfig()
