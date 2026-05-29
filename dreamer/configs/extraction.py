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
    LOAD_SHARD_CACHE: bool = field(
        default=False,
        metadata={
            "description": (
                "If True and a '<cmf>__shards.jsonl' file already exists for "
                "the CMF under EXPORT_CMFS, load those shards (encoding + "
                "interior point) and skip the expensive shard extraction. "
                "Newly found shards are still appended to the same file."
            )
        },
    )
    EXACT_UNBOUNDED_CHECK: str = field(
        default="lp",
        metadata={
            "description": (
                "Backend for the exact strategy's per-cell unbounded check: "
                "'lp' (default) uses an in-process recession-cone LP (~0.5ms, "
                "no external dependency); 'lrs' spawns the lrslib binary "
                "(~30x slower per cell, authoritative cross-check)."
            )
        },
    )
    EXACT_NUM_WORKERS: int = field(
        default=1,
        metadata={
            "description": (
                "Process count for the exact strategy's parallel reverse "
                "search.  >1 dispatches disjoint cell subtrees across "
                "processes (each enumerates + classifies + locates its own "
                "subtree, and salvages partial results on timeout).  1 "
                "(default) runs serially.  Forces the 'lp' unbounded check."
            )
        },
    )
    HEURISTIC_REFINE_WITNESSES: bool = field(
        default=False,
        metadata={
            "description": (
                "If True, the heuristic ray-shooter post-processes far-out "
                "shards with an MILP to return the L1-minimal "
                "(closest-to-origin) integer start point instead of the raw "
                "ray witness.  Only witnesses with L1 norm above "
                "HEURISTIC_REFINE_L1_THRESHOLD are recomputed.  Default False "
                "keeps the solver-free fast path.  Also applies to the "
                "heuristic top-up under STRATEGY='auto'."
            )
        },
    )
    HEURISTIC_REFINE_L1_THRESHOLD: float = field(
        default=50.0,
        metadata={
            "description": (
                "When HEURISTIC_REFINE_WITNESSES is on, only ray witnesses "
                "whose L1 norm (sum of |coordinates|) exceeds this are "
                "recomputed via MILP; smaller witnesses are already close "
                "enough to the origin and kept as-is.  0 refines every "
                "shard.  Default 50."
            )
        },
    )
    HEURISTIC_REFINE_WORKERS: int = field(
        default=1,
        metadata={
            "description": (
                "Process count for the heuristic's MILP witness refinement "
                "(the per-shard MILPs are independent).  >1 dispatches them "
                "across processes; 1 (default) runs serially."
            )
        },
    )


extraction_config: ExtractionConfig = ExtractionConfig()
