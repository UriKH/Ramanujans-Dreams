from dataclasses import dataclass, field
from typing import Tuple

from .configurable import Configurable


@dataclass
class PostProcessConfig(Configurable):
    """Configuration for the Tier-3 post-process stage.

    Runs once after the Search stage finishes.  Reads existing per-shard
    JSONL files, reconstructs ``TrajectoryAttributesHandler`` for every
    trajectory missing the configured attributes, and appends patch records
    that the merge-on-read reader folds in transparently.

    Empty ``TIER3_ATTRIBUTES`` (the default) short-circuits the whole
    stage — no JSONL files are read, no subprocesses spawned.
    """

    TIER3_ATTRIBUTES: Tuple[str, ...] = field(
        default=(),
        metadata={
            "description": (
                "Expensive symbolic attributes (e.g. 'asymptotics', 'kamidelta') "
                "computed in the post-process stage after Search has finished. "
                "Empty = skip the stage entirely."
            )
        },
    )


post_process_config: PostProcessConfig = PostProcessConfig()
