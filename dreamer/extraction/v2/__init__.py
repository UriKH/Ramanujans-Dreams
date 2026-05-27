"""
v2 shard extraction module.

Strategy-pattern reimplementation of the shard-finding stage that
replaces the brute-force lattice scan in
:mod:`dreamer.extraction.utils.initial_points`.  See
``context/SHARD_EXTRACTION_PLAN.md`` for the motivating design.

The old extractor remains the production path; nothing in this package
is wired into :class:`dreamer.extraction.extractor.ShardExtractor` yet.
"""

from .base import BaseExtractor, ShardMapping, SignEncoding
from .lrs_extractor import LrslibExtractor
from .manager import ExtractionManager, Strategy
from .ray_extractor import RayShootingExtractor

__all__ = [
    "BaseExtractor",
    "ExtractionManager",
    "LrslibExtractor",
    "RayShootingExtractor",
    "ShardMapping",
    "SignEncoding",
    "Strategy",
]
