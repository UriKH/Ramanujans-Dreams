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
from .symmetry import (
    BlockSortSymmetry,
    SymmetryStrategy,
    symmetry_for_cmf,
)

__all__ = [
    "BaseExtractor",
    "BlockSortSymmetry",
    "ExtractionManager",
    "LrslibExtractor",
    "RayShootingExtractor",
    "ShardMapping",
    "SignEncoding",
    "Strategy",
    "SymmetryStrategy",
    "symmetry_for_cmf",
]
