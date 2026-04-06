from .system import sys_config
from .database import db_config, DBUsages
from .analysis import analysis_config
from .search import search_config
from .extraction import extraction_config
from .logging import logging_config
from .config_manager import ConfigManager


config = ConfigManager()

__all__ = [
    'config',
    'sys_config',
    'db_config',
    'extraction_config',
    'DBUsages',
    'analysis_config',
    'search_config',
    'logging_config'
]
