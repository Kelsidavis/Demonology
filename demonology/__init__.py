"""
Demonology package - Core functionality for the mystical CLI.
"""

__version__ = "0.1.0"

from .client import DemonologyClient
from .ui import DemonologyUI
from .themes import Theme
from .config import Config

__all__ = ["DemonologyClient", "DemonologyUI", "Theme", "Config"]