from __future__ import annotations

import importlib.metadata

__version__ = importlib.metadata.version(__package__ or __name__)

from .attgt.attgt import ATTgt
from .datasets import load_data, simulate_data
from .twfe.twfe import TWFE

__all__ = ["TWFE", "ATTgt", "load_data", "simulate_data"]
