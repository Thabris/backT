"""
Signal module for BackT

This module provides utilities for creating trading signals and strategies,
including technical indicators and common signal processing functions.
"""

from .indicators import TechnicalIndicators
from .strategy_helpers import StrategyHelpers

__all__ = [
    "TechnicalIndicators",
    "StrategyHelpers"
]