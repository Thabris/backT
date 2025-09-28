"""
Utilities module for BackT

This module contains configuration classes, type definitions, constants,
and utility functions used throughout the backtesting framework.
"""

from .config import BacktestConfig, ExecutionConfig
from .types import Position, Fill, BacktestResult, OrderDict
from .constants import DEFAULT_COMMISSION, DEFAULT_SPREAD, SUPPORTED_FREQUENCIES
from .logging_config import setup_logging

__all__ = [
    "BacktestConfig",
    "ExecutionConfig",
    "Position",
    "Fill",
    "BacktestResult",
    "OrderDict",
    "DEFAULT_COMMISSION",
    "DEFAULT_SPREAD",
    "SUPPORTED_FREQUENCIES",
    "setup_logging"
]