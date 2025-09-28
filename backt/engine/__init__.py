"""
Engine module for BackT

Contains the main backtesting engine that orchestrates the event-driven
simulation, coordinating data feeds, strategy signals, execution, and portfolio updates.
"""

from .backtester import Backtester

__all__ = [
    "Backtester"
]