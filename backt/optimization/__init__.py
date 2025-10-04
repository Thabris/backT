"""
Optimization module for BackT

Provides strategy parameter optimization and grid search functionality.
"""

from .optimizer import StrategyOptimizer, OptimizationResult

__all__ = [
    "StrategyOptimizer",
    "OptimizationResult"
]
