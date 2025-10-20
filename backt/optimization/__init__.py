"""
Optimization module for BackT

Provides strategy parameter optimization and grid search functionality.
"""

from .optimizer import StrategyOptimizer, OptimizationResult
from .results import ParameterSetResult, OptimizationSummary

# Optional FLAML support
try:
    from .flaml_optimizer import FLAMLOptimizer
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False

__all__ = [
    "StrategyOptimizer",
    "OptimizationResult",
    "ParameterSetResult",
    "OptimizationSummary"
]

if HAS_FLAML:
    __all__.append("FLAMLOptimizer")
