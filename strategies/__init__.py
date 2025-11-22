"""
BackT Strategy Library

Organized collection of trading strategies by theme.

Available modules:
- momentum: Momentum and trend-following strategies
- aqr: AQR factor strategies
- benchmark: Benchmark strategies (buy & hold, etc.)
"""

# Import submodules so they can be accessed as strategies.momentum, strategies.aqr
from . import momentum
from . import aqr
from . import benchmark

# Also import commonly used functions for convenience
from .momentum import (
    ma_crossover_long_only,
    ma_crossover_long_short,
    kalman_ma_crossover_long_only,
    kalman_ma_crossover_long_short
)

__all__ = [
    'momentum',
    'aqr',
    'benchmark',
    'ma_crossover_long_only',
    'ma_crossover_long_short',
    'kalman_ma_crossover_long_only',
    'kalman_ma_crossover_long_short',
]
