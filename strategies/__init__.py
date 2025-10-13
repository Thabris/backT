"""
BackT Strategy Library

Organized collection of trading strategies by theme.

Available modules:
- momentum: Moving average crossover strategies (traditional and Kalman-enhanced)
"""

from .momentum import (
    ma_crossover_long_only,
    ma_crossover_long_short,
    kalman_ma_crossover_long_only,
    kalman_ma_crossover_long_short
)

__all__ = [
    'ma_crossover_long_only',
    'ma_crossover_long_short',
    'kalman_ma_crossover_long_only',
    'kalman_ma_crossover_long_short',
]
