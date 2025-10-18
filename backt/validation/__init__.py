"""
Validation module for BackT

Provides advanced validation techniques including:
- Combinatorial Purged Cross-Validation (CPCV)
- Overfitting detection metrics (PBO, DSR)
- Parameter grid optimization with cross-validation
- Statistical significance testing
"""

from backt.validation.cpcv_validator import CPCVValidator, CPCVConfig, CPCVResult
from backt.validation.overfitting import (
    calculate_probability_backtest_overfitting,
    calculate_deflated_sharpe_ratio,
    calculate_performance_degradation
)
from backt.validation.parameter_grid import ParameterGrid

__all__ = [
    "CPCVValidator",
    "CPCVConfig",
    "CPCVResult",
    "ParameterGrid",
    "calculate_probability_backtest_overfitting",
    "calculate_deflated_sharpe_ratio",
    "calculate_performance_degradation"
]
