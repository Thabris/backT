"""
Risk module for BackT

Provides risk analytics and performance metrics calculation including
Sharpe ratio, maximum drawdown, VaR, and other standard risk measures.
"""

from .metrics import MetricsEngine
from .risk_calculator import RiskCalculator

__all__ = [
    "MetricsEngine",
    "RiskCalculator"
]