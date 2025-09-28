"""
Strategy helper utilities for BackT

Common utilities and helpers for building trading strategies.
"""

from typing import Dict, Any
import pandas as pd
from ..utils.types import OrderDict, Position


class StrategyHelpers:
    """Helper functions for strategy development"""

    @staticmethod
    def create_market_order(action: str, size: float, **kwargs) -> Dict[str, Any]:
        """Create a market order dictionary"""
        order = {
            'action': action,
            'size': size,
            'order_type': 'market'
        }
        order.update(kwargs)
        return order

    @staticmethod
    def create_target_weight_order(weight: float, **kwargs) -> Dict[str, Any]:
        """Create a target weight order dictionary"""
        order = {
            'action': 'target_weight',
            'weight': weight,
            'order_type': 'market'
        }
        order.update(kwargs)
        return order

    @staticmethod
    def calculate_position_size(
        portfolio_value: float,
        target_weight: float,
        current_price: float
    ) -> float:
        """Calculate position size based on target weight"""
        target_value = portfolio_value * target_weight
        return target_value / current_price if current_price > 0 else 0

    @staticmethod
    def is_crossover(series1: pd.Series, series2: pd.Series) -> bool:
        """Check if series1 crossed over series2 in the last period"""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return series1.iloc[-2] <= series2.iloc[-2] and series1.iloc[-1] > series2.iloc[-1]

    @staticmethod
    def is_crossunder(series1: pd.Series, series2: pd.Series) -> bool:
        """Check if series1 crossed under series2 in the last period"""
        if len(series1) < 2 or len(series2) < 2:
            return False
        return series1.iloc[-2] >= series2.iloc[-2] and series1.iloc[-1] < series2.iloc[-1]