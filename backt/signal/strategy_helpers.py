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
    def calculate_equal_allocation(
        portfolio_value: float,
        symbols: list,
        current_prices: Dict[str, float],
        max_position_size: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate equal 1/N allocation for each symbol

        Args:
            portfolio_value: Total portfolio value
            symbols: List of symbols to allocate to
            current_prices: Current price for each symbol
            max_position_size: Maximum weight per position (0-1), default 1.0 (no limit)

        Returns:
            Dictionary mapping symbol to number of shares for equal allocation
        """
        n_symbols = len(symbols)
        if n_symbols == 0:
            return {}

        # Equal weight per symbol (1/N), but respect max_position_size
        weight_per_symbol = min(1.0 / n_symbols, max_position_size)
        allocation_per_symbol = portfolio_value * weight_per_symbol

        position_sizes = {}
        for symbol in symbols:
            if symbol in current_prices and current_prices[symbol] > 0:
                shares = allocation_per_symbol / current_prices[symbol]
                position_sizes[symbol] = shares
            else:
                position_sizes[symbol] = 0.0

        return position_sizes

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