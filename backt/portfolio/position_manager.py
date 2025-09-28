"""
Position management utilities for BackT

Additional position-level utilities and calculations.
"""

from ..utils.types import Position
from ..utils.logging_config import LoggerMixin


class PositionManager(LoggerMixin):
    """Additional position management utilities"""

    @staticmethod
    def calculate_position_return(position: Position, current_price: float) -> float:
        """Calculate return for a position"""
        if position.qty == 0 or position.avg_price == 0:
            return 0.0
        return (current_price - position.avg_price) / position.avg_price

    @staticmethod
    def get_position_weight(position: Position, current_price: float, total_portfolio_value: float) -> float:
        """Calculate position weight in portfolio"""
        if total_portfolio_value == 0:
            return 0.0
        position_value = abs(position.qty * current_price)
        return position_value / total_portfolio_value