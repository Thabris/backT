"""
Execution engine interface for BackT

Defines the abstract interface for execution engines,
allowing for pluggable execution models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import pandas as pd

from ..utils.types import Fill, OrderDict, TimeSeriesData


class ExecutionEngine(ABC):
    """Abstract base class for execution engines"""

    @abstractmethod
    def execute(
        self,
        orders: Dict[str, OrderDict],
        market_data: Dict[str, TimeSeriesData],
        current_time: pd.Timestamp,
        positions: Dict[str, "Position"],
        context: Dict[str, Any]
    ) -> List[Fill]:
        """
        Execute orders and return fills

        Args:
            orders: Dictionary of orders keyed by symbol
            market_data: Current market data for all symbols
            current_time: Current timestamp
            positions: Current positions
            context: Strategy context and additional information

        Returns:
            List of Fill objects representing executed trades
        """
        pass

    @abstractmethod
    def can_execute(
        self,
        order: OrderDict,
        symbol: str,
        market_data: TimeSeriesData,
        current_time: pd.Timestamp
    ) -> bool:
        """
        Check if an order can be executed given current market conditions

        Args:
            order: Order to check
            symbol: Symbol for the order
            market_data: Market data for the symbol
            current_time: Current timestamp

        Returns:
            True if order can be executed, False otherwise
        """
        pass

    @abstractmethod
    def calculate_commission(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> float:
        """
        Calculate commission for a trade

        Args:
            symbol: Symbol being traded
            quantity: Number of shares/units
            price: Price per share/unit

        Returns:
            Commission amount
        """
        pass

    @abstractmethod
    def calculate_slippage(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_data: TimeSeriesData
    ) -> float:
        """
        Calculate slippage for a trade

        Args:
            symbol: Symbol being traded
            quantity: Number of shares/units (signed for direction)
            price: Base price
            market_data: Current market data

        Returns:
            Slippage amount (positive reduces proceeds for sells, increases cost for buys)
        """
        pass