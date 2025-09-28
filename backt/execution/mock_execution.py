"""
Mock execution engine for BackT

Provides realistic execution simulation with configurable
market impact, slippage, and commission models.
"""

import uuid
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from .execution_interface import ExecutionEngine
from ..utils.types import Fill, OrderDict, Position, TimeSeriesData
from ..utils.config import ExecutionConfig
from ..utils.constants import (
    BUY_ACTION, SELL_ACTION, CLOSE_ACTION, TARGET_WEIGHT_ACTION,
    MARKET_ORDER, LIMIT_ORDER
)
from ..utils.logging_config import LoggerMixin


class MockExecutionEngine(ExecutionEngine, LoggerMixin):
    """Mock execution engine with realistic market simulation"""

    def __init__(self, config: ExecutionConfig):
        """
        Initialize mock execution engine

        Args:
            config: Execution configuration
        """
        self.config = config
        self._order_id_counter = 0

    def execute(
        self,
        orders: Dict[str, OrderDict],
        market_data: Dict[str, TimeSeriesData],
        current_time: pd.Timestamp,
        positions: Dict[str, Position],
        context: Dict[str, Any]
    ) -> List[Fill]:
        """Execute orders and return fills"""
        fills = []

        for symbol, order in orders.items():
            if symbol not in market_data:
                self.logger.warning(f"No market data for symbol: {symbol}")
                continue

            symbol_data = market_data[symbol]
            current_position = positions.get(symbol, Position(symbol, 0.0, 0.0))

            try:
                fill = self._execute_single_order(
                    symbol, order, symbol_data, current_time, current_position
                )
                if fill is not None:
                    fills.append(fill)

            except Exception as e:
                self.logger.error(f"Error executing order for {symbol}: {e}")

        return fills

    def _execute_single_order(
        self,
        symbol: str,
        order: Dict[str, Any],  # Allow both dict and OrderDict
        market_data: TimeSeriesData,
        current_time: pd.Timestamp,
        position: Position
    ) -> Optional[Fill]:
        """Execute a single order"""

        # Skip if no action or hold
        action = order.get('action')
        if action == "hold" or action is None:
            return None

        # Get current market prices
        try:
            current_bar = market_data.loc[current_time]
        except KeyError:
            self.logger.warning(f"No data for {symbol} at {current_time}")
            return None

        # Determine target quantity based on order type
        if action == TARGET_WEIGHT_ACTION:
            weight = order.get('weight')
            if weight is None:
                self.logger.warning(f"Target weight order for {symbol} missing weight")
                return None
            # Note: This would need portfolio value from context
            # For now, assume a simple implementation
            target_qty = self._weight_to_quantity(weight, current_bar['close'], 100000)
            qty_to_trade = target_qty - position.qty

        elif action == CLOSE_ACTION:
            qty_to_trade = -position.qty

        elif action in [BUY_ACTION, SELL_ACTION]:
            size = order.get('size')
            if size is None:
                self.logger.warning(f"Order for {symbol} missing size")
                return None

            qty_to_trade = size if action == BUY_ACTION else -size

        else:
            self.logger.warning(f"Unknown order action: {action}")
            return None

        # Skip if no quantity to trade
        if abs(qty_to_trade) < 1e-8:
            return None

        # Check if order can be executed
        if not self.can_execute(order, symbol, market_data, current_time):
            return None

        # Determine execution price
        execution_price = self._get_execution_price(
            order, current_bar, qty_to_trade
        )

        # Calculate slippage
        slippage = self.calculate_slippage(
            symbol, qty_to_trade, execution_price, market_data
        )

        # Adjust price for slippage
        if qty_to_trade > 0:  # buying
            final_price = execution_price + slippage
        else:  # selling
            final_price = execution_price - slippage

        # Calculate commission
        commission = self.calculate_commission(symbol, abs(qty_to_trade), final_price)

        # Create fill
        fill = Fill(
            symbol=symbol,
            filled_qty=qty_to_trade,
            fill_price=final_price,
            commission=commission,
            slippage=slippage,
            timestamp=current_time,
            order_id=self._generate_order_id(),
            side="buy" if qty_to_trade > 0 else "sell",
            meta={
                "order_type": order.get('order_type', 'market'),
                "original_action": action,
                "execution_method": "mock"
            }
        )

        self.logger.debug(
            f"Executed {fill.side} {abs(fill.filled_qty):.2f} shares of {symbol} "
            f"at ${fill.fill_price:.2f} (commission: ${fill.commission:.2f})"
        )

        return fill

    def can_execute(
        self,
        order: Dict[str, Any],
        symbol: str,
        market_data: TimeSeriesData,
        current_time: pd.Timestamp
    ) -> bool:
        """Check if order can be executed"""
        try:
            current_bar = market_data.loc[current_time]

            # For limit orders, check if price is achievable
            order_type = order.get('order_type', 'market')
            limit_price = order.get('limit_price')
            action = order.get('action')

            if order_type == LIMIT_ORDER and limit_price is not None:
                if action == BUY_ACTION:
                    # Can buy if limit price >= low of the bar
                    return limit_price >= current_bar['low']
                elif action == SELL_ACTION:
                    # Can sell if limit price <= high of the bar
                    return limit_price <= current_bar['high']

            # Market orders can generally be executed
            return True

        except KeyError:
            return False

    def calculate_commission(
        self,
        symbol: str,
        quantity: float,
        price: float
    ) -> float:
        """Calculate commission for a trade"""
        commission = 0.0

        # Per-share commission
        commission += self.config.commission_per_share * quantity

        # Per-trade commission
        commission += self.config.commission_per_trade

        # Percentage commission
        commission += self.config.commission_pct * quantity * price

        return commission

    def calculate_slippage(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_data: TimeSeriesData
    ) -> float:
        """Calculate slippage for a trade"""
        # Simple percentage-based slippage model
        trade_value = abs(quantity * price)
        slippage = self.config.slippage_pct * trade_value

        # Could add more sophisticated models here:
        # - Volume-based slippage
        # - Volatility-based slippage
        # - Time-of-day effects
        # - Market impact models

        return slippage

    def _get_execution_price(
        self,
        order: Dict[str, Any],
        current_bar: pd.Series,
        quantity: float
    ) -> float:
        """Determine execution price based on order type"""

        order_type = order.get('order_type', 'market')
        limit_price = order.get('limit_price')

        if order_type == LIMIT_ORDER and limit_price is not None:
            # For limit orders, execute at limit price if possible
            if quantity > 0:  # buying
                return min(limit_price, current_bar['high'])
            else:  # selling
                return max(limit_price, current_bar['low'])

        elif order_type == MARKET_ORDER:
            # For market orders, use mid price with spread
            mid_price = (current_bar['high'] + current_bar['low']) / 2

            # Apply bid-ask spread
            spread_adjustment = self.config.spread / 2
            if quantity > 0:  # buying - pay ask
                return mid_price + spread_adjustment
            else:  # selling - receive bid
                return mid_price - spread_adjustment

        else:
            # Default to close price
            return current_bar['close']

    def _weight_to_quantity(
        self,
        weight: float,
        price: float,
        portfolio_value: float
    ) -> float:
        """Convert target weight to quantity"""
        target_value = weight * portfolio_value
        return target_value / price

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self._order_id_counter += 1
        return f"mock_{self._order_id_counter}_{uuid.uuid4().hex[:8]}"


class AdvancedMockExecutionEngine(MockExecutionEngine):
    """Advanced mock execution with more sophisticated models"""

    def __init__(
        self,
        config: ExecutionConfig,
        volatility_impact: bool = True,
        volume_impact: bool = True,
        market_hours_impact: bool = False
    ):
        super().__init__(config)
        self.volatility_impact = volatility_impact
        self.volume_impact = volume_impact
        self.market_hours_impact = market_hours_impact

    def calculate_slippage(
        self,
        symbol: str,
        quantity: float,
        price: float,
        market_data: TimeSeriesData
    ) -> float:
        """Enhanced slippage calculation with market microstructure effects"""
        base_slippage = super().calculate_slippage(symbol, quantity, price, market_data)

        # Get recent data for advanced calculations
        recent_data = market_data.tail(20)  # Last 20 bars

        # Volatility-based adjustment
        if self.volatility_impact and len(recent_data) > 1:
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            vol_multiplier = 1 + volatility * 2  # Higher vol = more slippage
            base_slippage *= vol_multiplier

        # Volume-based adjustment
        if self.volume_impact and 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].mean()
            current_volume = recent_data['volume'].iloc[-1]

            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                # Lower relative volume = higher slippage
                volume_multiplier = 1 + (1 / max(volume_ratio, 0.1) - 1) * 0.5
                base_slippage *= volume_multiplier

        # Market hours effect (if enabled)
        if self.market_hours_impact:
            current_time = recent_data.index[-1]
            hour = current_time.hour

            # Higher slippage during market open/close
            if hour in [9, 10, 15, 16]:  # Market open/close hours
                base_slippage *= 1.2

        return base_slippage