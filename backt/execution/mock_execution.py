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

    def __init__(self, config: ExecutionConfig, portfolio_manager=None):
        """
        Initialize mock execution engine

        Args:
            config: Execution configuration
            portfolio_manager: Reference to portfolio manager for risk checks
        """
        self.config = config
        self.portfolio_manager = portfolio_manager
        self._order_id_counter = 0

    def execute(
        self,
        orders: Dict[str, OrderDict],
        market_data: Dict[str, TimeSeriesData],
        current_time: pd.Timestamp,
        positions: Dict[str, Position],
        context: Dict[str, Any],
        symbol_portfolio: Optional['PortfolioManager'] = None
    ) -> List[Fill]:
        """Execute orders and return fills with risk management

        Args:
            symbol_portfolio: Symbol-specific portfolio manager (for independent execution)
        """
        fills = []

        # Get current prices for all symbols
        current_prices = {}
        for symbol, data in market_data.items():
            try:
                current_prices[symbol] = data.loc[current_time, 'close']
            except KeyError:
                pass

        # Calculate current portfolio value for risk checks
        # Use symbol-specific portfolio if provided, otherwise use global portfolio
        if symbol_portfolio is not None:
            portfolio_value = symbol_portfolio.get_portfolio_value(current_prices)
        else:
            portfolio_value = self._get_portfolio_value(positions, current_prices)

        # Create a working copy of positions to track changes during this batch
        # This ensures we account for fills within the same batch when checking limits
        working_positions = {k: Position(v.symbol, v.qty, v.avg_price, v.unrealized_pnl, v.realized_pnl)
                           for k, v in positions.items()}

        for symbol, order in orders.items():
            if symbol not in market_data:
                self.logger.warning(f"No market data for symbol: {symbol}")
                continue

            symbol_data = market_data[symbol]
            current_position = working_positions.get(symbol, Position(symbol, 0.0, 0.0))

            try:
                fill = self._execute_single_order(
                    symbol, order, symbol_data, current_time, current_position,
                    working_positions, current_prices, portfolio_value
                )
                if fill is not None:
                    fills.append(fill)

                    # Update working positions to reflect this fill for subsequent orders
                    if symbol not in working_positions:
                        working_positions[symbol] = Position(symbol, 0.0, 0.0)

                    working_positions[symbol].qty += fill.filled_qty

            except Exception as e:
                self.logger.error(f"Error executing order for {symbol}: {e}")

        return fills

    def _execute_single_order(
        self,
        symbol: str,
        order: Dict[str, Any],  # Allow both dict and OrderDict
        market_data: TimeSeriesData,
        current_time: pd.Timestamp,
        position: Position,
        all_positions: Dict[str, Position],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> Optional[Fill]:
        """Execute a single order with risk management"""

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

        current_price = current_bar['close']

        # Determine target quantity based on order type
        if action == TARGET_WEIGHT_ACTION:
            weight = order.get('weight')
            if weight is None:
                self.logger.warning(f"Target weight order for {symbol} missing weight")
                return None

            # Use actual portfolio value instead of hardcoded
            target_qty = self._weight_to_quantity(weight, current_price, portfolio_value)
            qty_to_trade = target_qty - position.qty

            # Log rebalancing details for debugging (disabled to avoid spam)
            if False and abs(qty_to_trade) > 0.01:
                position_value = position.qty * current_price if position.qty != 0 else 0
                current_weight = (position_value / portfolio_value) if portfolio_value > 0 else 0
                target_value = weight * portfolio_value
                weight_drift = current_weight - weight

                self.logger.info(
                    f"📊 REBALANCE {symbol} @ {current_time.strftime('%Y-%m-%d')}: "
                    f"Portfolio=${portfolio_value:,.0f} | "
                    f"Current: {position.qty:.2f} sh (${position_value:,.0f}, {current_weight:.2%}) | "
                    f"Target: {target_qty:.2f} sh (${target_value:,.0f}, {weight:.2%}) | "
                    f"Drift: {weight_drift:+.2%} | "
                    f"Trade: {qty_to_trade:+.2f} sh"
                )

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

        # RISK MANAGEMENT CHECKS
        # Calculate proposed position after this trade
        proposed_qty = position.qty + qty_to_trade

        # Check position-level risk limits (per-symbol max size)
        if not self._check_position_risk_limits(
            symbol, proposed_qty, current_price, portfolio_value
        ):
            return None

        # Check global portfolio risk limits (aggregate exposure)
        if not self._check_global_risk_limits(
            symbol, proposed_qty, current_price, all_positions, current_prices, portfolio_value
        ):
            return None

        # Check if order can be executed (limit order logic)
        if not self.can_execute(order, symbol, market_data, current_time):
            return None

        # Determine execution price
        execution_price = self._get_execution_price(
            order, current_bar, qty_to_trade
        )

        # Calculate slippage (returns total $ amount)
        slippage = self.calculate_slippage(
            symbol, qty_to_trade, execution_price, market_data
        )

        # Convert slippage to per-share amount and adjust execution price
        slippage_per_share = slippage / abs(qty_to_trade) if qty_to_trade != 0 else 0
        if qty_to_trade > 0:  # buying
            final_price = execution_price + slippage_per_share
        else:  # selling
            final_price = execution_price - slippage_per_share

        # Calculate commission
        commission = self.calculate_commission(symbol, abs(qty_to_trade), final_price)

        # Create fill with merged metadata
        # Start with execution metadata
        fill_meta = {
            "order_type": order.get('order_type', 'market'),
            "original_action": action,
            "execution_method": "mock"
        }

        # Merge order metadata (strategy reasons, signals, etc.)
        if 'meta' in order and order['meta'] is not None:
            fill_meta.update(order['meta'])

        fill = Fill(
            symbol=symbol,
            filled_qty=qty_to_trade,
            fill_price=final_price,
            commission=commission,
            slippage=slippage,
            timestamp=current_time,
            order_id=self._generate_order_id(),
            side="buy" if qty_to_trade > 0 else "sell",
            meta=fill_meta
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
        return target_value / price if price > 0 else 0.0

    def _get_portfolio_value(
        self,
        positions: Dict[str, Position],
        current_prices: Dict[str, float]
    ) -> float:
        """Calculate total portfolio value from positions"""
        if self.portfolio_manager is not None:
            return self.portfolio_manager.get_portfolio_value(current_prices)

        # Fallback calculation if no portfolio manager
        total_value = 0.0
        for symbol, position in positions.items():
            if symbol in current_prices:
                total_value += position.qty * current_prices[symbol]
        return total_value

    def _check_position_risk_limits(
        self,
        symbol: str,
        proposed_qty: float,
        current_price: float,
        portfolio_value: float
    ) -> bool:
        """
        Check if proposed position violates per-symbol risk limits

        Args:
            symbol: Symbol being traded
            proposed_qty: Proposed position size (after trade)
            current_price: Current price
            portfolio_value: Total portfolio value

        Returns:
            True if within limits, False if violated
        """
        # Position size limits are disabled at portfolio level
        # Strategies control their own position sizing via target_weight
        return True

    def _check_global_risk_limits(
        self,
        symbol: str,
        proposed_qty: float,
        current_price: float,
        all_positions: Dict[str, Position],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> bool:
        """
        Check if proposed trade violates global portfolio risk limits

        Checks aggregate exposure across all positions to ensure diversification.

        Args:
            symbol: Symbol being traded
            proposed_qty: Proposed position size for this symbol
            current_price: Current price of symbol
            all_positions: All current positions
            current_prices: Current prices for all symbols
            portfolio_value: Total portfolio value

        Returns:
            True if within limits, False if violated
        """
        if self.portfolio_manager is None or portfolio_value <= 0:
            return True

        # Calculate total gross exposure after this trade
        total_exposure = 0.0

        # Add exposure from all existing positions
        for pos_symbol, position in all_positions.items():
            if pos_symbol == symbol:
                # Use proposed quantity for the symbol being traded
                qty = proposed_qty
            else:
                # Use current quantity for other positions
                qty = position.qty

            if pos_symbol in current_prices:
                position_value = abs(qty * current_prices[pos_symbol])
                total_exposure += position_value

        # Calculate gross exposure ratio
        exposure_ratio = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Check against max_leverage if set
        max_leverage = self.portfolio_manager.config.max_leverage
        if max_leverage is not None and max_leverage > 0:
            if exposure_ratio > max_leverage:
                self.logger.debug(
                    f"Global leverage limit exceeded: "
                    f"Total exposure {exposure_ratio:.2f}x > Max {max_leverage:.2f}x "
                    f"(${total_exposure:,.0f} / ${portfolio_value:,.0f}). "
                    f"Order for {symbol} rejected."
                )
                return False

        # Additional check: Warn if getting too concentrated (even without max_leverage set)
        # If no max_leverage, warn at 1.5x (150% gross exposure)
        if max_leverage is None and exposure_ratio > 1.5:
            self.logger.debug(
                f"High portfolio concentration: {exposure_ratio:.2f}x gross exposure "
                f"(${total_exposure:,.0f} / ${portfolio_value:,.0f})"
            )
            # Don't reject, just warn

        return True

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