"""
Portfolio management for BackT

Manages overall portfolio state, cash, equity calculations,
and position-level operations.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..utils.types import Position, Fill
from ..utils.config import BacktestConfig
from ..utils.logging_config import LoggerMixin


class PortfolioManager(LoggerMixin):
    """Manages portfolio state and calculations"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.initial_capital = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[Dict] = []

    def process_fill(self, fill: Fill, current_prices: Dict[str, float]) -> None:
        """Process a trade fill and update portfolio state"""
        symbol = fill.symbol

        # Initialize position if it doesn't exist
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0.0, 0.0)

        position = self.positions[symbol]
        trade_value = fill.filled_qty * fill.fill_price
        total_cost = trade_value + fill.commission

        # Update cash
        self.cash -= total_cost

        # Update position
        if position.qty == 0:
            # New position
            position.qty = fill.filled_qty
            position.avg_price = fill.fill_price
        else:
            # Existing position
            if (position.qty > 0 and fill.filled_qty > 0) or \
               (position.qty < 0 and fill.filled_qty < 0):
                # Adding to position
                total_cost_old = position.qty * position.avg_price
                total_qty = position.qty + fill.filled_qty
                position.avg_price = (total_cost_old + trade_value) / total_qty
                position.qty = total_qty
            else:
                # Reducing or reversing position
                if abs(fill.filled_qty) >= abs(position.qty):
                    # Close or reverse position
                    realized_pnl = (fill.fill_price - position.avg_price) * position.qty
                    position.realized_pnl += realized_pnl

                    # Handle reversal
                    remaining_qty = fill.filled_qty + position.qty
                    if abs(remaining_qty) > 1e-8:
                        position.qty = remaining_qty
                        position.avg_price = fill.fill_price
                    else:
                        position.qty = 0.0
                        position.avg_price = 0.0
                else:
                    # Partial close
                    close_qty = -fill.filled_qty if fill.filled_qty < 0 else fill.filled_qty
                    realized_pnl = (fill.fill_price - position.avg_price) * close_qty
                    position.realized_pnl += realized_pnl
                    position.qty += fill.filled_qty

        # Clean up zero positions
        if abs(position.qty) < 1e-8:
            position.qty = 0.0

        # Update unrealized PnL
        if symbol in current_prices and position.qty != 0:
            position.unrealized_pnl = (current_prices[symbol] - position.avg_price) * position.qty

        position.last_update = fill.timestamp

        self.logger.debug(f"Processed fill: {fill.side} {abs(fill.filled_qty)} {symbol} at ${fill.fill_price:.2f}")

    def update_positions(self, current_prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        """Update unrealized PnL for all positions"""
        for symbol, position in self.positions.items():
            if symbol in current_prices and position.qty != 0:
                current_price = current_prices[symbol]
                position.unrealized_pnl = (current_price - position.avg_price) * position.qty
                position.last_update = timestamp

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            current_prices.get(symbol, pos.avg_price) * pos.qty
            for symbol, pos in self.positions.items()
            if pos.qty != 0
        )
        return self.cash + positions_value

    def get_positions_value(self, current_prices: Dict[str, float]) -> float:
        """Get total value of all positions"""
        return sum(
            current_prices.get(symbol, pos.avg_price) * pos.qty
            for symbol, pos in self.positions.items()
            if pos.qty != 0
        )

    def get_equity_snapshot(self, current_prices: Dict[str, float], timestamp: pd.Timestamp) -> Dict:
        """Get current equity snapshot"""
        positions_value = self.get_positions_value(current_prices)
        total_equity = self.cash + positions_value

        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())

        snapshot = {
            'timestamp': timestamp,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_equity': total_equity,
            'unrealized_pnl': total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'total_pnl': total_unrealized_pnl + total_realized_pnl,
            'total_return': (total_equity - self.initial_capital) / self.initial_capital
        }

        self.equity_history.append(snapshot)
        return snapshot

    def get_position_summary(self) -> Dict[str, Dict]:
        """Get summary of all positions"""
        summary = {}
        for symbol, position in self.positions.items():
            if abs(position.qty) > 1e-8:  # Only include non-zero positions
                summary[symbol] = {
                    'qty': position.qty,
                    'avg_price': position.avg_price,
                    'market_value': position.market_value,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'side': 'long' if position.qty > 0 else 'short'
                }
        return summary

    def check_risk_limits(self, symbol: str, proposed_qty: float, current_price: float) -> bool:
        """Check if proposed trade violates risk limits"""
        if not self.config.max_position_size:
            return True

        total_equity = self.get_portfolio_value({symbol: current_price})
        proposed_value = abs(proposed_qty * current_price)
        max_position_value = self.config.max_position_size * total_equity

        if proposed_value > max_position_value:
            self.logger.warning(
                f"Position size limit exceeded for {symbol}: "
                f"${proposed_value:.2f} > ${max_position_value:.2f}"
            )
            return False

        return True

    def can_afford_trade(self, symbol: str, qty: float, price: float, commission: float) -> bool:
        """Check if portfolio has sufficient cash for trade"""
        if qty > 0:  # Buying
            required_cash = qty * price + commission
            if required_cash > self.cash:
                self.logger.warning(
                    f"Insufficient cash for {symbol} purchase: "
                    f"${required_cash:.2f} required, ${self.cash:.2f} available"
                )
                return False
        elif not self.config.allow_short:
            # Selling - check if we have the position
            current_position = self.positions.get(symbol, Position(symbol, 0.0, 0.0))
            if abs(qty) > current_position.qty + 1e-8:  # Allow for small rounding errors
                self.logger.warning(
                    f"Insufficient shares to sell {symbol}: "
                    f"{abs(qty)} requested, {current_position.qty} available"
                )
                return False

        return True

    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.equity_history)
        df.set_index('timestamp', inplace=True)
        return df

    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.equity_history.clear()

    def get_leverage(self, current_prices: Dict[str, float]) -> float:
        """Calculate current leverage"""
        total_equity = self.get_portfolio_value(current_prices)
        if total_equity <= 0:
            return 0.0

        gross_exposure = sum(
            abs(current_prices.get(symbol, pos.avg_price) * pos.qty)
            for symbol, pos in self.positions.items()
        )

        return gross_exposure / total_equity