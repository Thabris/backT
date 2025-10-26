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

    def __init__(self, config: BacktestConfig, allocated_capital: Optional[float] = None, symbol: Optional[str] = None):
        """
        Initialize portfolio manager

        Args:
            config: Backtest configuration
            allocated_capital: Capital allocated to this portfolio (for symbol-specific portfolios)
            symbol: Symbol this portfolio is managing (for independent symbol execution)
        """
        self.config = config
        self.allocated_capital = allocated_capital  # Capital allocated to this portfolio
        self.managed_symbol = symbol  # Symbol this portfolio manages (None = multi-symbol)
        self.initial_capital = allocated_capital if allocated_capital is not None else config.initial_capital
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_history: List[Dict] = []
        self.per_symbol_equity_history: Dict[str, List[Dict]] = {}  # Track per-symbol equity
        self.symbol_allocated_capital: Dict[str, float] = {}  # Track allocated capital per symbol
        self.universe_size: int = 0  # Number of symbols in universe

    def initialize_symbol_allocations(self, symbols: List[str]) -> None:
        """
        Initialize allocated capital for each symbol in the universe.
        For independent symbol trading, each gets equal allocation (1/N).
        """
        self.universe_size = len(symbols)
        if self.universe_size > 0:
            allocation_per_symbol = self.initial_capital / self.universe_size
            for symbol in symbols:
                self.symbol_allocated_capital[symbol] = allocation_per_symbol

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

        # Track per-symbol equity
        self._record_per_symbol_equity(current_prices, timestamp)

        return snapshot

    def _record_per_symbol_equity(self, current_prices: Dict[str, float], timestamp: pd.Timestamp) -> None:
        """Record equity snapshot for each symbol"""
        # For symbol-specific portfolios, only track the managed symbol
        if self.managed_symbol is not None:
            symbols_to_track = {self.managed_symbol}
        else:
            # Track all symbols that have allocated capital (not just active positions)
            symbols_to_track = set(self.symbol_allocated_capital.keys()) | set(self.positions.keys())

        for symbol in symbols_to_track:
            if symbol not in self.per_symbol_equity_history:
                self.per_symbol_equity_history[symbol] = []

            # Get allocated capital for this symbol
            if self.managed_symbol is not None:
                # Symbol-specific portfolio: use the portfolio's allocated capital
                allocated_capital = self.allocated_capital or self.initial_capital
            else:
                # Multi-symbol portfolio: use per-symbol allocation
                allocated_capital = self.symbol_allocated_capital.get(
                    symbol,
                    self.initial_capital / max(1, self.universe_size) if self.universe_size > 0 else self.initial_capital
                )

            # Get position if exists
            position = self.positions.get(symbol)

            if position and position.qty != 0 and symbol in current_prices:
                # Active position
                position_value = current_prices[symbol] * position.qty
                unrealized_pnl = position.unrealized_pnl
                realized_pnl = position.realized_pnl
                total_pnl = unrealized_pnl + realized_pnl
            elif position:
                # Closed position (qty=0 but has realized PnL)
                position_value = 0.0
                unrealized_pnl = 0.0
                realized_pnl = position.realized_pnl
                total_pnl = realized_pnl
            else:
                # No position yet
                position_value = 0.0
                unrealized_pnl = 0.0
                realized_pnl = 0.0
                total_pnl = 0.0

            # Total equity for this symbol = allocated capital + total PnL
            # This maintains continuity when position is flat
            total_equity = allocated_capital + total_pnl

            symbol_snapshot = {
                'timestamp': timestamp,
                'position_value': position_value,
                'qty': position.qty if position else 0.0,
                'price': current_prices.get(symbol, position.avg_price if position else 0.0),
                'avg_price': position.avg_price if position else 0.0,
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_pnl': total_pnl,
                'total_equity': total_equity,
                'allocated_capital': allocated_capital
            }

            self.per_symbol_equity_history[symbol].append(symbol_snapshot)

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

    def get_per_symbol_equity_curves(self) -> Dict[str, pd.DataFrame]:
        """Get per-symbol equity curves as DataFrames"""
        symbol_curves = {}

        for symbol, history in self.per_symbol_equity_history.items():
            if history:
                df = pd.DataFrame(history)
                df.set_index('timestamp', inplace=True)
                symbol_curves[symbol] = df

        return symbol_curves

    def reset(self) -> None:
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.equity_history.clear()
        self.per_symbol_equity_history.clear()

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