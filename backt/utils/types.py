"""
Type definitions for BackT

Defines all the core data structures and type hints used throughout
the backtesting framework.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd


@dataclass
class Position:
    """Represents a position in a security"""
    symbol: str
    qty: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    side: str = "long"  # "long" or "short"
    last_update: Optional[pd.Timestamp] = None

    @property
    def market_value(self) -> float:
        """Current market value of the position"""
        return self.qty * self.avg_price

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    @property
    def is_short(self) -> bool:
        return self.qty < 0

    @property
    def is_flat(self) -> bool:
        return self.qty == 0


@dataclass
class Fill:
    """Represents an executed trade fill"""
    symbol: str
    filled_qty: float
    fill_price: float
    commission: float
    slippage: float
    timestamp: pd.Timestamp
    order_id: str
    side: str = "buy"  # "buy" or "sell"
    meta: Optional[Dict[str, Any]] = None

    @property
    def gross_amount(self) -> float:
        """Gross trade amount before fees"""
        return abs(self.filled_qty * self.fill_price)

    @property
    def net_amount(self) -> float:
        """Net trade amount after fees"""
        return self.gross_amount - self.commission


@dataclass
class OrderDict:
    """Type hint for order dictionaries returned by strategies"""
    action: str  # 'buy', 'sell', 'close', 'hold', 'target_weight'
    size: Optional[float] = None  # absolute shares or notional
    weight: Optional[float] = None  # target weight (0-1)
    order_type: str = "market"  # 'market', 'limit'
    limit_price: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class BacktestResult:
    """Container for backtest results"""
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    positions_history: pd.DataFrame
    performance_metrics: Dict[str, float]
    config: "BacktestConfig"
    start_time: datetime
    end_time: datetime
    total_runtime_seconds: float
    per_symbol_equity_curves: Optional[Dict[str, pd.DataFrame]] = None
    per_symbol_metrics: Optional[Dict[str, Dict[str, float]]] = None
    returns_correlation_matrix: Optional[pd.DataFrame] = None

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the backtest results"""
        summary_dict = {
            "performance": self.performance_metrics,
            "total_trades": len(self.trades),
            "runtime_seconds": self.total_runtime_seconds,
            "period": {
                "start": self.equity_curve.index[0],
                "end": self.equity_curve.index[-1]
            }
        }

        # Add per-symbol summary if available
        if self.per_symbol_metrics:
            summary_dict["symbols_tracked"] = list(self.per_symbol_metrics.keys())
            summary_dict["num_symbols"] = len(self.per_symbol_metrics)

        return summary_dict


# Type aliases for strategy functions
StrategyFunction = Callable[
    [
        Union[Dict[str, pd.DataFrame], pd.DataFrame],  # market_data
        pd.Timestamp,  # current_time
        Dict[str, Position],  # positions
        Dict[str, Any],  # context
        Dict[str, Any]  # params
    ],
    Dict[str, OrderDict]
]

# Market data types
MarketData = Union[pd.DataFrame, Dict[str, pd.DataFrame]]
TimeSeriesData = pd.DataFrame