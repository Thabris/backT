"""
Configuration classes for BackT

Defines configuration objects that control backtesting behavior,
execution settings, and other framework parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd


@dataclass
class ExecutionConfig:
    """Configuration for execution simulation"""
    spread: float = 0.01  # bid-ask spread in price units or percentage
    slippage_pct: float = 0.0005  # slippage as proportion of trade size
    commission_per_share: float = 0.001  # commission per share
    commission_per_trade: float = 0.0  # flat commission per trade
    commission_pct: float = 0.0  # commission as percentage of trade value
    partial_fills: bool = False  # whether to simulate partial fills

    def __post_init__(self):
        """Validate configuration"""
        if self.spread < 0:
            raise ValueError("Spread must be non-negative")
        if self.slippage_pct < 0:
            raise ValueError("Slippage percentage must be non-negative")


@dataclass
class BacktestConfig:
    """Main configuration for backtesting"""

    # Time period
    start_date: str  # 'YYYY-MM-DD' format
    end_date: str    # 'YYYY-MM-DD' format

    # Capital settings
    initial_capital: float = 100000.0

    # Data settings
    data_frequency: str = "1D"  # pandas frequency string
    timezone: str = "UTC"

    # Trading rules
    allow_short: bool = False
    max_leverage: float = 1.0
    position_size_mode: str = "shares"  # "shares", "notional", "weight"

    # Risk management
    max_position_size: Optional[float] = None  # max position as fraction of portfolio
    max_positions: Optional[int] = None  # max number of concurrent positions

    # Execution settings
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # Technical settings
    lookback_window: int = 252  # number of periods to provide to strategy
    benchmark: Optional[str] = None  # benchmark symbol for comparison
    risk_free_rate: float = 0.0  # annual risk-free rate for metrics

    # Output settings
    save_trades: bool = True
    save_positions: bool = True
    save_equity_curve: bool = True
    output_dir: Optional[str] = None

    # Mock data settings
    use_mock_data: bool = False  # Use synthetic data instead of real market data
    mock_scenario: str = "normal"  # Mock market scenario: 'normal', 'bull', 'bear', 'volatile'
    mock_seed: Optional[int] = 42  # Random seed for reproducible mock data

    # Advanced settings
    random_seed: Optional[int] = None
    verbose: bool = True

    def __post_init__(self):
        """Validate and process configuration"""
        # Validate dates
        try:
            self.start_datetime = pd.to_datetime(self.start_date)
            self.end_datetime = pd.to_datetime(self.end_date)
        except Exception as e:
            raise ValueError(f"Invalid date format: {e}")

        if self.start_datetime >= self.end_datetime:
            raise ValueError("Start date must be before end date")

        # Validate capital
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")

        # Validate leverage
        if self.max_leverage is not None and self.max_leverage < 1.0:
            raise ValueError("Max leverage must be >= 1.0 or None")

        # Validate position size mode
        valid_modes = {"shares", "notional", "weight"}
        if self.position_size_mode not in valid_modes:
            raise ValueError(f"Position size mode must be one of {valid_modes}")

        # Validate frequency
        try:
            pd.Timedelta(self.data_frequency)
        except Exception:
            raise ValueError(f"Invalid data frequency: {self.data_frequency}")

    @property
    def period_days(self) -> int:
        """Number of calendar days in the backtest period"""
        return (self.end_datetime - self.start_datetime).days

    @property
    def trading_period_years(self) -> float:
        """Approximate trading period in years (252 trading days per year)"""
        return self.period_days / 365.25

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "data_frequency": self.data_frequency,
            "timezone": self.timezone,
            "allow_short": self.allow_short,
            "max_leverage": self.max_leverage,
            "position_size_mode": self.position_size_mode,
            "max_position_size": self.max_position_size,
            "max_positions": self.max_positions,
            "execution": {
                "spread": self.execution.spread,
                "slippage_pct": self.execution.slippage_pct,
                "commission_per_share": self.execution.commission_per_share,
                "commission_per_trade": self.execution.commission_per_trade,
                "commission_pct": self.execution.commission_pct,
                "partial_fills": self.execution.partial_fills
            },
            "lookback_window": self.lookback_window,
            "benchmark": self.benchmark,
            "risk_free_rate": self.risk_free_rate,
            "random_seed": self.random_seed,
            "verbose": self.verbose
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BacktestConfig":
        """Create config from dictionary"""
        execution_dict = config_dict.pop("execution", {})
        execution = ExecutionConfig(**execution_dict)
        return cls(execution=execution, **config_dict)