"""
BackT - A Professional Trading Backtesting Framework

A modular, extensible trading backtester designed for researching and validating
trading strategies across various asset classes and timeframes.

Key Features:
- Event-driven backtesting engine
- Standardized data contracts (pandas-based)
- Pluggable execution models with realistic market simulation
- Comprehensive risk and performance analytics
- Multi-asset support
- Flexible strategy API

Usage:
    from backt import Backtester, BacktestConfig
    from backt.data import YahooDataLoader

    # Create backtester
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-01-01',
        initial_capital=100000
    )

    backtester = Backtester(config)
    result = backtester.run(my_strategy, ['AAPL', 'MSFT'])
"""

from backt.engine.backtester import Backtester
from backt.utils.config import BacktestConfig
from backt.data.loaders import YahooDataLoader, CSVDataLoader
from backt.execution.mock_execution import MockExecutionEngine
from backt.portfolio.portfolio_manager import PortfolioManager
from backt.risk.metrics import MetricsEngine
from backt.utils.types import Position, Fill, BacktestResult

__version__ = "0.1.0"
__author__ = "BackT Development Team"

__all__ = [
    "Backtester",
    "BacktestConfig",
    "YahooDataLoader",
    "CSVDataLoader",
    "MockExecutionEngine",
    "PortfolioManager",
    "MetricsEngine",
    "Position",
    "Fill",
    "BacktestResult"
]