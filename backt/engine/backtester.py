"""
Main backtesting engine for BackT

Orchestrates the event-driven simulation, coordinating data feeds,
strategy signals, execution, and portfolio updates.
"""

import time
from datetime import datetime
from typing import Dict, List, Union, Callable, Optional, Any
import pandas as pd

from ..utils.types import StrategyFunction, BacktestResult, TimeSeriesData
from ..utils.config import BacktestConfig
from ..data.loaders import DataLoader, YahooDataLoader
from ..data.mock_data import MockDataLoader
from ..execution.mock_execution import MockExecutionEngine
from ..portfolio.portfolio_manager import PortfolioManager
from ..risk.metrics import MetricsEngine
from ..reporting.trade_logger import TradeLogger
from ..utils.logging_config import LoggerMixin, setup_logging


class Backtester(LoggerMixin):
    """Main backtesting engine"""

    def __init__(
        self,
        config: BacktestConfig,
        data_loader: Optional[DataLoader] = None,
        execution_engine: Optional["ExecutionEngine"] = None,
        portfolio_manager: Optional[PortfolioManager] = None,
        metrics_engine: Optional[MetricsEngine] = None
    ):
        """
        Initialize backtester

        Args:
            config: Backtest configuration
            data_loader: Data loading interface (default: YahooDataLoader)
            execution_engine: Execution engine (default: MockExecutionEngine)
            portfolio_manager: Portfolio manager (default: PortfolioManager)
            metrics_engine: Metrics calculation engine (default: MetricsEngine)
        """
        self.config = config

        # Set up logging
        if config.verbose:
            setup_logging("INFO")
        else:
            setup_logging("WARNING")

        # Initialize components
        if config.use_mock_data:
            self.data_loader = data_loader or MockDataLoader(
                scenario=config.mock_scenario,
                seed=config.mock_seed
            )
            self.logger.info(f"Using mock data loader with scenario: {config.mock_scenario}")
        else:
            self.data_loader = data_loader or YahooDataLoader()
            self.logger.info("Using Yahoo Finance data loader")

        # Initialize portfolio first (needed by execution engine)
        self.portfolio = portfolio_manager or PortfolioManager(config)

        # Initialize execution engine with portfolio reference for risk checks
        self.execution_engine = execution_engine or MockExecutionEngine(
            config.execution,
            portfolio_manager=self.portfolio
        )

        self.metrics_engine = metrics_engine or MetricsEngine(config)
        self.trade_logger = TradeLogger()

        # Runtime state
        self.market_data: Dict[str, TimeSeriesData] = {}
        self.strategy_context: Dict[str, Any] = {}

    def run(
        self,
        strategy: StrategyFunction,
        universe: Union[str, List[str]],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> BacktestResult:
        """
        Run backtest

        Args:
            strategy: Strategy function to test
            universe: Symbol or list of symbols to trade
            strategy_params: Parameters to pass to strategy

        Returns:
            BacktestResult containing all backtest outputs
        """
        start_time = datetime.now()
        self.logger.info("Starting backtest")

        try:
            # Prepare universe
            if isinstance(universe, str):
                universe = [universe]

            # Load data
            self.logger.info(f"Loading data for {len(universe)} symbols")
            self.market_data = self._load_data(universe)

            if not self.market_data:
                raise ValueError("No data loaded for any symbols")

            # Get date range from data
            all_dates = set()
            for symbol_data in self.market_data.values():
                all_dates.update(symbol_data.index)

            date_range = sorted(all_dates)
            self.logger.info(f"Backtesting {len(date_range)} periods from {date_range[0]} to {date_range[-1]}")

            # Initialize strategy parameters
            if strategy_params is None:
                strategy_params = {}

            # Reset portfolio
            self.portfolio.reset()
            self.strategy_context.clear()

            # Run simulation
            self._run_simulation(strategy, universe, strategy_params, date_range)

            # Calculate final metrics
            equity_curve = self.portfolio.get_equity_curve()
            trades_df = self.trade_logger.get_trades_dataframe()

            performance_metrics = {}
            if not equity_curve.empty:
                performance_metrics = self.metrics_engine.calculate_metrics(
                    equity_curve, trades_df
                )

            # Get per-symbol data
            per_symbol_equity_curves = self.portfolio.get_per_symbol_equity_curves()
            per_symbol_metrics = None
            returns_correlation_matrix = None

            if per_symbol_equity_curves:
                # Calculate per-symbol metrics
                per_symbol_metrics = self.metrics_engine.calculate_per_symbol_metrics(
                    per_symbol_equity_curves, trades_df
                )

                # Calculate correlation matrix
                if len(per_symbol_equity_curves) > 1:
                    returns_correlation_matrix = self.metrics_engine.calculate_returns_correlation_matrix(
                        per_symbol_equity_curves
                    )

            # Create result
            end_time = datetime.now()
            runtime = (end_time - start_time).total_seconds()

            result = BacktestResult(
                trades=trades_df,
                equity_curve=equity_curve,
                positions_history=self._get_positions_history(),
                performance_metrics=performance_metrics,
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                total_runtime_seconds=runtime,
                per_symbol_equity_curves=per_symbol_equity_curves,
                per_symbol_metrics=per_symbol_metrics,
                returns_correlation_matrix=returns_correlation_matrix
            )

            self.logger.info(f"Backtest completed in {runtime:.2f} seconds")
            return result

        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise

    def _load_data(self, universe: List[str]) -> Dict[str, TimeSeriesData]:
        """Load and prepare market data"""
        try:
            data = self.data_loader.load(
                symbols=universe,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

            if isinstance(data, pd.DataFrame):
                # Single symbol
                return {universe[0]: data}
            else:
                # Multiple symbols
                return data

        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise

    def _run_simulation(
        self,
        strategy: StrategyFunction,
        universe: List[str],
        strategy_params: Dict[str, Any],
        date_range: List[pd.Timestamp]
    ) -> None:
        """Run the main simulation loop"""

        for i, current_time in enumerate(date_range):
            try:
                # Build market snapshot up to current time
                market_snapshot = self._build_market_snapshot(current_time)

                if not market_snapshot:
                    continue

                # Get current prices for portfolio updates
                current_prices = {
                    symbol: data.loc[current_time, 'close']
                    for symbol, data in market_snapshot.items()
                    if current_time in data.index
                }

                # Update portfolio with current prices
                self.portfolio.update_positions(current_prices, current_time)

                # Generate strategy signals
                orders = strategy(
                    market_data=market_snapshot,
                    current_time=current_time,
                    positions=self.portfolio.positions.copy(),
                    context=self.strategy_context,
                    params=strategy_params
                )

                # Process orders
                if orders:
                    fills = self.execution_engine.execute(
                        orders=orders,
                        market_data=market_snapshot,
                        current_time=current_time,
                        positions=self.portfolio.positions,
                        context=self.strategy_context
                    )

                    # Process fills
                    for fill in fills:
                        self.portfolio.process_fill(fill, current_prices)
                        self.trade_logger.log_fill(fill)

                # Record equity snapshot
                self.portfolio.get_equity_snapshot(current_prices, current_time)

                # Progress logging
                if i % 100 == 0 or i == len(date_range) - 1:
                    progress = (i + 1) / len(date_range) * 100
                    equity = self.portfolio.get_portfolio_value(current_prices)
                    self.logger.info(
                        f"Progress: {progress:.1f}% - {current_time.date()} - "
                        f"Equity: ${equity:,.2f}"
                    )

            except Exception as e:
                self.logger.error(f"Error processing {current_time}: {e}")
                continue

    def _build_market_snapshot(self, current_time: pd.Timestamp) -> Dict[str, TimeSeriesData]:
        """Build market data snapshot up to current time"""
        snapshot = {}

        for symbol, data in self.market_data.items():
            # Get data up to current time
            available_data = data[data.index <= current_time]

            if not available_data.empty:
                # Apply lookback window
                if len(available_data) > self.config.lookback_window:
                    available_data = available_data.tail(self.config.lookback_window)

                snapshot[symbol] = available_data

        return snapshot

    def _get_positions_history(self) -> pd.DataFrame:
        """Get historical positions data"""
        # This would ideally be tracked during simulation
        # For now, return empty DataFrame
        return pd.DataFrame()

    def validate_strategy(self, strategy: StrategyFunction, test_data: Dict[str, Any]) -> bool:
        """
        Validate that a strategy function has the correct signature

        Args:
            strategy: Strategy function to validate
            test_data: Test data to use for validation

        Returns:
            True if strategy is valid
        """
        try:
            # Create minimal test inputs
            market_data = test_data.get('market_data', {})
            current_time = pd.Timestamp.now()
            positions = {}
            context = {}
            params = {}

            # Call strategy with test inputs
            result = strategy(market_data, current_time, positions, context, params)

            # Validate return type
            if not isinstance(result, dict):
                self.logger.error("Strategy must return a dictionary")
                return False

            # Validate order format
            for symbol, order in result.items():
                if not isinstance(order, dict):
                    self.logger.error(f"Order for {symbol} must be a dictionary")
                    return False

                if 'action' not in order:
                    self.logger.error(f"Order for {symbol} missing 'action' field")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Strategy validation failed: {e}")
            return False