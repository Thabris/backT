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

        # Initialize portfolios - can be single or per-symbol
        # Will be set up properly in run() when we know the universe
        self.portfolio = portfolio_manager or PortfolioManager(config)
        self.symbol_portfolios: Dict[str, PortfolioManager] = {}  # Per-symbol portfolios
        self.independent_execution = True  # Feature flag for independent symbol execution

        # Initialize execution engine with portfolio reference for risk checks
        self.execution_engine = execution_engine or MockExecutionEngine(
            config.execution,
            portfolio_manager=self.portfolio
        )

        # Initialize metrics engine with Numba JIT enabled by default for performance
        self.metrics_engine = metrics_engine or MetricsEngine(config, use_numba=True)
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
        if self.config.verbose:
            self.logger.info("Starting backtest")

        try:
            # Prepare universe
            if isinstance(universe, str):
                universe = [universe]

            # Load data
            if self.config.verbose:
                self.logger.info(f"Loading data for {len(universe)} symbols")
            self.market_data = self._load_data(universe)

            if self.market_data is None or len(self.market_data) == 0:
                raise ValueError("No data loaded for any symbols")

            # Get date range from data
            all_dates = set()
            for symbol_data in self.market_data.values():
                all_dates.update(symbol_data.index)

            date_range = sorted(all_dates)
            if self.config.verbose:
                self.logger.info(f"Backtesting {len(date_range)} periods from {date_range[0]} to {date_range[-1]}")

            # Initialize strategy parameters
            if strategy_params is None:
                strategy_params = {}

            # Inject initial_capital into strategy params (for equal allocation)
            if 'initial_capital' not in strategy_params:
                strategy_params['initial_capital'] = self.config.initial_capital

            # Initialize portfolios based on execution mode
            if self.independent_execution:
                # Create separate portfolio for each symbol with equal allocation
                allocation_per_symbol = self.config.initial_capital / len(universe)
                self.symbol_portfolios = {}

                for symbol in universe:
                    self.symbol_portfolios[symbol] = PortfolioManager(
                        self.config,
                        allocated_capital=allocation_per_symbol,
                        symbol=symbol
                    )

                self.logger.info(
                    f"Independent execution mode: {len(universe)} symbols, "
                    f"${allocation_per_symbol:,.2f} allocated per symbol"
                )
            else:
                # Legacy mode: single shared portfolio
                self.portfolio.reset()
                self.portfolio.initialize_symbol_allocations(universe)

            self.strategy_context.clear()

            # Run simulation
            self._run_simulation(strategy, universe, strategy_params, date_range)

            # Calculate final metrics
            if self.independent_execution:
                # Aggregate results from symbol portfolios
                equity_curve = self._aggregate_equity_curves()
                per_symbol_equity_curves = self._get_symbol_equity_curves()
            else:
                # Legacy mode: use single portfolio
                equity_curve = self.portfolio.get_equity_curve()
                per_symbol_equity_curves = self.portfolio.get_per_symbol_equity_curves()

            trades_df = self.trade_logger.get_trades_dataframe()

            performance_metrics = {}
            if not equity_curve.empty:
                performance_metrics = self.metrics_engine.calculate_metrics(
                    equity_curve, trades_df
                )

            # Get per-symbol data
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

            if self.config.verbose:
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

                if self.independent_execution:
                    # Independent execution: each symbol has its own portfolio
                    # Aggregate positions from all symbol portfolios
                    all_positions = {}
                    for symbol, portfolio in self.symbol_portfolios.items():
                        portfolio.update_positions({symbol: current_prices.get(symbol, 0)}, current_time)
                        all_positions.update(portfolio.positions)

                    # Generate strategy signals with aggregated positions
                    orders = strategy(
                        market_data=market_snapshot,
                        current_time=current_time,
                        positions=all_positions.copy(),
                        context=self.strategy_context,
                        params=strategy_params
                    )

                    # Process orders per symbol
                    if orders:
                        for symbol, order in orders.items():
                            if symbol not in self.symbol_portfolios:
                                continue

                            portfolio = self.symbol_portfolios[symbol]
                            symbol_snapshot = {symbol: market_snapshot.get(symbol)}

                            # CRITICAL FIX: Scale weights for independent execution
                            # Strategies return weights assuming shared portfolio (e.g., 0.25 for 1/4 symbols)
                            # But in independent mode, each symbol has its own capital, so scale up:
                            # weight=0.25 (25% of shared $400k) → weight=1.0 (100% of allocated $100k)
                            adjusted_order = order.copy()
                            if order.get('action') == 'target_weight' and 'weight' in order:
                                original_weight = order['weight']
                                # Scale by number of symbols: 0.25 * 4 = 1.0
                                scaled_weight = original_weight * len(universe)
                                adjusted_order['weight'] = scaled_weight

                                self.logger.debug(
                                    f"Independent execution: scaled {symbol} weight "
                                    f"{original_weight:.2f} → {scaled_weight:.2f}"
                                )

                            # Execute order for this specific symbol
                            fills = self.execution_engine.execute(
                                orders={symbol: adjusted_order},
                                market_data=symbol_snapshot,
                                current_time=current_time,
                                positions=portfolio.positions,
                                context=self.strategy_context,
                                symbol_portfolio=portfolio  # Pass symbol-specific portfolio
                            )

                            # Process fills in symbol's portfolio
                            for fill in fills:
                                portfolio.process_fill(fill, {symbol: current_prices.get(symbol, 0)})
                                self.trade_logger.log_fill(fill)

                    # Record equity snapshots for ALL symbols (not just those with orders)
                    for symbol, portfolio in self.symbol_portfolios.items():
                        if symbol in current_prices:
                            portfolio.get_equity_snapshot({symbol: current_prices[symbol]}, current_time)

                else:
                    # Legacy mode: shared portfolio
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

                # Progress logging (only if verbose enabled)
                if self.config.verbose and (i % 100 == 0 or i == len(date_range) - 1):
                    progress = (i + 1) / len(date_range) * 100

                    if self.independent_execution:
                        # Sum equity from all symbol portfolios
                        equity = sum(
                            portfolio.get_portfolio_value({symbol: current_prices.get(symbol, 0)})
                            for symbol, portfolio in self.symbol_portfolios.items()
                        )
                    else:
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

    def _aggregate_equity_curves(self) -> pd.DataFrame:
        """Aggregate equity curves from all symbol portfolios"""
        if not self.symbol_portfolios:
            return pd.DataFrame()

        # Get all equity curves and ensure they're aligned
        all_curves = {}
        for symbol, portfolio in self.symbol_portfolios.items():
            curve = portfolio.get_equity_curve()
            if not curve.empty:
                all_curves[symbol] = curve

        if not all_curves:
            return pd.DataFrame()

        # Get union of all timestamps
        all_timestamps = set()
        for curve in all_curves.values():
            all_timestamps.update(curve.index)

        all_timestamps = sorted(all_timestamps)

        # Create combined DataFrame by summing across all symbols at each timestamp
        combined_data = []
        for timestamp in all_timestamps:
            row = {
                'cash': 0.0,
                'positions_value': 0.0,
                'total_equity': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_pnl': 0.0,
                'total_return': 0.0
            }

            count = 0
            for symbol, curve in all_curves.items():
                if timestamp in curve.index:
                    symbol_data = curve.loc[timestamp]
                    row['cash'] += symbol_data.get('cash', 0)
                    row['positions_value'] += symbol_data.get('positions_value', 0)
                    row['total_equity'] += symbol_data.get('total_equity', 0)
                    row['unrealized_pnl'] += symbol_data.get('unrealized_pnl', 0)
                    row['realized_pnl'] += symbol_data.get('realized_pnl', 0)
                    row['total_pnl'] += symbol_data.get('total_pnl', 0)
                    count += 1

            # Calculate aggregate return
            if count > 0 and self.config.initial_capital > 0:
                row['total_return'] = (row['total_equity'] - self.config.initial_capital) / self.config.initial_capital

            combined_data.append(row)

        combined = pd.DataFrame(combined_data, index=all_timestamps)
        combined.index.name = 'timestamp'
        return combined

    def _get_symbol_equity_curves(self) -> Dict[str, pd.DataFrame]:
        """Get individual equity curves from symbol portfolios"""
        symbol_curves = {}

        for symbol, portfolio in self.symbol_portfolios.items():
            # For symbol-specific portfolios, get the per-symbol equity curve
            per_symbol_curves = portfolio.get_per_symbol_equity_curves()
            if per_symbol_curves and symbol in per_symbol_curves:
                symbol_curves[symbol] = per_symbol_curves[symbol]
            elif not portfolio.get_equity_curve().empty:
                # Fallback to portfolio equity curve if per-symbol not available
                symbol_curves[symbol] = portfolio.get_equity_curve()

        return symbol_curves

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