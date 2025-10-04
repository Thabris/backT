"""
Strategy optimizer for BackT

Performs grid search and parameter optimization for trading strategies.
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
import warnings

from ..engine.backtester import Backtester
from ..utils.config import BacktestConfig
from ..utils.types import BacktestResult
from ..utils.logging_config import LoggerMixin


@dataclass
class OptimizationResult:
    """Results from strategy optimization"""

    # Best performing parameters
    best_params: Dict[str, Any]
    best_metric_value: float
    best_result: BacktestResult

    # All results
    all_results: pd.DataFrame
    optimization_metric: str

    # Metadata
    total_combinations: int
    execution_time: float
    start_time: datetime
    end_time: datetime

    def summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            'best_params': self.best_params,
            'best_metric_value': self.best_metric_value,
            'optimization_metric': self.optimization_metric,
            'total_combinations': self.total_combinations,
            'execution_time_seconds': self.execution_time,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat()
        }

    def get_top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N parameter combinations by optimization metric"""
        return self.all_results.nlargest(n, self.optimization_metric)

    def get_bottom_n(self, n: int = 10) -> pd.DataFrame:
        """Get bottom N parameter combinations by optimization metric"""
        return self.all_results.nsmallest(n, self.optimization_metric)


class StrategyOptimizer(LoggerMixin):
    """
    Optimize strategy parameters using grid search

    Example:
        >>> optimizer = StrategyOptimizer(
        ...     strategy_function=my_strategy,
        ...     config=backtest_config,
        ...     symbols=['AAPL', 'MSFT']
        ... )
        >>>
        >>> param_grid = {
        ...     'fast_ma': range(10, 50, 5),
        ...     'slow_ma': range(50, 200, 10)
        ... }
        >>>
        >>> result = optimizer.optimize(
        ...     param_grid=param_grid,
        ...     optimization_metric='sharpe_ratio'
        ... )
        >>>
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Best Sharpe: {result.best_metric_value}")
    """

    def __init__(
        self,
        strategy_function: Callable,
        config: BacktestConfig,
        symbols: List[str],
        fixed_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize optimizer

        Args:
            strategy_function: Strategy function to optimize
            config: Backtesting configuration
            symbols: List of symbols to trade
            fixed_params: Parameters that remain constant across optimization
        """
        self.strategy_function = strategy_function
        self.config = config
        self.symbols = symbols
        self.fixed_params = fixed_params or {}

        # Cache for market data
        self._data_cache = None
        self._backtester = None

    def _load_data_once(self):
        """Load market data once and cache it"""
        if self._data_cache is None:
            self.logger.info(f"Loading market data for {len(self.symbols)} symbols...")

            # Create backtester instance
            self._backtester = Backtester(self.config)

            # Load data using backtester's data loader
            self._data_cache = self._backtester.data_loader.load(
                symbols=self.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

            self.logger.info(f"Data loaded and cached for optimization")

    def _generate_param_combinations(
        self,
        param_grid: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from grid

        Args:
            param_grid: Dictionary of parameter_name -> list/range of values

        Returns:
            List of parameter dictionaries
        """
        # Extract parameter names and values
        param_names = list(param_grid.keys())
        param_values = [
            list(param_grid[name]) if hasattr(param_grid[name], '__iter__')
            and not isinstance(param_grid[name], str)
            else [param_grid[name]]
            for name in param_names
        ]

        # Generate cartesian product
        combinations = list(product(*param_values))

        # Convert to list of dicts
        param_dicts = [
            {**self.fixed_params, **dict(zip(param_names, combo))}
            for combo in combinations
        ]

        return param_dicts

    def optimize(
        self,
        param_grid: Dict[str, Any],
        optimization_metric: str = 'sharpe_ratio',
        minimize: bool = False,
        n_jobs: int = 1,
        verbose: bool = True,
        verbose_comp: bool = False
    ) -> OptimizationResult:
        """
        Run optimization over parameter grid

        Args:
            param_grid: Dictionary of parameter ranges to optimize
                Example: {'fast_ma': range(10, 50, 5), 'slow_ma': [50, 100, 150]}
            optimization_metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
            minimize: If True, minimize the metric; if False, maximize it
            n_jobs: Number of parallel jobs (currently only supports 1)
            verbose: Print progress at intervals
            verbose_comp: Print completion percentage for every scenario tested

        Returns:
            OptimizationResult object with all results and best parameters
        """
        start_time = datetime.now()

        # Load data once
        self._load_data_once()

        # Generate all parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        total_combinations = len(param_combinations)

        self.logger.info(f"Starting optimization with {total_combinations} parameter combinations")
        self.logger.info(f"Optimizing for: {optimization_metric} ({'minimize' if minimize else 'maximize'})")

        # Store results
        results_list = []

        # Run backtests for each parameter combination
        for idx, params in enumerate(param_combinations, 1):
            # Print progress based on verbosity settings
            if verbose_comp:
                # Print every scenario
                progress = (idx / total_combinations) * 100
                print(f"\rProgress: {idx}/{total_combinations} ({progress:.1f}%) - Testing params: {params}", end='', flush=True)
            elif verbose and idx % max(1, total_combinations // 10) == 0:
                # Print at 10% intervals
                progress = (idx / total_combinations) * 100
                self.logger.info(f"Progress: {idx}/{total_combinations} ({progress:.1f}%)")

            try:
                # Run backtest with these parameters
                result = self._run_single_backtest(params)

                # Extract metrics
                metrics = result.performance_metrics.copy()

                # Add parameter values to metrics
                for param_name, param_value in params.items():
                    metrics[f'param_{param_name}'] = param_value

                # Store full result object reference (by index)
                metrics['_result_idx'] = len(results_list)

                results_list.append({
                    'params': params,
                    'metrics': metrics,
                    'result': result
                })

            except Exception as e:
                self.logger.warning(f"Backtest failed for params {params}: {e}")
                # Add failed result with NaN metrics
                metrics = {f'param_{k}': v for k, v in params.items()}
                metrics[optimization_metric] = np.nan
                metrics['_result_idx'] = -1
                results_list.append({
                    'params': params,
                    'metrics': metrics,
                    'result': None
                })

        # Clear progress line if using verbose_comp
        if verbose_comp:
            print()  # New line after progress updates

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        # Create results DataFrame
        all_metrics = [r['metrics'] for r in results_list]
        results_df = pd.DataFrame(all_metrics)

        # Find best parameters
        if optimization_metric not in results_df.columns:
            raise ValueError(f"Optimization metric '{optimization_metric}' not found in results")

        # Remove NaN results
        valid_results = results_df.dropna(subset=[optimization_metric])

        if len(valid_results) == 0:
            raise RuntimeError("All optimization runs failed - no valid results")

        # Find best
        if minimize:
            best_idx = valid_results[optimization_metric].idxmin()
        else:
            best_idx = valid_results[optimization_metric].idxmax()

        best_result_data = results_list[best_idx]
        best_params = best_result_data['params']
        best_metric_value = best_result_data['metrics'][optimization_metric]
        best_result = best_result_data['result']

        # Sort results by optimization metric
        results_df = results_df.sort_values(
            optimization_metric,
            ascending=minimize
        ).reset_index(drop=True)

        self.logger.info(f"Optimization complete in {execution_time:.2f} seconds")
        self.logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")

        return OptimizationResult(
            best_params=best_params,
            best_metric_value=best_metric_value,
            best_result=best_result,
            all_results=results_df,
            optimization_metric=optimization_metric,
            total_combinations=total_combinations,
            execution_time=execution_time,
            start_time=start_time,
            end_time=end_time
        )

    def _run_single_backtest(self, params: Dict[str, Any]) -> BacktestResult:
        """
        Run a single backtest with given parameters

        Args:
            params: Strategy parameters

        Returns:
            BacktestResult object
        """
        # Create a new backtester that uses cached data
        backtester = Backtester(self.config)

        # Override the data loader to use cached data
        backtester._market_data = self._data_cache

        # Run backtest with strategy parameters
        result = backtester.run(
            strategy=self.strategy_function,
            universe=self.symbols,
            strategy_params=params
        )

        return result

    def analyze_parameter_impact(
        self,
        result: OptimizationResult,
        param_name: str,
        metric: str = 'sharpe_ratio'
    ) -> pd.DataFrame:
        """
        Analyze how a single parameter affects performance

        Args:
            result: OptimizationResult from optimize()
            param_name: Name of parameter to analyze
            metric: Performance metric to analyze

        Returns:
            DataFrame with parameter values and corresponding metric values
        """
        param_col = f'param_{param_name}'

        if param_col not in result.all_results.columns:
            raise ValueError(f"Parameter '{param_name}' not found in results")

        # Group by parameter value and get average metric
        analysis = result.all_results.groupby(param_col)[metric].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).reset_index()

        analysis.columns = [param_name, f'{metric}_mean', f'{metric}_std',
                           f'{metric}_min', f'{metric}_max', 'count']

        return analysis.sort_values(param_name)

    def plot_parameter_heatmap(
        self,
        result: OptimizationResult,
        param_x: str,
        param_y: str,
        metric: str = 'sharpe_ratio',
        figsize: tuple = (10, 8)
    ):
        """
        Create a heatmap showing metric values across two parameters

        Args:
            result: OptimizationResult from optimize()
            param_x: Parameter name for x-axis
            param_y: Parameter name for y-axis
            metric: Metric to plot
            figsize: Figure size
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            self.logger.error("matplotlib and seaborn required for plotting")
            return

        param_x_col = f'param_{param_x}'
        param_y_col = f'param_{param_y}'

        # Create pivot table
        pivot_data = result.all_results.pivot_table(
            values=metric,
            index=param_y_col,
            columns=param_x_col,
            aggfunc='mean'
        )

        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0 if metric in ['sharpe_ratio', 'sortino_ratio'] else None,
            cbar_kws={'label': metric}
        )

        plt.title(f'{metric} Heatmap: {param_y} vs {param_x}')
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.tight_layout()

        return plt.gcf()
