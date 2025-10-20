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
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
import logging

# Suppress noisy warnings from parallel workers
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*No runtime found.*')
warnings.filterwarnings('ignore', message='.*to view a Streamlit app.*')

# Suppress Streamlit warnings at logging level too
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

from ..engine.backtester import Backtester
from ..utils.config import BacktestConfig
from ..utils.types import BacktestResult
from ..utils.logging_config import LoggerMixin


def _run_backtest_worker(
    params: Dict[str, Any],
    strategy_function: Callable,
    backtest_config_dict: Dict[str, Any],
    symbols: List[str],
    cached_data: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, Any], Dict[str, float], Optional[BacktestResult]]:
    """
    Worker function for parallel backtest execution

    Args:
        params: Strategy parameters to test
        strategy_function: Strategy function
        backtest_config_dict: Backtest config as dict (picklable)
        symbols: Trading symbols
        cached_data: Cached market data

    Returns:
        Tuple of (params, metrics, result)
    """
    # Suppress warnings in subprocess worker
    import warnings
    import logging
    warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
    warnings.filterwarnings('ignore', message='.*No runtime found.*')
    warnings.filterwarnings('ignore', message='.*to view a Streamlit app.*')
    logging.getLogger('streamlit').setLevel(logging.ERROR)

    try:
        # Reconstruct BacktestConfig from dict
        config = BacktestConfig(**backtest_config_dict)

        # Create cached data loader
        from ..data.loaders import DataLoader

        class CachedDataLoader(DataLoader):
            def __init__(self, cached_data_dict):
                self.cached_data = cached_data_dict
                # Debug logging
                import logging
                logger = logging.getLogger(__name__)
                if cached_data_dict is not None and len(cached_data_dict) > 0:
                    logger.info(f"[WORKER] CachedDataLoader initialized with symbols: {list(cached_data_dict.keys())}")
                else:
                    logger.warning("[WORKER] CachedDataLoader initialized with EMPTY cache!")

            def load(self, symbols, start_date, end_date):
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"[WORKER] CachedDataLoader.load() called with symbols={symbols}")
                cache_info = list(self.cached_data.keys()) if (self.cached_data is not None and len(self.cached_data) > 0) else 'NONE'
                logger.info(f"[WORKER]   Available in cache: {cache_info}")

                result = {}
                if self.cached_data is None or len(self.cached_data) == 0:
                    logger.error("[WORKER] CachedDataLoader.load(): cached_data is None or empty!")
                    return result

                for symbol in symbols:
                    if symbol in self.cached_data:
                        df = self.cached_data[symbol]
                        start_ts = pd.Timestamp(start_date)
                        end_ts = pd.Timestamp(end_date)
                        if df.index.tz is not None:
                            start_ts = start_ts.tz_localize('UTC')
                            end_ts = end_ts.tz_localize('UTC')
                        mask = (df.index >= start_ts) & (df.index <= end_ts)
                        result[symbol] = df[mask].copy()
                        logger.info(f"[WORKER]   Loaded {len(result[symbol])} rows for {symbol}")
                    else:
                        logger.warning(f"[WORKER]   Symbol {symbol} NOT found in cache!")

                if not result:
                    logger.error(f"[WORKER] CachedDataLoader.load() returning EMPTY result!")
                return result

            def validate_data(self, data):
                return True

        # Create backtester with cached data
        cached_loader = CachedDataLoader(cached_data)
        backtester = Backtester(config, data_loader=cached_loader)

        # Run backtest
        result = backtester.run(strategy_function, symbols, params)

        # Extract metrics
        metrics = result.performance_metrics.copy()

        return (params, metrics, result)

    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Backtest failed for params {params}: {type(e).__name__}: {str(e)}")

        # Return failed result with error info
        return (params, {'error': str(e)}, None)


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

    # Parallelization info
    n_jobs: int = 1
    parallel_speedup: Optional[float] = None

    def summary(self) -> Dict[str, Any]:
        """Get optimization summary"""
        return {
            'best_params': self.best_params,
            'best_metric_value': self.best_metric_value,
            'optimization_metric': self.optimization_metric,
            'total_combinations': self.total_combinations,
            'execution_time_seconds': self.execution_time,
            'n_jobs': self.n_jobs,
            'avg_time_per_backtest': self.execution_time / max(self.total_combinations, 1),
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
            self.logger.info(f"Loading market data for {len(self.symbols)} symbols: {self.symbols}")

            # Create backtester instance
            self._backtester = Backtester(self.config)

            # Load data using backtester's data loader
            self._data_cache = self._backtester.data_loader.load(
                symbols=self.symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

            # Debug: Check what was actually loaded
            if self._data_cache is not None and len(self._data_cache) > 0:
                self.logger.info(f"Data loaded successfully. Symbols in cache: {list(self._data_cache.keys())}")
                for symbol, df in self._data_cache.items():
                    self.logger.info(f"  {symbol}: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
            else:
                self.logger.warning(f"WARNING: Data cache is empty after loading!")

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

    def _run_sequential(
        self,
        param_combinations: List[Dict[str, Any]],
        total_combinations: int,
        verbose: bool,
        verbose_comp: bool,
        optimization_metric: str
    ) -> List[Dict[str, Any]]:
        """Run backtests sequentially with progress bar"""
        results_list = []

        # Use tqdm progress bar if available, otherwise fall back to logging
        if HAS_TQDM and not verbose_comp:
            iterator = tqdm(param_combinations, desc="Grid Search", unit="backtest", ncols=100)
        else:
            iterator = param_combinations

        for idx, params in enumerate(iterator, 1):
            # Legacy verbose logging (only if tqdm not available)
            if not HAS_TQDM:
                if verbose_comp:
                    progress = (idx / total_combinations) * 100
                    print(f"\rProgress: {idx}/{total_combinations} ({progress:.1f}%) - Testing params: {params}", end='', flush=True)
                elif verbose and idx % max(1, total_combinations // 10) == 0:
                    progress = (idx / total_combinations) * 100
                    self.logger.info(f"Progress: {idx}/{total_combinations} ({progress:.1f}%)")

            try:
                # Run backtest
                result = self._run_single_backtest(params)
                metrics = result.performance_metrics.copy()

                # Add parameter values to metrics
                for param_name, param_value in params.items():
                    metrics[f'param_{param_name}'] = param_value

                metrics['_result_idx'] = len(results_list)

                results_list.append({
                    'params': params,
                    'metrics': metrics,
                    'result': result
                })

            except Exception as e:
                self.logger.warning(f"Backtest failed for params {params}: {type(e).__name__}: {str(e)}")
                metrics = {f'param_{k}': v for k, v in params.items()}
                metrics[optimization_metric] = np.nan
                metrics['error'] = str(e)
                metrics['_result_idx'] = -1
                results_list.append({
                    'params': params,
                    'metrics': metrics,
                    'result': None
                })

        return results_list

    def _run_parallel(
        self,
        param_combinations: List[Dict[str, Any]],
        total_combinations: int,
        n_jobs: int,
        verbose: bool,
        optimization_metric: str
    ) -> List[Dict[str, Any]]:
        """Run backtests in parallel using ProcessPoolExecutor"""

        # Prepare config dict for pickling
        from dataclasses import fields
        config_dict = {}
        for field_obj in fields(BacktestConfig):
            value = getattr(self.config, field_obj.name)
            config_dict[field_obj.name] = value

        # Create worker function
        worker_func = partial(
            _run_backtest_worker,
            strategy_function=self.strategy_function,
            backtest_config_dict=config_dict,
            symbols=self.symbols,
            cached_data=self._data_cache
        )

        results_list = []
        completed = 0

        # Try parallel execution with fallback
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_params = {
                    executor.submit(worker_func, params): params
                    for params in param_combinations
                }

                # Wrap with progress bar if available
                futures_iter = as_completed(future_to_params)
                if HAS_TQDM:
                    futures_iter = tqdm(futures_iter, total=total_combinations,
                                       desc="Parallel Grid Search", unit="backtest", ncols=100)

                # Collect results
                for future in futures_iter:
                    params = future_to_params[future]
                    completed += 1

                    try:
                        result_params, metrics, result = future.result()

                        # Check if this is a successful result (has optimization metric)
                        if result is not None and optimization_metric in metrics:
                            # Success - add parameter values to metrics
                            for param_name, param_value in result_params.items():
                                metrics[f'param_{param_name}'] = param_value

                            metrics['_result_idx'] = len(results_list)

                            results_list.append({
                                'params': result_params,
                                'metrics': metrics,
                                'result': result
                            })
                        else:
                            # Failed - merge error info from worker with params
                            failed_metrics = {f'param_{k}': v for k, v in params.items()}
                            failed_metrics[optimization_metric] = np.nan
                            failed_metrics['_result_idx'] = -1
                            # Keep error message if present
                            if 'error' in metrics:
                                failed_metrics['error'] = metrics['error']
                            results_list.append({
                                'params': params,
                                'metrics': failed_metrics,
                                'result': None
                            })

                        # Legacy logging (only if tqdm not available)
                        if not HAS_TQDM and verbose and completed % max(1, total_combinations // 10) == 0:
                            progress = (completed / total_combinations) * 100
                            self.logger.info(f"Progress: {completed}/{total_combinations} ({progress:.1f}%)")

                    except Exception as e:
                        self.logger.error(f"Parallel execution error for {params}: {e}")
                        metrics = {f'param_{k}': v for k, v in params.items()}
                        metrics[optimization_metric] = np.nan
                        metrics['_result_idx'] = -1
                        results_list.append({
                            'params': params,
                            'metrics': metrics,
                            'result': None
                        })

        except Exception as e:
            self.logger.warning(f"Parallel execution failed: {e}. Falling back to sequential.")
            # Fallback to sequential
            return self._run_sequential(
                param_combinations, total_combinations, verbose, False, optimization_metric
            )

        return results_list

    def optimize(
        self,
        param_grid: Dict[str, Any],
        optimization_metric: str = 'sharpe_ratio',
        minimize: bool = False,
        n_jobs: int = -1,  # Default to all CPUs
        verbose: bool = True,
        verbose_comp: bool = False
    ) -> OptimizationResult:
        """
        Run optimization over parameter grid (with parallel execution!)

        Args:
            param_grid: Dictionary of parameter ranges to optimize
                Example: {'fast_ma': range(10, 50, 5), 'slow_ma': [50, 100, 150]}
            optimization_metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
            minimize: If True, minimize the metric; if False, maximize it
            n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential)
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

        # Determine number of workers
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs < 1:
            n_jobs = 1

        self.logger.info(f"Starting optimization with {total_combinations} parameter combinations")
        self.logger.info(f"Using {n_jobs} parallel workers")
        self.logger.info(f"Optimizing for: {optimization_metric} ({'minimize' if minimize else 'maximize'})")

        # Store results
        results_list = []

        # Choose execution method
        if n_jobs == 1:
            # Sequential execution (original code)
            results_list = self._run_sequential(
                param_combinations, total_combinations, verbose, verbose_comp, optimization_metric
            )
        else:
            # Parallel execution (NEW!)
            results_list = self._run_parallel(
                param_combinations, total_combinations, n_jobs, verbose, optimization_metric
            )

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
            # Build detailed error message
            error_details = []
            error_details.append("=" * 80)
            error_details.append("ALL OPTIMIZATION RUNS FAILED - DIAGNOSTIC DETAILS")
            error_details.append("=" * 80)
            error_details.append(f"Total runs: {len(results_df)}")
            error_details.append(f"Available columns: {results_df.columns.tolist()}")
            error_details.append("")

            # Check if there are error messages
            if 'error' in results_df.columns:
                error_counts = results_df['error'].value_counts()
                error_details.append("ERROR BREAKDOWN:")
                for error_msg, count in error_counts.head(5).items():
                    error_details.append(f"  {count}x: {error_msg}")
                error_details.append("")

            # Show first few results for debugging
            if len(results_df) > 0:
                error_details.append("SAMPLE FAILED RESULTS:")
                for idx, row in results_df.head(3).iterrows():
                    param_cols = [col for col in results_df.columns if col.startswith('param_')]
                    params = {col.replace('param_', ''): row[col] for col in param_cols}
                    error_msg = row.get('error', 'Unknown error')
                    error_details.append(f"  Params: {params}")
                    error_details.append(f"    Error: {error_msg}")
                    error_details.append(f"    {optimization_metric}: {row.get(optimization_metric, 'N/A')}")
                    error_details.append("")

            error_details.append("=" * 80)

            # Print to stdout (Streamlit will capture this)
            error_message = "\n".join(error_details)
            print(error_message)

            # Also log it
            for line in error_details:
                self.logger.error(line)

            raise RuntimeError(
                f"All optimization runs failed - no valid results.\n\n{error_message}"
            )

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
            end_time=end_time,
            n_jobs=n_jobs
        )

    def _run_single_backtest(self, params: Dict[str, Any]) -> BacktestResult:
        """
        Run a single backtest with given parameters

        Args:
            params: Strategy parameters

        Returns:
            BacktestResult object
        """
        # Create cached data loader (same as parallel worker)
        from ..data.loaders import DataLoader
        import logging
        logger = logging.getLogger(__name__)

        class CachedDataLoader(DataLoader):
            def __init__(self, cached_data_dict):
                self.cached_data = cached_data_dict
                # Debug logging
                if cached_data_dict is not None and len(cached_data_dict) > 0:
                    logger.info(f"[SEQUENTIAL] CachedDataLoader initialized with symbols: {list(cached_data_dict.keys())}")
                else:
                    logger.warning("[SEQUENTIAL] CachedDataLoader initialized with EMPTY cache!")

            def load(self, symbols, start_date, end_date):
                logger.info(f"[SEQUENTIAL] CachedDataLoader.load() called with symbols={symbols}")
                cache_info = list(self.cached_data.keys()) if (self.cached_data is not None and len(self.cached_data) > 0) else 'NONE'
                logger.info(f"[SEQUENTIAL]   Available in cache: {cache_info}")

                result = {}
                if self.cached_data is None or len(self.cached_data) == 0:
                    logger.error("[SEQUENTIAL] CachedDataLoader.load(): cached_data is None or empty!")
                    return result

                for symbol in symbols:
                    if symbol in self.cached_data:
                        df = self.cached_data[symbol]
                        start_ts = pd.Timestamp(start_date)
                        end_ts = pd.Timestamp(end_date)
                        if df.index.tz is not None:
                            start_ts = start_ts.tz_localize('UTC')
                            end_ts = end_ts.tz_localize('UTC')
                        mask = (df.index >= start_ts) & (df.index <= end_ts)
                        result[symbol] = df[mask].copy()
                        logger.info(f"[SEQUENTIAL]   Loaded {len(result[symbol])} rows for {symbol}")
                    else:
                        logger.warning(f"[SEQUENTIAL]   Symbol {symbol} NOT found in cache!")

                if not result:
                    logger.error(f"[SEQUENTIAL] CachedDataLoader.load() returning EMPTY result!")
                return result

            def validate_data(self, data):
                return True

        # Create backtester with cached data loader
        cached_loader = CachedDataLoader(self._data_cache)
        backtester = Backtester(self.config, data_loader=cached_loader)

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

    def optimize_with_cpcv(
        self,
        param_grid: Dict[str, Any],
        optimization_metric: str = 'sharpe_ratio',
        minimize: bool = False,
        n_jobs: int = -1,
        top_k: int = 3,
        cpcv_config: Optional[Any] = None,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run grid search followed by CPCV validation on top K parameters

        This implements a 2-tier optimization strategy:
        1. Grid search (parallel) to find top K parameter sets
        2. CPCV validation on top K to detect overfitting

        Args:
            param_grid: Parameter grid for grid search
            optimization_metric: Metric to optimize
            minimize: If True, minimize; if False, maximize
            n_jobs: Number of parallel workers
            top_k: Number of top parameters to validate with CPCV
            cpcv_config: CPCVConfig instance for validation
            verbose: Print progress

        Returns:
            OptimizationResult with CPCV validation on top K parameters
        """
        # Step 1: Run grid search optimization
        self.logger.info("=" * 80)
        self.logger.info("TIER 1: PARALLEL GRID SEARCH")
        self.logger.info("=" * 80)

        result = self.optimize(
            param_grid=param_grid,
            optimization_metric=optimization_metric,
            minimize=minimize,
            n_jobs=n_jobs,
            verbose=verbose
        )

        # Step 2: CPCV validation on top K
        if cpcv_config is not None:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"TIER 2: CPCV VALIDATION ON TOP {top_k} PARAMETERS")
            self.logger.info("=" * 80)

            from ..validation.cpcv_validator import CPCVValidator

            # Get top K parameter sets from DataFrame
            top_k_df = result.all_results.nlargest(top_k, optimization_metric)

            # Extract parameter columns
            param_cols = [col for col in top_k_df.columns if col.startswith('param_')]

            # Run CPCV on each
            cpcv_results = []
            for idx, row in enumerate(top_k_df.itertuples(index=False), 1):
                # Extract parameters from row
                params = {}
                for col in param_cols:
                    param_name = col.replace('param_', '')
                    params[param_name] = getattr(row, col)

                self.logger.info(f"\nValidating parameter set {idx}/{top_k}: {params}")

                # Run CPCV - pass both backtest_config and cpcv_config
                validator = CPCVValidator(self.config, cpcv_config)
                cpcv_result = validator.validate(
                    strategy=self.strategy_function,
                    symbols=self.symbols,
                    strategy_params=params
                )

                cpcv_results.append({
                    'params': params,
                    'cpcv_result': cpcv_result
                })

                self.logger.info(f"  PBO: {cpcv_result.overfitting_metrics.pbo:.3f}")
                self.logger.info(f"  DSR: {cpcv_result.overfitting_metrics.dsr:.3f}")
                self.logger.info(f"  Degradation: {cpcv_result.overfitting_metrics.degradation_pct:.1f}%")

            # Store CPCV results in OptimizationResult
            # Note: OptimizationResult doesn't have cpcv fields yet, but we can log them

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("VALIDATION COMPLETE")
            self.logger.info("=" * 80)

            # Return result with CPCV info attached
            result.cpcv_results = cpcv_results

        return result
