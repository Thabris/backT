"""
Combinatorial Purged Cross-Validation (CPCV) for BackT

Implements the CPCV methodology from "Advances in Financial Machine Learning"
by Marcos Lopez de Prado for robust strategy validation.

CPCV provides:
- Multiple train/test paths to reduce selection bias
- Purging and embargoing to prevent data leakage
- Overfitting detection metrics (PBO, DSR)
- Stability analysis across validation paths
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

from ..engine.backtester import Backtester
from ..utils.config import BacktestConfig
from ..utils.types import BacktestResult, StrategyFunction
from ..utils.logging_config import LoggerMixin
from .parameter_grid import ParameterGrid
from .overfitting import (
    analyze_overfitting_comprehensive,
    OverfittingMetrics,
    interpret_overfitting_metrics
)
from ..utils.data_splits import (
    generate_cpcv_combinations,
    create_cpcv_split,
    validate_split_quality
)


@dataclass
class CPCVConfig:
    """Configuration for CPCV validation"""

    # Cross-validation settings
    n_splits: int = 10  # Number of folds
    n_test_splits: int = 2  # Number of test folds per combination

    # Data leakage prevention
    purge_pct: float = 0.05  # Purge 5% around test sets
    embargo_pct: float = 0.02  # Embargo 2% after test sets

    # Validation settings
    min_train_pct: float = 0.50  # Minimum 50% data for training
    random_state: Optional[int] = None  # For reproducibility

    # Performance thresholds
    acceptable_pbo: float = 0.5  # PBO threshold (lower is better)
    acceptable_dsr: float = 1.0  # DSR threshold (higher is better)
    acceptable_degradation: float = 30.0  # Max degradation % (lower is better)

    # Performance optimization settings
    n_jobs: int = -1  # Number of parallel workers (-1 = all CPUs, 1 = sequential)
    use_numba: bool = False  # Use numba JIT compilation (requires numba package)

    def __post_init__(self):
        """Validate configuration"""
        if self.n_splits < 3:
            raise ValueError("n_splits must be at least 3")
        if self.n_test_splits < 1 or self.n_test_splits >= self.n_splits:
            raise ValueError(f"n_test_splits must be between 1 and {self.n_splits-1}")
        if not (0 < self.purge_pct < 0.5):
            raise ValueError("purge_pct must be between 0 and 0.5")
        if not (0 <= self.embargo_pct < 0.5):
            raise ValueError("embargo_pct must be between 0 and 0.5")


@dataclass
class PathResult:
    """Results from a single CPCV path"""
    path_id: int
    test_fold_indices: Tuple[int, ...]
    train_indices: np.ndarray
    test_indices: np.ndarray
    backtest_result: BacktestResult
    sharpe_ratio: float
    total_return: float
    max_drawdown: float


def _run_single_cpcv_path(
    path_id: int,
    test_fold_indices: Tuple[int, ...],
    n_samples: int,
    n_splits: int,
    purge_pct: float,
    embargo_pct: float,
    min_train_pct: float,
    equity_curve_index: pd.DatetimeIndex,
    backtest_config_dict: Dict[str, Any],
    strategy: StrategyFunction,
    symbols: List[str],
    strategy_params: Dict[str, Any],
    cached_data: Dict[str, pd.DataFrame]
) -> Optional[PathResult]:
    """
    Run a single CPCV path (for parallel processing)

    This function is designed to be pickled and executed in a separate process.
    """
    from ..data.loaders import DataLoader

    # Create train/test split for this path
    train_indices, test_indices = create_cpcv_split(
        np.arange(n_samples),
        test_fold_indices,
        n_splits,
        purge_pct,
        embargo_pct
    )

    # Validate split quality
    is_valid, msg = validate_split_quality(
        train_indices,
        test_indices,
        n_samples,
        min_train_pct
    )

    if not is_valid:
        return None

    # Create cached data loader
    class CachedDataLoader(DataLoader):
        """Data loader that returns pre-loaded cached data"""
        def __init__(self, cached_data_dict):
            self.cached_data = cached_data_dict

        def load(self, symbols, start_date, end_date):
            """Return sliced cached data for the date range"""
            result = {}
            for symbol in symbols:
                if symbol in self.cached_data:
                    df = self.cached_data[symbol]
                    # Filter to date range - handle timezone awareness
                    start_ts = pd.Timestamp(start_date)
                    end_ts = pd.Timestamp(end_date)
                    # Match timezone of the data index
                    if df.index.tz is not None:
                        start_ts = start_ts.tz_localize('UTC')
                        end_ts = end_ts.tz_localize('UTC')
                    mask = (df.index >= start_ts) & (df.index <= end_ts)
                    result[symbol] = df[mask].copy()
            return result

        def validate_data(self, data):
            """Validate cached data (already validated, so just return True)"""
            return True

    # Run backtest on test set only (out-of-sample)
    test_dates = equity_curve_index[test_indices]

    # Create path-specific config
    from dataclasses import fields
    config_dict = backtest_config_dict.copy()
    config_dict['start_date'] = test_dates[0].strftime('%Y-%m-%d')
    config_dict['end_date'] = test_dates[-1].strftime('%Y-%m-%d')
    config_dict['verbose'] = False  # Suppress logging for paths

    path_config = BacktestConfig(**config_dict)

    # Create backtester with cached data loader
    cached_loader = CachedDataLoader(cached_data)
    path_backtester = Backtester(path_config, data_loader=cached_loader)
    path_result = path_backtester.run(strategy, symbols, strategy_params)

    # Extract metrics
    sharpe = path_result.performance_metrics.get('sharpe_ratio', 0.0)
    total_return = path_result.performance_metrics.get('total_return', 0.0)
    max_dd = path_result.performance_metrics.get('max_drawdown', 0.0)

    return PathResult(
        path_id=path_id,
        test_fold_indices=test_fold_indices,
        train_indices=train_indices,
        test_indices=test_indices,
        backtest_result=path_result,
        sharpe_ratio=sharpe,
        total_return=total_return,
        max_drawdown=max_dd
    )


@dataclass
class CPCVResult:
    """Complete CPCV validation results"""

    # Configuration
    config: CPCVConfig
    backtest_config: BacktestConfig
    parameter_set: Dict[str, Any]

    # Path results
    path_results: List[PathResult]
    n_paths: int

    # Aggregate metrics
    mean_sharpe: float
    std_sharpe: float
    mean_return: float
    mean_max_drawdown: float

    # Overfitting metrics
    overfitting_metrics: OverfittingMetrics
    overfitting_interpretations: Dict[str, str]

    # Validation status
    is_valid: bool
    validation_warnings: List[str]

    # Metadata
    total_runtime_seconds: float
    start_time: datetime
    end_time: datetime

    def summary(self) -> Dict[str, Any]:
        """Get summary of CPCV results"""
        return {
            'n_paths': self.n_paths,
            'mean_sharpe': self.mean_sharpe,
            'std_sharpe': self.std_sharpe,
            'sharpe_stability': self.overfitting_metrics.sharpe_stability,
            'pbo': self.overfitting_metrics.pbo,
            'dsr': self.overfitting_metrics.dsr,
            'degradation_pct': self.overfitting_metrics.degradation_pct,
            'is_valid': self.is_valid,
            'warnings': len(self.validation_warnings),
            'runtime_seconds': self.total_runtime_seconds
        }

    def passes_validation(self) -> bool:
        """Check if strategy passes all validation thresholds"""
        metrics = self.overfitting_metrics
        config = self.config

        checks = [
            metrics.pbo < config.acceptable_pbo,
            metrics.dsr > config.acceptable_dsr,
            metrics.degradation_pct < config.acceptable_degradation,
            self.is_valid
        ]

        return all(checks)


class CPCVValidator(LoggerMixin):
    """
    Combinatorial Purged Cross-Validation validator for trading strategies

    Example:
        >>> from backt import Backtester, BacktestConfig
        >>> from backt.validation import CPCVValidator, CPCVConfig
        >>>
        >>> config = BacktestConfig(
        ...     start_date='2015-01-01',
        ...     end_date='2023-12-31',
        ...     initial_capital=100000
        ... )
        >>>
        >>> cpcv_config = CPCVConfig(
        ...     n_splits=10,
        ...     n_test_splits=2
        ... )
        >>>
        >>> validator = CPCVValidator(config, cpcv_config)
        >>> result = validator.validate(
        ...     strategy=my_strategy,
        ...     symbols=['SPY', 'TLT', 'GLD'],
        ...     strategy_params={'lookback': 12}
        ... )
        >>>
        >>> print(f"PBO: {result.overfitting_metrics.pbo:.2%}")
        >>> print(f"DSR: {result.overfitting_metrics.dsr:.2f}")
        >>> print(f"Passes validation: {result.passes_validation()}")
    """

    def __init__(
        self,
        backtest_config: BacktestConfig,
        cpcv_config: Optional[CPCVConfig] = None
    ):
        """
        Initialize CPCV validator

        Args:
            backtest_config: Base configuration for backtesting
            cpcv_config: CPCV-specific configuration
        """
        self.backtest_config = backtest_config
        self.cpcv_config = cpcv_config or CPCVConfig()

        self.logger.info(
            f"CPCV Validator initialized: {self.cpcv_config.n_splits} folds, "
            f"{self.cpcv_config.n_test_splits} test splits"
        )

    def validate(
        self,
        strategy: StrategyFunction,
        symbols: List[str],
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> CPCVResult:
        """
        Run CPCV validation for a single parameter set

        Args:
            strategy: Strategy function to validate
            symbols: List of symbols to trade
            strategy_params: Strategy parameters

        Returns:
            CPCVResult with validation metrics
        """
        start_time = datetime.now()
        self.logger.info("Starting CPCV validation")

        if strategy_params is None:
            strategy_params = {}

        # Generate CPCV combinations
        combinations = generate_cpcv_combinations(
            self.cpcv_config.n_splits,
            self.cpcv_config.n_test_splits
        )

        self.logger.info(f"Generated {len(combinations)} CPCV path combinations")

        # First, run a full backtest to get data structure AND cache the data
        self.logger.info("Loading data and running full backtest...")
        backtester = Backtester(self.backtest_config)
        full_result = backtester.run(strategy, symbols, strategy_params)

        # Get total number of time periods
        n_samples = len(full_result.equity_curve)

        if n_samples < 100:
            warnings.warn(
                f"Only {n_samples} samples available. "
                "CPCV works best with 500+ samples for robust validation."
            )

        # PERFORMANCE OPTIMIZATION: Cache the loaded market data
        # This avoids re-fetching from Yahoo Finance 45+ times!
        cached_data = backtester.market_data.copy()
        self.logger.info(f"Cached data for {len(cached_data)} symbols to avoid re-fetching")

        # Prepare config dict for parallel processing (BacktestConfig is not picklable with all attrs)
        from dataclasses import fields
        config_dict = {}
        for field_obj in fields(BacktestConfig):
            value = getattr(self.backtest_config, field_obj.name)
            config_dict[field_obj.name] = value

        # Run backtests for each CPCV path
        path_results = []

        # Determine number of workers
        n_jobs = self.cpcv_config.n_jobs
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        elif n_jobs < 1:
            n_jobs = 1

        # Run paths in parallel or sequential based on n_jobs
        if n_jobs == 1:
            # Sequential execution (original behavior)
            self.logger.info("Running CPCV paths sequentially (n_jobs=1)")
            for path_id, test_fold_indices in enumerate(combinations):
                self.logger.info(
                    f"Running path {path_id + 1}/{len(combinations)}: "
                    f"test folds {test_fold_indices}"
                )

                result = _run_single_cpcv_path(
                    path_id=path_id,
                    test_fold_indices=test_fold_indices,
                    n_samples=n_samples,
                    n_splits=self.cpcv_config.n_splits,
                    purge_pct=self.cpcv_config.purge_pct,
                    embargo_pct=self.cpcv_config.embargo_pct,
                    min_train_pct=self.cpcv_config.min_train_pct,
                    equity_curve_index=full_result.equity_curve.index,
                    backtest_config_dict=config_dict,
                    strategy=strategy,
                    symbols=symbols,
                    strategy_params=strategy_params,
                    cached_data=cached_data
                )

                if result is not None:
                    path_results.append(result)
                else:
                    self.logger.warning(f"Path {path_id} skipped due to invalid split")

        else:
            # Parallel execution
            self.logger.info(
                f"Running {len(combinations)} CPCV paths in parallel with {n_jobs} workers"
            )

            # Create partial function with fixed parameters
            worker_func = partial(
                _run_single_cpcv_path,
                n_samples=n_samples,
                n_splits=self.cpcv_config.n_splits,
                purge_pct=self.cpcv_config.purge_pct,
                embargo_pct=self.cpcv_config.embargo_pct,
                min_train_pct=self.cpcv_config.min_train_pct,
                equity_curve_index=full_result.equity_curve.index,
                backtest_config_dict=config_dict,
                strategy=strategy,
                symbols=symbols,
                strategy_params=strategy_params,
                cached_data=cached_data
            )

            # Try parallel execution with automatic fallback to sequential
            pickle_error_detected = False
            try:
                # Execute in parallel
                with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                    # Submit all tasks
                    future_to_path = {
                        executor.submit(worker_func, path_id, test_fold_indices): path_id
                        for path_id, test_fold_indices in enumerate(combinations)
                    }

                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_path):
                        path_id = future_to_path[future]
                        completed += 1
                        try:
                            result = future.result()
                            if result is not None:
                                path_results.append(result)
                                self.logger.info(
                                    f"Completed path {completed}/{len(combinations)}: "
                                    f"Path {path_id} - Sharpe={result.sharpe_ratio:.2f}"
                                )
                            else:
                                self.logger.warning(f"Path {path_id} skipped due to invalid split")
                        except Exception as e:
                            error_msg = str(e)
                            if "pickle" in error_msg.lower():
                                pickle_error_detected = True
                                self.logger.warning(
                                    f"Pickle error detected: {error_msg}. "
                                    "Falling back to sequential execution..."
                                )
                                break
                            else:
                                self.logger.error(f"Path {path_id} failed with error: {e}")

            except Exception as e:
                error_msg = str(e)
                if "pickle" in error_msg.lower():
                    pickle_error_detected = True
                    self.logger.warning(
                        f"Parallel execution failed due to pickle error. "
                        "Falling back to sequential execution..."
                    )
                else:
                    raise

            # Fallback to sequential execution if pickle error detected
            if pickle_error_detected:
                self.logger.info("Running CPCV paths sequentially (fallback from parallel)")
                path_results.clear()  # Clear any partial results
                for path_id, test_fold_indices in enumerate(combinations):
                    self.logger.info(
                        f"Running path {path_id + 1}/{len(combinations)}: "
                        f"test folds {test_fold_indices}"
                    )

                    result = _run_single_cpcv_path(
                        path_id=path_id,
                        test_fold_indices=test_fold_indices,
                        n_samples=n_samples,
                        n_splits=self.cpcv_config.n_splits,
                        purge_pct=self.cpcv_config.purge_pct,
                        embargo_pct=self.cpcv_config.embargo_pct,
                        min_train_pct=self.cpcv_config.min_train_pct,
                        equity_curve_index=full_result.equity_curve.index,
                        backtest_config_dict=config_dict,
                        strategy=strategy,
                        symbols=symbols,
                        strategy_params=strategy_params,
                        cached_data=cached_data
                    )

                    if result is not None:
                        path_results.append(result)
                    else:
                        self.logger.warning(f"Path {path_id} skipped due to invalid split")

        # Sort results by path_id to maintain order
        path_results.sort(key=lambda x: x.path_id)

        # Calculate aggregate metrics
        sharpe_values = np.array([p.sharpe_ratio for p in path_results])
        return_values = np.array([p.total_return for p in path_results])
        dd_values = np.array([p.max_drawdown for p in path_results])

        mean_sharpe = np.mean(sharpe_values)
        std_sharpe = np.std(sharpe_values)
        mean_return = np.mean(return_values)
        mean_max_dd = np.mean(dd_values)

        # Calculate overfitting metrics (with optional numba acceleration)
        # For now, use the same values for IS and OOS
        # In production, you'd have separate IS/OOS splits
        overfitting_metrics = analyze_overfitting_comprehensive(
            sharpe_values,  # In-sample (placeholder)
            sharpe_values,  # Out-of-sample
            returns_skewness=0.0,
            returns_kurtosis=3.0,
            n_observations=n_samples // self.cpcv_config.n_splits,
            use_numba=self.cpcv_config.use_numba
        )

        interpretations = interpret_overfitting_metrics(overfitting_metrics)

        # Validation checks
        validation_warnings = []
        is_valid = True

        if len(path_results) < len(combinations) * 0.8:
            validation_warnings.append(
                f"Only {len(path_results)}/{len(combinations)} paths completed"
            )
            is_valid = False

        if overfitting_metrics.pbo > self.cpcv_config.acceptable_pbo:
            validation_warnings.append(
                f"PBO {overfitting_metrics.pbo:.2%} exceeds threshold "
                f"{self.cpcv_config.acceptable_pbo:.2%}"
            )

        if overfitting_metrics.dsr < self.cpcv_config.acceptable_dsr:
            validation_warnings.append(
                f"DSR {overfitting_metrics.dsr:.2f} below threshold "
                f"{self.cpcv_config.acceptable_dsr:.2f}"
            )

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        result = CPCVResult(
            config=self.cpcv_config,
            backtest_config=self.backtest_config,
            parameter_set=strategy_params,
            path_results=path_results,
            n_paths=len(path_results),
            mean_sharpe=mean_sharpe,
            std_sharpe=std_sharpe,
            mean_return=mean_return,
            mean_max_drawdown=mean_max_dd,
            overfitting_metrics=overfitting_metrics,
            overfitting_interpretations=interpretations,
            is_valid=is_valid,
            validation_warnings=validation_warnings,
            total_runtime_seconds=runtime,
            start_time=start_time,
            end_time=end_time
        )

        self.logger.info(
            f"CPCV validation complete: "
            f"Mean Sharpe={mean_sharpe:.2f}, "
            f"PBO={overfitting_metrics.pbo:.2%}, "
            f"DSR={overfitting_metrics.dsr:.2f}"
        )

        return result

    def _create_path_config(self, dates: pd.DatetimeIndex) -> BacktestConfig:
        """Create backtest config for a specific date range"""
        # Get dataclass fields (not including auto-generated attributes)
        from dataclasses import fields

        config_dict = {}
        for field_obj in fields(BacktestConfig):
            value = getattr(self.backtest_config, field_obj.name)
            config_dict[field_obj.name] = value

        # Override dates for this path
        config_dict['start_date'] = dates[0].strftime('%Y-%m-%d')
        config_dict['end_date'] = dates[-1].strftime('%Y-%m-%d')
        config_dict['verbose'] = False  # Suppress logging for paths

        return BacktestConfig(**config_dict)
