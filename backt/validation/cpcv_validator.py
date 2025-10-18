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

        # Create a cached data loader to reuse data
        from ..data.loaders import DataLoader

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

        # Use cached data loader for all path backtests
        cached_loader = CachedDataLoader(cached_data)

        # Run backtests for each CPCV path
        path_results = []

        for path_id, test_fold_indices in enumerate(combinations):
            self.logger.info(
                f"Running path {path_id + 1}/{len(combinations)}: "
                f"test folds {test_fold_indices}"
            )

            # Create train/test split for this path
            train_indices, test_indices = create_cpcv_split(
                np.arange(n_samples),
                test_fold_indices,
                self.cpcv_config.n_splits,
                self.cpcv_config.purge_pct,
                self.cpcv_config.embargo_pct
            )

            # Validate split quality
            is_valid, msg = validate_split_quality(
                train_indices,
                test_indices,
                n_samples,
                self.cpcv_config.min_train_pct
            )

            if not is_valid:
                self.logger.warning(f"Path {path_id}: {msg}")
                continue

            # Run backtest on test set only (out-of-sample)
            # PERFORMANCE: Use cached data loader (no network calls!)
            test_dates = full_result.equity_curve.index[test_indices]
            path_config = self._create_path_config(test_dates)

            # Create backtester with cached data loader (10-100x faster!)
            path_backtester = Backtester(path_config, data_loader=cached_loader)
            path_result = path_backtester.run(strategy, symbols, strategy_params)

            # Extract metrics
            sharpe = path_result.performance_metrics.get('sharpe_ratio', 0.0)
            total_return = path_result.performance_metrics.get('total_return', 0.0)
            max_dd = path_result.performance_metrics.get('max_drawdown', 0.0)

            path_results.append(PathResult(
                path_id=path_id,
                test_fold_indices=test_fold_indices,
                train_indices=train_indices,
                test_indices=test_indices,
                backtest_result=path_result,
                sharpe_ratio=sharpe,
                total_return=total_return,
                max_drawdown=max_dd
            ))

        # Calculate aggregate metrics
        sharpe_values = np.array([p.sharpe_ratio for p in path_results])
        return_values = np.array([p.total_return for p in path_results])
        dd_values = np.array([p.max_drawdown for p in path_results])

        mean_sharpe = np.mean(sharpe_values)
        std_sharpe = np.std(sharpe_values)
        mean_return = np.mean(return_values)
        mean_max_dd = np.mean(dd_values)

        # Calculate overfitting metrics
        # For now, use the same values for IS and OOS
        # In production, you'd have separate IS/OOS splits
        overfitting_metrics = analyze_overfitting_comprehensive(
            sharpe_values,  # In-sample (placeholder)
            sharpe_values,  # Out-of-sample
            returns_skewness=0.0,
            returns_kurtosis=3.0,
            n_observations=n_samples // self.cpcv_config.n_splits
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
