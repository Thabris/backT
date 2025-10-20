"""
FLAML-based parameter optimizer for BackT

Uses Microsoft's Fast and Lightweight AutoML (FLAML) library for
intelligent parameter search with cost-frugal optimization.

FLAML achieves same/better results with ~10% of the computation of grid search
by starting with cheap configurations and gradually moving to expensive ones.
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress FLAML automl warnings (we only need flaml.tune, not automl)
warnings.filterwarnings('ignore', message='.*flaml.automl is not available.*')

try:
    from flaml import tune
    HAS_FLAML = True
except ImportError:
    HAS_FLAML = False
    warnings.warn("FLAML not installed. Install with: pip install flaml")

from ..engine.backtester import Backtester
from ..utils.config import BacktestConfig
from ..utils.types import BacktestResult
from ..utils.logging_config import LoggerMixin
from .results import OptimizationResult as NewOptimizationResult
from .results import ParameterSetResult


class FLAMLOptimizer(LoggerMixin):
    """
    FLAML-powered intelligent parameter optimizer

    Uses Cost-Frugal Optimization (CFO) to find optimal parameters with
    10% of the computation of grid search.

    Key Features:
    - Intelligent search: Learns which parameter regions are promising
    - Cost-aware: Starts with cheap configs, moves to expensive only when needed
    - Fast convergence: Achieves good results in fraction of grid search time
    - Flexible: Supports continuous and discrete parameter spaces

    Example:
        >>> optimizer = FLAMLOptimizer(
        ...     strategy_function=my_strategy,
        ...     config=backtest_config,
        ...     symbols=['AAPL', 'MSFT']
        ... )
        >>>
        >>> param_space = {
        ...     'fast_ma': {'domain': tune.randint(5, 50)},
        ...     'slow_ma': {'domain': tune.randint(50, 200)},
        ...     'threshold': {'domain': tune.uniform(0.01, 0.10)}
        ... }
        >>>
        >>> result = optimizer.optimize(
        ...     param_space=param_space,
        ...     optimization_metric='sharpe_ratio',
        ...     time_budget_s=600,  # 10 minutes
        ...     num_samples=-1  # Unlimited until budget exhausted
        ... )
        >>>
        >>> print(f"Best params: {result.best_params}")
        >>> print(f"Evaluations: {result.n_evaluations}")
    """

    def __init__(
        self,
        strategy_function: Callable,
        config: BacktestConfig,
        symbols: List[str],
        fixed_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FLAML optimizer

        Args:
            strategy_function: Strategy function to optimize
            config: Backtesting configuration
            symbols: List of symbols to trade
            fixed_params: Parameters that remain constant
        """
        if not HAS_FLAML:
            raise ImportError(
                "FLAML is required for FLAMLOptimizer. "
                "Install with: pip install flaml"
            )

        self.strategy_function = strategy_function
        self.config = config
        self.symbols = symbols
        self.fixed_params = fixed_params or {}

        # Cache for market data
        self._data_cache = None
        self._backtester = None

        # Track evaluations
        self._evaluation_count = 0
        self._all_results = []

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

            self.logger.info(f"Data loaded and cached for FLAML optimization")

    def _evaluate_config(
        self,
        config_dict: Dict[str, Any],
        optimization_metric: str,
        minimize: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single parameter configuration

        This is called by FLAML for each parameter set to test.

        Args:
            config_dict: Parameter configuration from FLAML
            optimization_metric: Metric to optimize
            minimize: If True, minimize metric; if False, maximize

        Returns:
            Dictionary with metric value (negated if maximizing)
        """
        try:
            # Merge with fixed params
            params = {**self.fixed_params, **config_dict}

            # Create cached data loader (like GridSearchOptimizer does)
            from ..data.loaders import DataLoader

            class CachedDataLoader(DataLoader):
                def __init__(self, cached_data_dict):
                    self.cached_data = cached_data_dict

                def load(self, symbols, start_date, end_date):
                    result = {}
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
                    return result

                def validate_data(self, data):
                    return True

            # Create backtester with cached data loader
            cached_loader = CachedDataLoader(self._data_cache)
            backtester = Backtester(self.config, data_loader=cached_loader)

            # Run backtest
            start_time = datetime.now()
            result = backtester.run(
                strategy=self.strategy_function,
                universe=self.symbols,
                strategy_params=params
            )
            end_time = datetime.now()
            backtest_time = (end_time - start_time).total_seconds()

            # Extract metrics
            metrics = result.performance_metrics.copy()
            metric_value = metrics.get(optimization_metric, np.nan)

            # Track evaluation
            self._evaluation_count += 1

            # Store result for later analysis
            param_result = ParameterSetResult(
                parameters=params,
                metrics=metrics,
                backtest_time_seconds=backtest_time,
                evaluation_id=self._evaluation_count
            )
            self._all_results.append(param_result)

            # Log progress
            if self._evaluation_count % 10 == 0:
                self.logger.info(
                    f"Evaluation {self._evaluation_count}: "
                    f"{optimization_metric}={metric_value:.4f}, "
                    f"params={params}"
                )

            # FLAML minimizes by default, so negate if we want to maximize
            if minimize:
                return {optimization_metric: metric_value}
            else:
                return {optimization_metric: -metric_value}

        except Exception as e:
            self.logger.warning(f"Evaluation failed for {config_dict}: {e}")
            # Return worst possible score
            if minimize:
                return {optimization_metric: np.inf}
            else:
                return {optimization_metric: -np.inf}

    def optimize(
        self,
        param_space: Dict[str, Dict[str, Any]],
        optimization_metric: str = 'sharpe_ratio',
        minimize: bool = False,
        time_budget_s: int = 600,
        num_samples: int = -1,
        verbose: int = 3
    ) -> NewOptimizationResult:
        """
        Run FLAML intelligent parameter search

        Args:
            param_space: FLAML parameter space specification
                Example: {
                    'fast_ma': {'domain': tune.randint(5, 50)},
                    'slow_ma': {'domain': tune.randint(50, 200)},
                    'threshold': {'domain': tune.uniform(0.01, 0.10)}
                }
            optimization_metric: Metric to optimize (e.g., 'sharpe_ratio')
            minimize: If True, minimize metric; if False, maximize
            time_budget_s: Time budget in seconds
            num_samples: Max number of evaluations (-1 = unlimited)
            verbose: FLAML verbosity level (0-3)

        Returns:
            OptimizationResult with best parameters and all evaluated configs
        """
        start_time = datetime.now()

        # Load data once
        self._load_data_once()

        # Reset evaluation tracking
        self._evaluation_count = 0
        self._all_results = []

        self.logger.info(f"Starting FLAML optimization")
        self.logger.info(f"Time budget: {time_budget_s} seconds")
        self.logger.info(f"Max samples: {num_samples if num_samples > 0 else 'unlimited'}")
        self.logger.info(f"Optimizing for: {optimization_metric} ({'minimize' if minimize else 'maximize'})")

        # Create evaluation function for FLAML
        def objective(config):
            return self._evaluate_config(config, optimization_metric, minimize)

        # Prepare search space for FLAML
        flaml_config = {}
        for param_name, param_config in param_space.items():
            if 'domain' in param_config:
                flaml_config[param_name] = param_config['domain']
            else:
                raise ValueError(
                    f"Parameter '{param_name}' must have 'domain' key. "
                    f"Example: {{'domain': tune.randint(5, 50)}}"
                )

        # Run FLAML optimization
        try:
            analysis = tune.run(
                objective,
                config=flaml_config,
                metric=optimization_metric,
                mode='min' if minimize else 'max',  # Note: we negate in objective, so mode is always max
                num_samples=num_samples,
                time_budget_s=time_budget_s,
                verbose=verbose
            )

            # Get best result
            best_trial = analysis.best_trial
            best_params = {**self.fixed_params, **best_trial.config}

            # Un-negate the metric if we were maximizing
            if minimize:
                best_metric_value = best_trial.last_result[optimization_metric]
            else:
                best_metric_value = -best_trial.last_result[optimization_metric]

            # Find the corresponding ParameterSetResult
            best_param_result = None
            for result in self._all_results:
                if result.parameters == best_params:
                    best_param_result = result
                    break

        except Exception as e:
            self.logger.error(f"FLAML optimization failed: {e}")
            raise

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        self.logger.info(f"FLAML optimization complete in {total_time:.2f} seconds")
        self.logger.info(f"Total evaluations: {self._evaluation_count}")
        self.logger.info(f"Best {optimization_metric}: {best_metric_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")

        # Create OptimizationResult
        return NewOptimizationResult(
            best_parameters=best_params,
            best_metrics=best_param_result.metrics if best_param_result else {},
            all_results=self._all_results,
            method='flaml',
            optimization_metric=optimization_metric,
            total_evaluations=self._evaluation_count,
            total_time_seconds=total_time,
            start_time=start_time,
            end_time=end_time,
            param_space=param_space
        )

    def optimize_with_cpcv(
        self,
        param_space: Dict[str, Dict[str, Any]],
        optimization_metric: str = 'sharpe_ratio',
        minimize: bool = False,
        time_budget_s: int = 600,
        num_samples: int = -1,
        top_k: int = 3,
        cpcv_config: Optional[Any] = None,
        verbose: int = 3
    ) -> NewOptimizationResult:
        """
        Run FLAML optimization followed by CPCV validation on top K parameters

        This implements the 2-tier optimization strategy:
        1. FLAML intelligent search to find top K parameter sets
        2. CPCV validation on top K to detect overfitting

        Args:
            param_space: FLAML parameter space
            optimization_metric: Metric to optimize
            minimize: If True, minimize; if False, maximize
            time_budget_s: FLAML time budget in seconds
            num_samples: FLAML max samples
            top_k: Number of top parameters to validate with CPCV
            cpcv_config: CPCVConfig instance for validation
            verbose: FLAML verbosity

        Returns:
            OptimizationResult with CPCV validation on top K parameters
        """
        # Step 1: Run FLAML optimization
        self.logger.info("=" * 80)
        self.logger.info("TIER 1: FLAML INTELLIGENT SEARCH")
        self.logger.info("=" * 80)

        result = self.optimize(
            param_space=param_space,
            optimization_metric=optimization_metric,
            minimize=minimize,
            time_budget_s=time_budget_s,
            num_samples=num_samples,
            verbose=verbose
        )

        # Step 2: CPCV validation on top K
        if cpcv_config is not None:
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info(f"TIER 2: CPCV VALIDATION ON TOP {top_k} PARAMETERS")
            self.logger.info("=" * 80)

            from ..validation.cpcv_validator import CPCVValidator

            # Get top K parameter sets
            top_k_results = result.get_top_k(top_k)

            # Run CPCV on each
            cpcv_results = []
            for idx, param_result in enumerate(top_k_results, 1):
                self.logger.info(f"\nValidating parameter set {idx}/{top_k}: {param_result.parameters}")

                # Run CPCV - pass both backtest_config and cpcv_config
                validator = CPCVValidator(self.config, cpcv_config)
                cpcv_result = validator.validate(
                    strategy=self.strategy_function,
                    symbols=self.symbols,
                    strategy_params=param_result.parameters
                )

                # Store CPCV result
                param_result.cpcv_result = cpcv_result
                cpcv_results.append(cpcv_result)

                self.logger.info(f"  PBO: {cpcv_result.overfitting_metrics.pbo:.3f}")
                self.logger.info(f"  DSR: {cpcv_result.overfitting_metrics.dsr:.3f}")
                self.logger.info(f"  Degradation: {cpcv_result.overfitting_metrics.degradation_pct:.1f}%")

            # Update result with CPCV validation
            result.cpcv_validated = True
            result.top_k_cpcv_results = cpcv_results

            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("VALIDATION COMPLETE")
            self.logger.info("=" * 80)

        return result
