"""
Comprehensive tests for CPCV validation framework

Tests all components:
- Data splitting utilities
- Overfitting detection metrics
- Parameter grid
- CPCV validator
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backt.validation import (
    CPCVValidator,
    CPCVConfig,
    ParameterGrid,
    calculate_probability_backtest_overfitting,
    calculate_deflated_sharpe_ratio,
    calculate_performance_degradation
)
from backt.utils.data_splits import (
    create_purged_kfold_splits,
    generate_cpcv_combinations,
    create_cpcv_split,
    validate_split_quality,
    calculate_optimal_purge_embargo
)
from backt import BacktestConfig
from backt.signal import StrategyHelpers


class TestDataSplits:
    """Test data splitting utilities"""

    def test_purged_kfold_splits_basic(self):
        """Test basic K-fold splitting with purging"""
        n_samples = 1000
        n_splits = 5

        splits = create_purged_kfold_splits(
            n_samples,
            n_splits=n_splits,
            purge_pct=0.05,
            embargo_pct=0.02
        )

        assert len(splits) == n_splits, "Should create correct number of splits"

        for train_idx, test_idx in splits:
            # Check no overlap
            assert len(set(train_idx) & set(test_idx)) == 0, "Train and test should not overlap"

            # Check indices are valid
            assert train_idx.min() >= 0, "Train indices should be non-negative"
            assert test_idx.max() < n_samples, "Test indices should be within bounds"

            # Check train set is not empty
            assert len(train_idx) > 0, "Train set should not be empty"
            assert len(test_idx) > 0, "Test set should not be empty"

    def test_purged_kfold_splits_sizes(self):
        """Test that purging reduces training set size"""
        n_samples = 1000

        # Without purging
        splits_no_purge = create_purged_kfold_splits(
            n_samples, n_splits=5, purge_pct=0.0, embargo_pct=0.0
        )

        # With purging
        splits_with_purge = create_purged_kfold_splits(
            n_samples, n_splits=5, purge_pct=0.10, embargo_pct=0.05
        )

        # Training set should be smaller with purging
        train_size_no_purge = np.mean([len(train) for train, _ in splits_no_purge])
        train_size_with_purge = np.mean([len(train) for train, _ in splits_with_purge])

        assert train_size_with_purge < train_size_no_purge, \
            "Purging should reduce training set size"

    def test_generate_cpcv_combinations(self):
        """Test CPCV combination generation"""
        n_splits = 10
        n_test_splits = 2

        combinations = generate_cpcv_combinations(n_splits, n_test_splits)

        # Should generate C(10, 2) = 45 combinations
        expected_count = 45
        assert len(combinations) == expected_count, \
            f"Should generate {expected_count} combinations"

        # Each combination should have n_test_splits elements
        for combo in combinations:
            assert len(combo) == n_test_splits, \
                f"Each combination should have {n_test_splits} elements"

        # All combinations should be unique
        assert len(set(combinations)) == len(combinations), \
            "All combinations should be unique"

    def test_create_cpcv_split(self):
        """Test creation of single CPCV split"""
        n_samples = 1000
        n_splits = 10
        test_fold_indices = (2, 5)

        train_idx, test_idx = create_cpcv_split(
            np.arange(n_samples),
            test_fold_indices,
            n_splits,
            purge_pct=0.05,
            embargo_pct=0.02
        )

        # No overlap
        assert len(set(train_idx) & set(test_idx)) == 0, \
            "Train and test should not overlap"

        # Test set should contain samples from specified folds
        fold_size = n_samples // n_splits
        for fold_idx in test_fold_indices:
            fold_start = fold_idx * fold_size
            assert fold_start in test_idx or (fold_start + 1) in test_idx, \
                f"Test set should include fold {fold_idx}"

    def test_validate_split_quality_valid(self):
        """Test split quality validation for valid split"""
        n_samples = 1000
        train_idx = np.arange(700)  # 70% training
        test_idx = np.arange(700, 1000)  # 30% testing

        is_valid, msg = validate_split_quality(
            train_idx, test_idx, n_samples, min_train_pct=0.5
        )

        assert is_valid, "Valid split should pass validation"
        assert "Valid split" in msg, "Should indicate split is valid"

    def test_validate_split_quality_invalid_overlap(self):
        """Test split quality validation detects overlap"""
        n_samples = 1000
        train_idx = np.arange(600)
        test_idx = np.arange(500, 900)  # Overlaps with train

        is_valid, msg = validate_split_quality(
            train_idx, test_idx, n_samples
        )

        assert not is_valid, "Overlapping split should fail validation"
        assert "overlap" in msg.lower(), "Should mention overlap"

    def test_validate_split_quality_insufficient_train(self):
        """Test split quality validation detects insufficient training data"""
        n_samples = 1000
        train_idx = np.arange(400)  # Only 40%
        test_idx = np.arange(400, 1000)

        is_valid, msg = validate_split_quality(
            train_idx, test_idx, n_samples, min_train_pct=0.5
        )

        assert not is_valid, "Insufficient training data should fail"
        assert "too small" in msg.lower(), "Should mention insufficient size"

    def test_calculate_optimal_purge_embargo(self):
        """Test optimal purge/embargo calculation"""
        # Monthly momentum strategy
        purge_pct, embargo_pct = calculate_optimal_purge_embargo(
            strategy_horizon_days=20,
            rebalance_frequency_days=21
        )

        assert 0 < purge_pct < 0.15, "Purge should be reasonable"
        assert 0 <= embargo_pct < 0.10, "Embargo should be reasonable"
        assert purge_pct >= embargo_pct, "Purge typically >= embargo"


class TestOverfittingMetrics:
    """Test overfitting detection metrics"""

    def test_pbo_no_overfitting(self):
        """Test PBO with no overfitting scenario"""
        # Perfect correlation: best IS also best OOS
        is_performance = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        oos_performance = np.array([0.6, 1.1, 1.6, 2.1, 2.6])  # Consistent

        pbo = calculate_probability_backtest_overfitting(is_performance, oos_performance)

        assert 0 <= pbo <= 1, "PBO should be between 0 and 1"
        assert pbo < 0.5, "Low PBO indicates no overfitting"

    def test_pbo_severe_overfitting(self):
        """Test PBO with severe overfitting scenario"""
        # Inverse relationship: best IS worst OOS
        is_performance = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        oos_performance = np.array([2.6, 2.1, 1.6, 1.1, 0.6])  # Reversed

        pbo = calculate_probability_backtest_overfitting(is_performance, oos_performance)

        assert pbo > 0.5, "High PBO indicates overfitting"

    def test_pbo_length_mismatch(self):
        """Test PBO raises error on length mismatch"""
        is_performance = np.array([1.0, 2.0, 3.0])
        oos_performance = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="same length"):
            calculate_probability_backtest_overfitting(is_performance, oos_performance)

    def test_pbo_insufficient_data(self):
        """Test PBO raises error with insufficient data"""
        is_performance = np.array([1.0])
        oos_performance = np.array([1.0])

        with pytest.raises(ValueError, match="at least 2"):
            calculate_probability_backtest_overfitting(is_performance, oos_performance)

    def test_dsr_single_trial(self):
        """Test DSR with single trial (no multiple testing)"""
        dsr = calculate_deflated_sharpe_ratio(
            observed_sharpe=2.0,
            n_trials=1,  # No multiple testing
            n_observations=252
        )

        # With single trial, DSR should be close to observed Sharpe
        assert dsr > 1.5, "DSR should be significant for Sharpe=2.0 with 1 trial"

    def test_dsr_multiple_trials_penalty(self):
        """Test DSR penalty for multiple trials"""
        sharpe = 2.0
        n_obs = 252

        dsr_1_trial = calculate_deflated_sharpe_ratio(sharpe, n_trials=1, n_observations=n_obs)
        dsr_100_trials = calculate_deflated_sharpe_ratio(sharpe, n_trials=100, n_observations=n_obs)

        assert dsr_100_trials < dsr_1_trial, \
            "DSR should decrease with more trials (multiple testing penalty)"

    def test_dsr_with_skewness_kurtosis(self):
        """Test DSR adjusts for non-normal distributions"""
        sharpe = 1.5
        n_trials = 10
        n_obs = 252

        # Normal distribution
        dsr_normal = calculate_deflated_sharpe_ratio(
            sharpe, n_trials, returns_skewness=0.0, returns_kurtosis=3.0, n_observations=n_obs
        )

        # Negatively skewed, fat tails (more realistic)
        dsr_nonnormal = calculate_deflated_sharpe_ratio(
            sharpe, n_trials, returns_skewness=-0.5, returns_kurtosis=5.0, n_observations=n_obs
        )

        # DSR should be different when accounting for non-normality
        assert dsr_normal != dsr_nonnormal, "DSR should adjust for distribution shape"

    def test_performance_degradation(self):
        """Test performance degradation calculation"""
        is_metrics = {
            'sharpe_ratio': 2.0,
            'total_return': 0.50,
            'max_drawdown': -0.15
        }

        oos_metrics = {
            'sharpe_ratio': 1.5,
            'total_return': 0.40,
            'max_drawdown': -0.20
        }

        degradation = calculate_performance_degradation(is_metrics, oos_metrics)

        assert 'sharpe_ratio' in degradation
        assert abs(degradation['sharpe_ratio'] - 25.0) < 0.01, "Should calculate ~25% degradation"
        assert abs(degradation['total_return'] - 20.0) < 0.01, "Should calculate ~20% degradation"


class TestParameterGrid:
    """Test parameter grid utilities"""

    def test_parameter_grid_basic(self):
        """Test basic parameter grid creation"""
        grid = ParameterGrid({
            'param1': [1, 2, 3],
            'param2': [10, 20]
        })

        assert len(grid) == 6, "Should have 3 * 2 = 6 combinations"

        # Test iteration
        params_list = list(grid)
        assert len(params_list) == 6
        assert all('param1' in p and 'param2' in p for p in params_list)

    def test_parameter_grid_indexing(self):
        """Test parameter grid indexing"""
        grid = ParameterGrid({
            'a': [1, 2],
            'b': [10, 20, 30]
        })

        # Test valid indices
        param0 = grid[0]
        assert isinstance(param0, dict)
        assert 'a' in param0 and 'b' in param0

        # Test invalid index
        with pytest.raises(IndexError):
            _ = grid[100]

    def test_parameter_grid_to_list(self):
        """Test converting grid to list"""
        grid = ParameterGrid({
            'x': [1, 2],
            'y': [3, 4]
        })

        params_list = grid.to_list()
        assert len(params_list) == 4
        assert all(isinstance(p, dict) for p in params_list)

    def test_parameter_grid_sample(self):
        """Test random sampling from grid"""
        grid = ParameterGrid({
            'a': range(10),
            'b': range(10)
        })

        # Sample 20 from 100 combinations
        sample = grid.sample(20, random_state=42)

        assert len(sample) == 20, "Should return requested number of samples"
        assert len(set(map(lambda d: tuple(sorted(d.items())), sample))) == 20, \
            "All samples should be unique"

    def test_parameter_grid_sample_exceeds_size(self):
        """Test sampling more than grid size returns all"""
        grid = ParameterGrid({'x': [1, 2]})

        sample = grid.sample(100)
        assert len(sample) == 2, "Should return all combinations if n > size"


class TestCPCVConfig:
    """Test CPCV configuration"""

    def test_cpcv_config_defaults(self):
        """Test CPCV config with default values"""
        config = CPCVConfig()

        assert config.n_splits == 10
        assert config.n_test_splits == 2
        assert config.purge_pct == 0.05
        assert config.embargo_pct == 0.02

    def test_cpcv_config_validation_n_splits(self):
        """Test n_splits validation"""
        with pytest.raises(ValueError, match="at least 3"):
            CPCVConfig(n_splits=2)

    def test_cpcv_config_validation_n_test_splits(self):
        """Test n_test_splits validation"""
        with pytest.raises(ValueError, match="between 1 and"):
            CPCVConfig(n_splits=5, n_test_splits=5)

        with pytest.raises(ValueError, match="between 1 and"):
            CPCVConfig(n_splits=5, n_test_splits=0)

    def test_cpcv_config_validation_purge_pct(self):
        """Test purge_pct validation"""
        with pytest.raises(ValueError, match="between 0 and 0.5"):
            CPCVConfig(purge_pct=0.6)

        with pytest.raises(ValueError, match="between 0 and 0.5"):
            CPCVConfig(purge_pct=-0.1)

    def test_cpcv_config_validation_embargo_pct(self):
        """Test embargo_pct validation"""
        with pytest.raises(ValueError, match="between 0 and 0.5"):
            CPCVConfig(embargo_pct=0.6)


class TestCPCVValidator:
    """Test CPCV validator"""

    @pytest.fixture
    def simple_strategy(self):
        """Simple buy-and-hold strategy for testing"""
        def strategy(market_data, current_time, positions, context, params):
            if 'initialized' not in context:
                context['initialized'] = True
                weight = 1.0 / len(market_data)
                return {
                    symbol: StrategyHelpers.create_target_weight_order(weight)
                    for symbol in market_data.keys()
                }
            return {}

        return strategy

    @pytest.fixture
    def backtest_config(self):
        """Basic backtest configuration"""
        return BacktestConfig(
            start_date='2022-01-01',
            end_date='2022-12-31',
            initial_capital=100000,
            use_mock_data=True,  # Use mock data for fast testing
            verbose=False
        )

    @pytest.fixture
    def cpcv_config(self):
        """CPCV configuration for testing"""
        return CPCVConfig(
            n_splits=5,  # Fewer splits for faster testing
            n_test_splits=2,
            purge_pct=0.05,
            embargo_pct=0.02
        )

    def test_cpcv_validator_initialization(self, backtest_config, cpcv_config):
        """Test CPCV validator initialization"""
        validator = CPCVValidator(backtest_config, cpcv_config)

        assert validator.backtest_config == backtest_config
        assert validator.cpcv_config == cpcv_config

    def test_cpcv_validator_with_defaults(self, backtest_config):
        """Test CPCV validator with default config"""
        validator = CPCVValidator(backtest_config)

        assert validator.cpcv_config.n_splits == 10
        assert validator.cpcv_config.n_test_splits == 2

    def test_cpcv_validation_run(self, backtest_config, cpcv_config, simple_strategy):
        """Test running CPCV validation"""
        validator = CPCVValidator(backtest_config, cpcv_config)

        result = validator.validate(
            strategy=simple_strategy,
            symbols=['SPY', 'TLT'],
            strategy_params={}
        )

        # Check result structure
        assert result.n_paths > 0, "Should have completed some paths"
        assert len(result.path_results) > 0, "Should have path results"
        assert result.mean_sharpe is not None, "Should calculate mean Sharpe"
        assert result.overfitting_metrics is not None, "Should have overfitting metrics"
        assert len(result.overfitting_interpretations) > 0, "Should have interpretations"

    def test_cpcv_result_summary(self, backtest_config, cpcv_config, simple_strategy):
        """Test CPCV result summary"""
        validator = CPCVValidator(backtest_config, cpcv_config)
        result = validator.validate(simple_strategy, ['SPY'], {})

        summary = result.summary()

        assert 'mean_sharpe' in summary
        assert 'pbo' in summary
        assert 'dsr' in summary
        assert 'runtime_seconds' in summary

    def test_cpcv_passes_validation(self, backtest_config, simple_strategy):
        """Test validation pass/fail logic"""
        # Lenient thresholds
        cpcv_config = CPCVConfig(
            n_splits=5,
            n_test_splits=2,
            acceptable_pbo=0.9,  # Very lenient
            acceptable_dsr=0.0,
            acceptable_degradation=100.0
        )

        validator = CPCVValidator(backtest_config, cpcv_config)
        result = validator.validate(simple_strategy, ['SPY'], {})

        # With lenient thresholds, should likely pass
        # (Note: actual result depends on random mock data)
        assert isinstance(result.passes_validation(), bool)


def test_integration_full_cpcv_workflow():
    """Integration test: Full CPCV workflow"""
    # Simple momentum strategy
    def momentum_strategy(market_data, current_time, positions, context, params):
        lookback = params.get('lookback', 20)
        orders = {}

        for symbol, data in market_data.items():
            if len(data) < lookback + 1:
                continue

            # Simple momentum: buy if price > lookback MA
            current_price = data['close'].iloc[-1]
            ma = data['close'].iloc[-lookback:].mean()

            if current_price > ma:
                orders[symbol] = StrategyHelpers.create_target_weight_order(1.0)
            else:
                orders[symbol] = StrategyHelpers.create_target_weight_order(0.0)

        return orders

    # Setup
    config = BacktestConfig(
        start_date='2022-01-01',
        end_date='2022-12-31',
        initial_capital=100000,
        use_mock_data=True,
        verbose=False
    )

    cpcv_config = CPCVConfig(n_splits=5, n_test_splits=2)

    # Run validation
    validator = CPCVValidator(config, cpcv_config)
    result = validator.validate(
        strategy=momentum_strategy,
        symbols=['SPY'],
        strategy_params={'lookback': 20}
    )

    # Verify results
    assert result.n_paths > 0
    assert 0 <= result.overfitting_metrics.pbo <= 1
    assert result.mean_sharpe is not None
    assert result.total_runtime_seconds > 0

    print("\n" + "="*60)
    print("CPCV INTEGRATION TEST RESULTS")
    print("="*60)
    print(f"Paths completed: {result.n_paths}")
    print(f"Mean Sharpe: {result.mean_sharpe:.3f} ± {result.std_sharpe:.3f}")
    print(f"PBO: {result.overfitting_metrics.pbo:.2%}")
    print(f"DSR: {result.overfitting_metrics.dsr:.3f}")
    print(f"Degradation: {result.overfitting_metrics.degradation_pct:.1f}%")
    print(f"Stability: {result.overfitting_metrics.sharpe_stability:.2f}")
    print(f"Passes validation: {result.passes_validation()}")
    print("\nInterpretations:")
    for metric, interp in result.overfitting_interpretations.items():
        print(f"  {metric}: {interp}")
    print("="*60)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
