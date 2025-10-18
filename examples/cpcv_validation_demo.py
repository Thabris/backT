"""
CPCV Validation Demo - Professional Strategy Validation

This example demonstrates how to use Combinatorial Purged Cross-Validation (CPCV)
to validate trading strategies and detect overfitting.

Run with: python examples/cpcv_validation_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backt import Backtester, BacktestConfig, CPCVValidator, CPCVConfig, ParameterGrid
from backt.signal import TechnicalIndicators, StrategyHelpers
import pandas as pd


def momentum_strategy(market_data, current_time, positions, context, params):
    """
    Time Series Momentum Strategy

    Goes long when price > moving average, otherwise cash.

    Parameters:
        lookback: int, lookback period for momentum (default=20)
        threshold: float, minimum momentum to enter (default=0.0)
    """
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.0)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < lookback + 1:
            continue

        # Calculate momentum signal
        current_price = data['close'].iloc[-1]
        ma = TechnicalIndicators.sma(data['close'], lookback)
        momentum = (current_price / ma.iloc[-1] - 1) if ma.iloc[-1] > 0 else 0

        # Generate orders
        if momentum > threshold:
            # Go long with equal weight
            orders[symbol] = StrategyHelpers.create_target_weight_order(1.0 / len(market_data))
        else:
            # Exit to cash
            orders[symbol] = StrategyHelpers.create_target_weight_order(0.0)

    return orders


def mean_reversion_strategy(market_data, current_time, positions, context, params):
    """
    RSI Mean Reversion Strategy

    Buys when oversold, sells when overbought.

    Parameters:
        rsi_period: int, RSI calculation period (default=14)
        oversold: float, oversold threshold (default=30)
        overbought: float, overbought threshold (default=70)
    """
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < rsi_period + 1:
            continue

        rsi = TechnicalIndicators.rsi(data['close'], rsi_period)
        current_rsi = rsi.iloc[-1]

        if current_rsi < oversold:
            orders[symbol] = StrategyHelpers.create_target_weight_order(1.0)
        elif current_rsi > overbought:
            orders[symbol] = StrategyHelpers.create_target_weight_order(0.0)

    return orders


def run_basic_cpcv_example():
    """Example 1: Basic CPCV validation of a single strategy"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic CPCV Validation")
    print("="*80)

    # Setup backtest configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        use_mock_data=False,  # Use real Yahoo Finance data
        verbose=False
    )

    # Setup CPCV configuration
    cpcv_config = CPCVConfig(
        n_splits=10,          # 10 folds
        n_test_splits=2,      # 2 test folds per combination = C(10,2) = 45 paths
        purge_pct=0.05,       # Purge 5% of data around test sets
        embargo_pct=0.02,     # Embargo 2% after test sets
        acceptable_pbo=0.5,   # Max acceptable PBO
        acceptable_dsr=1.0,   # Min acceptable DSR
        acceptable_degradation=30.0  # Max acceptable degradation %
    )

    # Create validator
    print("\nInitializing CPCV Validator...")
    validator = CPCVValidator(config, cpcv_config)

    # Run validation on momentum strategy
    print("\nRunning CPCV validation on Momentum Strategy...")
    print("This will run 45 different train/test combinations...")

    result = validator.validate(
        strategy=momentum_strategy,
        symbols=['SPY'],  # Start with single symbol for speed
        strategy_params={'lookback': 20, 'threshold': 0.0}
    )

    # Display results
    print("\n" + "-"*80)
    print("VALIDATION RESULTS")
    print("-"*80)
    print(f"Paths Completed: {result.n_paths}/45")
    print(f"Mean Sharpe Ratio: {result.mean_sharpe:.3f} ± {result.std_sharpe:.3f}")
    print(f"Mean Return: {result.mean_return:.2%}")
    print(f"Mean Max Drawdown: {result.mean_max_drawdown:.2%}")
    print()
    print("Overfitting Metrics:")
    print(f"  PBO (Probability of Backtest Overfitting): {result.overfitting_metrics.pbo:.2%}")
    print(f"  DSR (Deflated Sharpe Ratio): {result.overfitting_metrics.dsr:.3f}")
    print(f"  Performance Degradation: {result.overfitting_metrics.degradation_pct:.1f}%")
    print(f"  Sharpe Stability: {result.overfitting_metrics.sharpe_stability:.2f}")
    print()
    print("Interpretations:")
    for metric, interpretation in result.overfitting_interpretations.items():
        print(f"  {metric}: {interpretation}")
    print()
    print(f"Passes Validation: {result.passes_validation()}")

    if result.validation_warnings:
        print("\nWarnings:")
        for warning in result.validation_warnings:
            print(f"  ⚠️  {warning}")

    print("-"*80)

    return result


def run_parameter_optimization_with_cpcv():
    """Example 2: Parameter optimization with CPCV validation"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Parameter Optimization with CPCV")
    print("="*80)

    # Setup configurations
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        use_mock_data=False,
        verbose=False
    )

    cpcv_config = CPCVConfig(
        n_splits=5,  # Use fewer splits for faster demonstration
        n_test_splits=2,
        purge_pct=0.05,
        embargo_pct=0.02
    )

    # Define parameter grid
    param_grid = ParameterGrid({
        'lookback': [10, 20, 30],
        'threshold': [0.0, 0.01, 0.02]
    })

    print(f"\nTesting {len(param_grid)} parameter combinations...")
    print("Each combination will be validated across multiple paths...\n")

    # Test each parameter combination
    results = []

    for i, params in enumerate(param_grid):
        print(f"Testing {i+1}/{len(param_grid)}: {params}")

        validator = CPCVValidator(config, cpcv_config)
        result = validator.validate(
            strategy=momentum_strategy,
            symbols=['SPY'],
            strategy_params=params
        )

        results.append({
            'params': params,
            'mean_sharpe': result.mean_sharpe,
            'pbo': result.overfitting_metrics.pbo,
            'dsr': result.overfitting_metrics.dsr,
            'passes': result.passes_validation()
        })

    # Display results
    print("\n" + "-"*80)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("-"*80)

    results_df = pd.DataFrame(results)
    print("\nAll Parameter Sets:")
    for i, row in results_df.iterrows():
        status = "✅ PASS" if row['passes'] else "❌ FAIL"
        print(f"{i+1}. {row['params']}")
        print(f"   Sharpe: {row['mean_sharpe']:.3f} | PBO: {row['pbo']:.2%} | DSR: {row['dsr']:.2f} | {status}")

    # Best parameter set
    best_idx = results_df['mean_sharpe'].idxmax()
    best = results_df.loc[best_idx]

    print("\n" + "="*80)
    print("BEST PARAMETER SET")
    print("="*80)
    print(f"Parameters: {best['params']}")
    print(f"Mean Sharpe: {best['mean_sharpe']:.3f}")
    print(f"PBO: {best['pbo']:.2%}")
    print(f"DSR: {best['dsr']:.3f}")
    print(f"Passes Validation: {'✅ YES' if best['passes'] else '❌ NO'}")
    print("="*80)

    return results


def run_comparison_example():
    """Example 3: Compare multiple strategies with CPCV"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Strategy Comparison with CPCV")
    print("="*80)

    config = BacktestConfig(
        start_date='2021-01-01',
        end_date='2023-12-31',
        initial_capital=100000,
        use_mock_data=False,
        verbose=False
    )

    cpcv_config = CPCVConfig(n_splits=5, n_test_splits=2)

    strategies = {
        'Momentum (20d)': (momentum_strategy, {'lookback': 20}),
        'Momentum (50d)': (momentum_strategy, {'lookback': 50}),
        'Mean Reversion (RSI 14)': (mean_reversion_strategy, {'rsi_period': 14}),
    }

    print("\nComparing strategies across multiple validation paths...\n")

    comparison = []

    for name, (strategy_func, params) in strategies.items():
        print(f"Validating: {name}")

        validator = CPCVValidator(config, cpcv_config)
        result = validator.validate(
            strategy=strategy_func,
            symbols=['SPY'],
            strategy_params=params
        )

        comparison.append({
            'Strategy': name,
            'Mean Sharpe': result.mean_sharpe,
            'Std Sharpe': result.std_sharpe,
            'PBO': result.overfitting_metrics.pbo,
            'DSR': result.overfitting_metrics.dsr,
            'Stability': result.overfitting_metrics.sharpe_stability,
            'Passes': result.passes_validation()
        })

    # Display comparison
    print("\n" + "-"*80)
    print("STRATEGY COMPARISON")
    print("-"*80)

    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))

    print("\nBest Strategy (by Sharpe):")
    best = comp_df.loc[comp_df['Mean Sharpe'].idxmax()]
    print(f"  {best['Strategy']}")
    print(f"  Sharpe: {best['Mean Sharpe']:.3f} ± {best['Std Sharpe']:.3f}")
    print(f"  PBO: {best['PBO']:.2%} | DSR: {best['DSR']:.2f}")
    print("-"*80)

    return comparison


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" CPCV VALIDATION DEMONSTRATION")
    print(" Professional Strategy Validation with Overfitting Detection")
    print("="*80)

    # Example 1: Basic validation
    print("\n📊 Running basic CPCV validation example...")
    result1 = run_basic_cpcv_example()

    # Example 2: Parameter optimization
    print("\n🔍 Running parameter optimization example...")
    result2 = run_parameter_optimization_with_cpcv()

    # Example 3: Strategy comparison
    print("\n⚖️  Running strategy comparison example...")
    result3 = run_comparison_example()

    print("\n" + "="*80)
    print(" DEMONSTRATION COMPLETE")
    print("="*80)
    print("\n✅ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("  • CPCV provides robust validation across multiple train/test paths")
    print("  • PBO < 0.5 indicates low overfitting risk")
    print("  • DSR > 1.0 indicates statistically significant performance")
    print("  • Stability metric shows consistency across paths")
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run just the basic example for quick demonstration
    # Uncomment main() to run all examples

    print("Running CPCV Basic Validation Example...")
    print("(This uses real Yahoo Finance data and may take a minute)")

    run_basic_cpcv_example()

    print("\n💡 To run all examples (including parameter optimization and comparisons),")
    print("   uncomment the main() call at the bottom of this file.")
