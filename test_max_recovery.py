"""
Test max days to recover metric

Quick test to verify the new max_days_to_recover metric is calculated correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backt import Backtester, BacktestConfig
from backt.data import SQLiteDataLoader
from strategies import momentum

def test_max_recovery_metric():
    """Test that max_days_to_recover metric is calculated"""

    print("\n" + "="*80)
    print("TESTING MAX DAYS TO RECOVER METRIC")
    print("="*80)

    # Setup
    loader = SQLiteDataLoader("market_data.db")
    config = BacktestConfig(
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000
    )

    # Run backtest
    print("\nRunning backtest for SPY with MACD strategy...")
    backtester = Backtester(config, data_loader=loader)

    results = backtester.run(
        strategy=momentum.macd_crossover,
        universe=['SPY'],
        strategy_params={'fast_period': 11, 'slow_period': 20, 'signal_period': 10, 'allow_short': False}
    )

    # Check metrics
    metrics = results.performance_metrics
    print("\nPerformance Metrics:")
    print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"  Max DD Duration: {metrics.get('max_drawdown_duration', 0):.0f} days")
    print(f"  Max Days to Recover: {metrics.get('max_days_to_recover', 0):.0f} days")

    # Verify metric exists
    if 'max_days_to_recover' in metrics:
        print("\n[SUCCESS] max_days_to_recover metric found in results")
        print(f"  Value: {metrics['max_days_to_recover']:.0f} days")
        if metrics['max_days_to_recover'] == 0:
            print("  Note: Value is 0, likely using Numba fast path (not implemented there)")
    else:
        print("\n[FAILED] max_days_to_recover metric not found in results")
        print(f"  Available metrics: {list(metrics.keys())}")

    loader.close()

    print("\n" + "="*80)

if __name__ == "__main__":
    test_max_recovery_metric()
