"""
Simple Moving Average Crossover Strategy Example

This example demonstrates how to create and run a basic moving average
crossover strategy using the BackT framework.
"""

import sys
from pathlib import Path

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, Any

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers


def moving_average_crossover_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Simple moving average crossover strategy

    Buy when short MA crosses above long MA
    Sell when short MA crosses below long MA
    """

    # Get parameters
    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < long_period:
            continue  # Not enough data

        # Calculate moving averages
        short_ma = TechnicalIndicators.sma(data['close'], short_period)
        long_ma = TechnicalIndicators.sma(data['close'], long_period)

        # Check for crossover signals
        if StrategyHelpers.is_crossover(short_ma, long_ma):
            # Buy signal
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)

        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            # Sell signal
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def run_example():
    """Run the moving average crossover strategy example"""

    # Create configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-01-01',
        initial_capital=100000,
        data_frequency='1D',
        verbose=True
    )

    # Create backtester
    backtester = Backtester(config)

    # Define strategy parameters
    strategy_params = {
        'short_period': 20,
        'long_period': 50
    }

    # Run backtest
    print("Running moving average crossover strategy...")
    result = backtester.run(
        strategy=moving_average_crossover_strategy,
        universe=['AAPL'],  # Test with Apple stock
        strategy_params=strategy_params
    )

    # Print results
    print("\n=== Backtest Results ===")
    print(f"Total Return: {result.performance_metrics.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {result.performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Trades: {len(result.trades)}")

    return result


if __name__ == "__main__":
    # Run the example
    result = run_example()

    # Optional: Create plots if matplotlib is available
    try:
        from backt.reporting import PlotGenerator

        plotter = PlotGenerator()
        plotter.plot_equity_curve(result.equity_curve, "Moving Average Crossover - AAPL")
        plotter.plot_drawdown(result.equity_curve, "Drawdown - AAPL")

    except ImportError:
        print("Matplotlib not available - skipping plots")