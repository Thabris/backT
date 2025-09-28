"""
Buy and Hold Strategy Example

This example demonstrates a simple buy and hold strategy
that purchases stocks at the beginning and holds them.
"""

import sys
from pathlib import Path

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from typing import Dict, Any

from backt import Backtester, BacktestConfig
from backt.signal import StrategyHelpers


def buy_and_hold_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Buy and hold strategy - buy once at the beginning and hold
    """

    # Initialize context if first run
    if 'initialized' not in context:
        context['initialized'] = True

        # Buy all symbols with equal weight
        target_weight = 1.0 / len(market_data)
        orders = {}

        for symbol in market_data.keys():
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=target_weight)

        return orders

    # After initial purchase, hold (no new orders)
    return {}


def run_buy_and_hold():
    """Run buy and hold strategy example"""

    # Create configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-01-01',
        initial_capital=100000,
        verbose=True
    )

    # Create backtester
    backtester = Backtester(config)

    # Run backtest on multiple stocks
    print("Running buy and hold strategy...")
    result = backtester.run(
        strategy=buy_and_hold_strategy,
        universe=['AAPL', 'MSFT', 'GOOGL'],
        strategy_params={}
    )

    # Print results
    print("\n=== Buy and Hold Results ===")
    print(f"Total Return: {result.performance_metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {result.performance_metrics.get('cagr', 0):.2%}")
    print(f"Volatility: {result.performance_metrics.get('annualized_volatility', 0):.2%}")
    print(f"Sharpe Ratio: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {result.performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Trades: {len(result.trades)}")

    return result


if __name__ == "__main__":
    result = run_buy_and_hold()