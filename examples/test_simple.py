"""
Simple BackT Framework Test

Tests the framework with synthetic data to verify everything works.
"""

import sys
from pathlib import Path

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any

from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers
from backt.data.loaders import CustomDataLoader


def create_synthetic_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # For reproducible results

    # Generate synthetic price data
    n_periods = len(dates)
    returns = np.random.normal(0.001, 0.02, n_periods)

    # Create price series
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLC from prices
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))

    # Ensure valid OHLC relationships
    for i in range(n_periods):
        high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
        low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])

    volumes = np.random.randint(100000, 1000000, n_periods)

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def synthetic_data_loader(symbols, start_date, end_date, **kwargs):
    """Custom data loader that creates synthetic data"""
    if isinstance(symbols, str):
        symbols = [symbols]

    data = {}
    for symbol in symbols:
        data[symbol] = create_synthetic_data(symbol, start_date, end_date)

    return data


def simple_strategy(market_data, current_time, positions, context, params):
    """Simple moving average strategy"""
    orders = {}

    for symbol, data in market_data.items():
        if len(data) < 20:
            continue

        # Calculate moving averages
        short_ma = TechnicalIndicators.sma(data['close'], 10)
        long_ma = TechnicalIndicators.sma(data['close'], 20)

        # Simple crossover logic
        if len(short_ma) >= 2 and len(long_ma) >= 2:
            if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
                # Buy signal
                orders[symbol] = {'action': 'target_weight', 'weight': 1.0}
            elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
                # Sell signal
                orders[symbol] = {'action': 'target_weight', 'weight': 0.0}

    return orders


def main():
    """Main test function"""
    print("Testing BackT Framework")
    print("=" * 40)

    # Create configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2020-06-30',
        initial_capital=100000,
        verbose=True
    )

    # Create custom data loader
    data_loader = CustomDataLoader(synthetic_data_loader)

    # Create backtester
    backtester = Backtester(config, data_loader=data_loader)

    print("\nRunning backtest...")

    # Run backtest
    result = backtester.run(
        strategy=simple_strategy,
        universe=['TEST'],
        strategy_params={}
    )

    # Display results
    print("\nRESULTS:")
    print("-" * 20)

    metrics = result.performance_metrics
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Trades: {len(result.trades)}")
    print(f"Runtime: {result.total_runtime_seconds:.2f} seconds")

    print("\nSUCCESS: BackT framework is working!")
    print("You can now install yfinance to use real market data.")

    return result


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()