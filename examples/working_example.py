"""
Working BackT Strategy Example

A complete working example that handles data loading gracefully
and demonstrates all framework features.
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
from backt.data.loaders import YahooDataLoader, CustomDataLoader


def create_synthetic_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Create realistic synthetic OHLCV data"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    np.random.seed(hash(symbol) % 1000)
    n_periods = len(dates)
    base_price = 150.0  # AAPL-like price

    # Create realistic returns
    returns = np.random.normal(0.0005, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLC
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    noise = 0.005
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, noise, n_periods)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, noise, n_periods)))

    volumes = np.random.randint(500000, 2000000, n_periods)

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def synthetic_data_loader(symbols, start_date, end_date, **kwargs):
    """Synthetic data loader"""
    if isinstance(symbols, str):
        symbols = [symbols]

    print(f"Creating synthetic data for {symbols}")
    return {symbol: create_synthetic_data(symbol, start_date, end_date) for symbol in symbols}


def moving_average_strategy(market_data, current_time, positions, context, params):
    """Moving average crossover strategy"""
    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < long_period:
            continue

        short_ma = TechnicalIndicators.sma(data['close'], short_period)
        long_ma = TechnicalIndicators.sma(data['close'], long_period)

        if StrategyHelpers.is_crossover(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)
        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def main():
    """Main function"""
    print("BackT Framework - Complete Example")
    print("=" * 40)

    # Configuration
    config = BacktestConfig(
        start_date='2022-01-01',
        end_date='2023-01-01',
        initial_capital=100000,
        verbose=True
    )

    strategy_params = {
        'short_period': 20,
        'long_period': 50
    }

    # Try real data first, fallback to synthetic
    print("\nAttempting to load real data from Yahoo Finance...")

    try:
        yahoo_loader = YahooDataLoader()
        # Test with a simple call first
        test_data = yahoo_loader.load('AAPL', '2023-01-01', '2023-01-10')

        if test_data is not None and not test_data.empty:
            print("SUCCESS: Real data loaded from Yahoo Finance")
            backtester = Backtester(config, data_loader=yahoo_loader)
            data_source = "Yahoo Finance"
        else:
            raise ValueError("No data returned")

    except Exception as e:
        print(f"Yahoo Finance failed: {str(e)[:100]}...")
        print("Using synthetic data instead")

        synthetic_loader = CustomDataLoader(synthetic_data_loader)
        backtester = Backtester(config, data_loader=synthetic_loader)
        data_source = "Synthetic"

    print(f"\nData Source: {data_source}")
    print("Running backtest...")

    # Run the backtest
    result = backtester.run(
        strategy=moving_average_strategy,
        universe=['AAPL'],
        strategy_params=strategy_params
    )

    # Results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)

    metrics = result.performance_metrics
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {metrics.get('cagr', 0):.2%}")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"Total Trades: {len(result.trades)}")
    print(f"Runtime: {result.total_runtime_seconds:.2f}s")

    if not result.trades.empty:
        print(f"\nTrade Summary:")
        print(f"  Buy trades: {len(result.trades[result.trades['side'] == 'buy'])}")
        print(f"  Sell trades: {len(result.trades[result.trades['side'] == 'sell'])}")
        print(f"  Avg trade size: ${result.trades['value'].mean():.2f}")
        print(f"  Total commission: ${result.trades['commission'].sum():.2f}")

    if not result.equity_curve.empty:
        initial = result.equity_curve['total_equity'].iloc[0]
        final = result.equity_curve['total_equity'].iloc[-1]
        print(f"\nPortfolio:")
        print(f"  Initial: ${initial:,.2f}")
        print(f"  Final: ${final:,.2f}")
        print(f"  Profit/Loss: ${final - initial:,.2f}")

    print(f"\nSUCCESS: BackT framework working perfectly!")
    print(f"Data source: {data_source}")

    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()