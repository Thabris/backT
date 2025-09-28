"""
Test BackT Framework - No External Dependencies

This example tests the framework with synthetic data,
demonstrating that the core engine works without external data sources.
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
    returns = np.random.normal(0.001, 0.02, n_periods)  # Daily returns with drift

    # Create price series
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLC from prices with some noise
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods)))

    # Ensure high >= max(open, close) and low <= min(open, close)
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


def simple_ma_strategy(market_data, current_time, positions, context, params):
    """Simple moving average crossover strategy"""
    short_period = params.get('short_period', 10)
    long_period = params.get('long_period', 20)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < long_period:
            continue

        # Calculate moving averages
        short_ma = TechnicalIndicators.sma(data['close'], short_period)
        long_ma = TechnicalIndicators.sma(data['close'], long_period)

        # Check for crossover signals
        if StrategyHelpers.is_crossover(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)
        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def test_framework():
    """Test the BackT framework with synthetic data"""
    print("🚀 Testing BackT Framework with Synthetic Data")
    print("=" * 50)

    # Create configuration
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2020-12-31',
        initial_capital=100000,
        verbose=True
    )

    # Create custom data loader with synthetic data
    data_loader = CustomDataLoader(synthetic_data_loader)

    # Create backtester with custom data loader
    backtester = Backtester(config, data_loader=data_loader)

    # Strategy parameters
    strategy_params = {
        'short_period': 10,
        'long_period': 20
    }

    # Run backtest
    print("\n📊 Running backtest with synthetic AAPL data...")
    result = backtester.run(
        strategy=simple_ma_strategy,
        universe=['AAPL'],
        strategy_params=strategy_params
    )

    # Display results
    print("\n" + "=" * 50)
    print("📈 BACKTEST RESULTS")
    print("=" * 50)

    metrics = result.performance_metrics
    print(f"💰 Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"📊 CAGR: {metrics.get('cagr', 0):.2%}")
    print(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"🔄 Total Trades: {len(result.trades)}")
    print(f"⏱️  Runtime: {result.total_runtime_seconds:.2f} seconds")

    # Show some trade details
    if not result.trades.empty:
        print(f"\n🔄 Trade Summary:")
        print(f"   • Buy trades: {len(result.trades[result.trades['side'] == 'buy'])}")
        print(f"   • Sell trades: {len(result.trades[result.trades['side'] == 'sell'])}")
        print(f"   • Average trade size: ${result.trades['value'].mean():.2f}")
        print(f"   • Total commission: ${result.trades['commission'].sum():.2f}")

    # Show equity curve summary
    if not result.equity_curve.empty:
        print(f"\n📊 Portfolio Summary:")
        initial_equity = result.equity_curve['total_equity'].iloc[0]
        final_equity = result.equity_curve['total_equity'].iloc[-1]
        print(f"   • Initial Portfolio Value: ${initial_equity:,.2f}")
        print(f"   • Final Portfolio Value: ${final_equity:,.2f}")
        print(f"   • Net Profit/Loss: ${final_equity - initial_equity:,.2f}")

    print("\n✅ Framework test completed successfully!")
    print("\n💡 The BackT framework is working correctly!")
    print("   You can now install yfinance to use real market data:")
    print("   pip install yfinance")

    return result


if __name__ == "__main__":
    try:
        result = test_framework()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()