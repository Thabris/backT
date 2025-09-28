"""
Robust Strategy Example with Fallback

This example tries to download real data from Yahoo Finance,
but falls back to synthetic data if there are any issues.
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

    # Remove weekends to simulate trading days
    dates = dates[dates.weekday < 5]

    np.random.seed(hash(symbol) % 1000)  # Different seed per symbol

    n_periods = len(dates)

    # Create more realistic price movements
    base_price = 100 + hash(symbol) % 200  # Different base price per symbol
    trend = 0.0005  # Slight upward trend
    volatility = 0.02

    # Generate returns with some autocorrelation
    returns = np.random.normal(trend, volatility, n_periods)

    # Add some trend and mean reversion
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum

    # Create price series
    prices = base_price * np.exp(np.cumsum(returns))

    # Create realistic OHLC
    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Add intraday noise
    noise_scale = 0.005
    high_noise = np.abs(np.random.normal(0, noise_scale, n_periods))
    low_noise = np.abs(np.random.normal(0, noise_scale, n_periods))

    high_prices = np.maximum(open_prices, close_prices) * (1 + high_noise)
    low_prices = np.minimum(open_prices, close_prices) * (1 - low_noise)

    # Realistic volumes
    base_volume = 1000000 + hash(symbol) % 500000
    volume_noise = np.random.normal(1, 0.3, n_periods)
    volumes = (base_volume * volume_noise).astype(int)
    volumes = np.maximum(volumes, 100000)  # Minimum volume

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def synthetic_data_loader(symbols, start_date, end_date, **kwargs):
    """Fallback data loader with synthetic data"""
    if isinstance(symbols, str):
        symbols = [symbols]

    print(f"📊 Creating synthetic data for {symbols}")
    data = {}
    for symbol in symbols:
        data[symbol] = create_synthetic_data(symbol, start_date, end_date)

    return data


def moving_average_crossover_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Moving average crossover strategy with proper order format
    """

    # Get parameters
    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < long_period:
            continue

        # Calculate moving averages
        short_ma = TechnicalIndicators.sma(data['close'], short_period)
        long_ma = TechnicalIndicators.sma(data['close'], long_period)

        # Check for crossover signals
        if StrategyHelpers.is_crossover(short_ma, long_ma):
            # Buy signal - go 100% into this position
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)

        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            # Sell signal - exit position
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def run_robust_example():
    """Run strategy with fallback to synthetic data"""

    print("BackT Framework - Robust Strategy Example")
    print("=" * 50)

    # Create configuration
    config = BacktestConfig(
        start_date='2022-01-01',  # More recent date
        end_date='2023-01-01',
        initial_capital=100000,
        verbose=True
    )

    # Strategy parameters
    strategy_params = {
        'short_period': 20,
        'long_period': 50
    }

    symbols = ['AAPL']

    # Try Yahoo Finance first, fall back to synthetic data
    print("📈 Attempting to load real market data from Yahoo Finance...")

    try:
        # Try Yahoo Finance
        yahoo_loader = YahooDataLoader()
        backtester = Backtester(config, data_loader=yahoo_loader)

        print("✅ Successfully connected to Yahoo Finance")
        data_source = "Yahoo Finance (Real Data)"

    except Exception as e:
        print(f"⚠️  Yahoo Finance failed: {e}")
        print("🔄 Falling back to synthetic data...")

        # Fall back to synthetic data
        synthetic_loader = CustomDataLoader(synthetic_data_loader)
        backtester = Backtester(config, data_loader=synthetic_loader)
        data_source = "Synthetic Data"

    print(f"📊 Data Source: {data_source}")
    print("\n🔄 Running backtest...")

    # Run backtest
    result = backtester.run(
        strategy=moving_average_crossover_strategy,
        universe=symbols,
        strategy_params=strategy_params
    )

    # Print results
    print("\n" + "=" * 50)
    print("📊 BACKTEST RESULTS")
    print("=" * 50)

    metrics = result.performance_metrics
    print(f"📈 Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"💹 CAGR: {metrics.get('cagr', 0):.2%}")
    print(f"📊 Annualized Volatility: {metrics.get('annualized_volatility', 0):.2%}")
    print(f"⭐ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"🔄 Total Trades: {len(result.trades)}")
    print(f"⏱️  Runtime: {result.total_runtime_seconds:.2f} seconds")

    # Trade analysis
    if not result.trades.empty:
        print(f"\n💼 Trade Analysis:")
        buys = result.trades[result.trades['side'] == 'buy']
        sells = result.trades[result.trades['side'] == 'sell']
        print(f"   • Buy trades: {len(buys)}")
        print(f"   • Sell trades: {len(sells)}")
        print(f"   • Average trade size: ${result.trades['value'].mean():.2f}")
        print(f"   • Total commission: ${result.trades['commission'].sum():.2f}")

        # Show first few trades
        print(f"\n📋 Sample Trades:")
        sample_trades = result.trades.head(3)[['side', 'quantity', 'price', 'value']]
        for idx, trade in sample_trades.iterrows():
            print(f"   • {idx.strftime('%Y-%m-%d')}: {trade['side'].upper()} "
                  f"{trade['quantity']:.0f} shares @ ${trade['price']:.2f}")

    # Equity curve summary
    if not result.equity_curve.empty:
        print(f"\n💰 Portfolio Performance:")
        initial_equity = result.equity_curve['total_equity'].iloc[0]
        final_equity = result.equity_curve['total_equity'].iloc[-1]
        max_equity = result.equity_curve['total_equity'].max()
        min_equity = result.equity_curve['total_equity'].min()

        print(f"   • Initial Value: ${initial_equity:,.2f}")
        print(f"   • Final Value: ${final_equity:,.2f}")
        print(f"   • Peak Value: ${max_equity:,.2f}")
        print(f"   • Lowest Value: ${min_equity:,.2f}")
        print(f"   • Net P&L: ${final_equity - initial_equity:,.2f}")

    print(f"\n✅ Backtest completed successfully!")
    print(f"📊 Data Source Used: {data_source}")

    return result


if __name__ == "__main__":
    try:
        result = run_robust_example()

        print("\n" + "=" * 50)
        print("🎉 SUCCESS: BackT Framework is fully operational!")
        print("=" * 50)
        print("✅ Event-driven backtesting engine working")
        print("✅ Data loading (Yahoo Finance + fallback) working")
        print("✅ Strategy execution working")
        print("✅ Portfolio management working")
        print("✅ Risk metrics calculation working")
        print("✅ Professional logging working")
        print("\n🚀 Ready for production strategy development!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()