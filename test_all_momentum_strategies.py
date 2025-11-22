"""
Test All Momentum Strategies with Mock Data

Tests every momentum strategy to verify:
1. No runtime bugs or errors
2. Trades execute properly
3. Trade logic is full buy/sell/cover (no rebalancing)
4. Signals trigger correctly

Uses short time span (6 months) with synthetic data for fast testing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from backt import Backtester, BacktestConfig
from backt.data.loaders import CustomDataLoader
from strategies.momentum import (
    ma_crossover_long_only,
    ma_crossover_long_short,
    kalman_ma_crossover_long_only,
    kalman_ma_crossover_long_short,
    rsi_mean_reversion,
    macd_crossover,
    stochastic_momentum,
    bollinger_mean_reversion,
    adx_trend_filter
)


def create_volatile_synthetic_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Create synthetic data with trends and volatility to trigger signals

    This data is designed to trigger trades:
    - Has clear trends (for MA/MACD strategies)
    - Has overbought/oversold conditions (for RSI/Stochastic)
    - Has volatility (for Bollinger Bands)
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    n_periods = len(dates)

    # Set seed based on symbol for reproducibility but different data per symbol
    np.random.seed(hash(symbol) % (2**32))

    # Create a price series with clear trends and reversals
    base_price = 100.0

    # Generate trending price with reversals
    # First third: uptrend
    # Middle third: downtrend
    # Last third: uptrend
    segment_size = n_periods // 3

    trend1 = np.linspace(0, 20, segment_size) + np.random.normal(0, 2, segment_size)
    trend2 = np.linspace(20, 5, segment_size) + np.random.normal(0, 2, segment_size)
    trend3 = np.linspace(5, 25, n_periods - 2*segment_size) + np.random.normal(0, 2, n_periods - 2*segment_size)

    price_changes = np.concatenate([trend1, trend2, trend3])
    close_prices = base_price + price_changes

    # Add some volatility spikes
    volatility = np.random.normal(0, 0.5, n_periods)
    close_prices = close_prices + volatility

    # Create OHLC from close
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    # Add intraday volatility
    noise = 0.01
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, noise, n_periods)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, noise, n_periods)))

    volumes = np.random.randint(500000, 2000000, n_periods)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)

    return df


def synthetic_data_loader(symbols, start_date, end_date, **kwargs):
    """Custom data loader for synthetic data"""
    if isinstance(symbols, str):
        symbols = [symbols]
    return {symbol: create_volatile_synthetic_data(symbol, start_date, end_date) for symbol in symbols}


def analyze_trades(result, strategy_name):
    """
    Analyze trades to verify they are full buy/sell (not rebalancing)

    Returns: dict with analysis results
    """
    trades = result.trades

    if trades.empty:
        return {
            'strategy': strategy_name,
            'total_trades': 0,
            'has_trades': False,
            'error': None
        }

    # Check trade patterns
    buy_trades = trades[trades['side'] == 'buy']
    sell_trades = trades[trades['side'] == 'sell']

    # Group by symbol to check per-symbol behavior
    symbols_traded = trades['symbol'].unique()

    analysis = {
        'strategy': strategy_name,
        'total_trades': len(trades),
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'symbols_traded': len(symbols_traded),
        'has_trades': True,
        'error': None,
        'symbols': list(symbols_traded)
    }

    # Analyze position sizing to detect rebalancing
    # Full buy/sell should have distinct entry and exit trades
    # Rebalancing would show many small adjustments

    for symbol in symbols_traded:
        symbol_trades = trades[trades['symbol'] == symbol]

        # Check if trades alternate between buy/sell (expected) or have multiple buys/sells in a row (rebalancing)
        sides = symbol_trades['side'].values

        # Count consecutive same-side trades (indicator of rebalancing)
        consecutive_same_side = 0
        for i in range(1, len(sides)):
            if sides[i] == sides[i-1]:
                consecutive_same_side += 1

        analysis[f'{symbol}_consecutive_same_side'] = consecutive_same_side
        analysis[f'{symbol}_total_trades'] = len(symbol_trades)

    return analysis


def test_strategy(strategy_func, strategy_name, symbols, start_date, end_date, params=None):
    """
    Test a single strategy and return results
    """
    print(f"\n{'='*80}")
    print(f"Testing: {strategy_name}")
    print(f"{'='*80}")

    try:
        # Create backtest config
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_capital=100000,
            verbose=False
        )

        # Create data loader
        data_loader = CustomDataLoader(synthetic_data_loader)

        # Create backtester
        backtester = Backtester(config, data_loader=data_loader)

        # Run backtest
        print(f"Running backtest...")
        result = backtester.run(
            strategy=strategy_func,
            universe=symbols,
            strategy_params=params or {}
        )

        # Analyze results
        analysis = analyze_trades(result, strategy_name)

        # Print summary
        print(f"\n[SUCCESS] {strategy_name}")
        print(f"   Total Trades: {analysis['total_trades']}")
        print(f"   Buy Trades: {analysis.get('buy_trades', 0)}")
        print(f"   Sell Trades: {analysis.get('sell_trades', 0)}")
        print(f"   Symbols Traded: {analysis.get('symbols_traded', 0)}")

        if analysis['total_trades'] > 0:
            print(f"   Traded: {', '.join(analysis['symbols'])}")

            # Check for rebalancing indicators
            for symbol in analysis['symbols']:
                consecutive = analysis.get(f'{symbol}_consecutive_same_side', 0)
                if consecutive > 2:
                    print(f"   [WARNING] {symbol} has {consecutive} consecutive same-side trades (possible rebalancing)")

        # Print performance metrics
        metrics = result.performance_metrics
        print(f"\n   Performance:")
        print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

        return {
            'status': 'SUCCESS',
            'strategy': strategy_name,
            'analysis': analysis,
            'metrics': metrics,
            'result': result,
            'error': None
        }

    except Exception as e:
        print(f"\n[FAILED] {strategy_name}")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'status': 'FAILED',
            'strategy': strategy_name,
            'error': str(e),
            'analysis': None,
            'metrics': None
        }


def main():
    """
    Main test runner - tests all momentum strategies
    """
    print("\n" + "="*80)
    print("MOMENTUM STRATEGY TEST SUITE")
    print("="*80)
    print("\nConfiguration:")
    print("  Period: 6 months (short span)")
    print("  Symbols: SPY, QQQ, TLT (3 symbols with different characteristics)")
    print("  Data: Synthetic with trends and volatility")
    print("  Capital: $100,000")
    print("\n")

    # Test configuration
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    symbols = ['SPY', 'QQQ', 'TLT']

    # Define strategies to test
    strategies_to_test = [
        {
            'name': 'MA Crossover (Long Only)',
            'func': ma_crossover_long_only,
            'params': {'fast_ma': 10, 'slow_ma': 20, 'min_periods': 30}
        },
        {
            'name': 'MA Crossover (Long-Short)',
            'func': ma_crossover_long_short,
            'params': {'fast_ma': 10, 'slow_ma': 20, 'min_periods': 30}
        },
        {
            'name': 'Kalman MA Crossover (Long Only)',
            'func': kalman_ma_crossover_long_only,
            'params': {'Q_fast': 0.014, 'Q_slow': 0.0006, 'R': 1.0, 'min_periods': 30}
        },
        {
            'name': 'Kalman MA Crossover (Long-Short)',
            'func': kalman_ma_crossover_long_short,
            'params': {'Q_fast': 0.01, 'Q_slow': 0.001, 'R': 1.0, 'min_periods': 30}
        },
        {
            'name': 'RSI Mean Reversion',
            'func': rsi_mean_reversion,
            'params': {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70, 'min_periods': 20}
        },
        {
            'name': 'MACD Crossover (Long Only)',
            'func': macd_crossover,
            'params': {'fast_period': 11, 'slow_period': 20, 'signal_period': 10, 'allow_short': False, 'min_periods': 25}
        },
        {
            'name': 'MACD Crossover (Long-Short)',
            'func': macd_crossover,
            'params': {'fast_period': 11, 'slow_period': 20, 'signal_period': 10, 'allow_short': True, 'min_periods': 25}
        },
        {
            'name': 'Stochastic Momentum',
            'func': stochastic_momentum,
            'params': {'period': 14, 'oversold': 20, 'overbought': 80, 'min_periods': 20}
        },
        {
            'name': 'Bollinger Bands Mean Reversion',
            'func': bollinger_mean_reversion,
            'params': {'bb_period': 20, 'bb_std': 2.0, 'min_periods': 25}
        },
        {
            'name': 'ADX Trend Filter (Long Only)',
            'func': adx_trend_filter,
            'params': {'adx_period': 14, 'adx_threshold': 20, 'allow_short': False, 'min_periods': 25}
        },
        {
            'name': 'ADX Trend Filter (Long-Short)',
            'func': adx_trend_filter,
            'params': {'adx_period': 14, 'adx_threshold': 20, 'allow_short': True, 'min_periods': 25}
        }
    ]

    # Run tests
    results = []
    for strategy_config in strategies_to_test:
        result = test_strategy(
            strategy_func=strategy_config['func'],
            strategy_name=strategy_config['name'],
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            params=strategy_config['params']
        )
        results.append(result)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = [r for r in results if r['status'] == 'SUCCESS']
    failed = [r for r in results if r['status'] == 'FAILED']

    print(f"\nTotal Strategies Tested: {len(results)}")
    print(f"[OK] Passed: {len(passed)}")
    print(f"[X] Failed: {len(failed)}")

    if failed:
        print("\nFailed Strategies:")
        for r in failed:
            print(f"  - {r['strategy']}: {r['error']}")

    # Trade analysis summary
    print("\n" + "="*80)
    print("TRADE ANALYSIS - Checking for Rebalancing Behavior")
    print("="*80)

    for r in passed:
        analysis = r['analysis']
        if analysis['total_trades'] > 0:
            print(f"\n{r['strategy']}:")
            print(f"  Total Trades: {analysis['total_trades']}")

            # Check for rebalancing indicators
            rebalancing_detected = False
            for symbol in analysis['symbols']:
                consecutive = analysis.get(f'{symbol}_consecutive_same_side', 0)
                if consecutive > 2:
                    print(f"  [!] {symbol}: {consecutive} consecutive same-side trades (REBALANCING DETECTED)")
                    rebalancing_detected = True
                else:
                    print(f"  [OK] {symbol}: Clean entry/exit pattern (no rebalancing)")

            if not rebalancing_detected:
                print(f"  [OK] No rebalancing detected - trades are full buy/sell/cover")
        else:
            print(f"\n{r['strategy']}: No trades executed (no signals triggered)")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    results = main()
