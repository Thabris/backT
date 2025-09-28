"""
Advanced Multi-Indicator Strategy Example

This example demonstrates a more sophisticated strategy that combines
multiple technical indicators for signal generation.
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


def advanced_multi_indicator_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Advanced strategy combining multiple indicators:
    - Moving average trend
    - RSI for momentum
    - Bollinger Bands for volatility
    """

    # Get parameters
    ma_short = params.get('ma_short', 20)
    ma_long = params.get('ma_long', 50)
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    bb_period = params.get('bb_period', 20)
    position_size = params.get('position_size', 0.2)  # 20% per position

    orders = {}

    for symbol, data in market_data.items():
        if len(data) < max(ma_long, bb_period, rsi_period):
            continue

        # Calculate indicators
        sma_short = TechnicalIndicators.sma(data['close'], ma_short)
        sma_long = TechnicalIndicators.sma(data['close'], ma_long)
        rsi = TechnicalIndicators.rsi(data['close'], rsi_period)
        bb = TechnicalIndicators.bollinger_bands(data['close'], bb_period)

        # Get current values
        current_price = data['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_bb_upper = bb['upper'].iloc[-1]
        current_bb_lower = bb['lower'].iloc[-1]

        # Get current position
        current_position = positions.get(symbol)
        current_qty = current_position.qty if current_position else 0

        # Signal generation
        trend_bullish = sma_short.iloc[-1] > sma_long.iloc[-1]
        trend_bearish = sma_short.iloc[-1] < sma_long.iloc[-1]

        momentum_oversold = current_rsi < rsi_oversold
        momentum_overbought = current_rsi > rsi_overbought

        price_near_lower_bb = current_price <= current_bb_lower * 1.02  # Within 2%
        price_near_upper_bb = current_price >= current_bb_upper * 0.98  # Within 2%

        # Buy conditions: bullish trend + oversold RSI + near lower BB
        if (trend_bullish and momentum_oversold and price_near_lower_bb and current_qty <= 0):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=position_size)

        # Sell conditions: bearish trend + overbought RSI + near upper BB
        elif (trend_bearish and momentum_overbought and price_near_upper_bb and current_qty > 0):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

        # Risk management: exit if RSI extreme opposite
        elif current_qty > 0 and current_rsi > 80:  # Exit long on extreme overbought
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

        elif current_qty < 0 and current_rsi < 20:  # Exit short on extreme oversold
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def run_advanced_strategy():
    """Run the advanced multi-indicator strategy"""

    # Create configuration with more sophisticated settings
    config = BacktestConfig(
        start_date='2020-01-01',
        end_date='2023-01-01',
        initial_capital=100000,
        allow_short=False,  # Long-only for this example
        max_position_size=0.25,  # Max 25% in any single position
        verbose=True
    )

    # Create backtester
    backtester = Backtester(config)

    # Strategy parameters
    strategy_params = {
        'ma_short': 20,
        'ma_long': 50,
        'rsi_period': 14,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'bb_period': 20,
        'position_size': 0.2
    }

    # Test on a diversified set of stocks
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    print("Running advanced multi-indicator strategy...")
    result = backtester.run(
        strategy=advanced_multi_indicator_strategy,
        universe=universe,
        strategy_params=strategy_params
    )

    # Print detailed results
    print("\n=== Advanced Strategy Results ===")
    print(f"Total Return: {result.performance_metrics.get('total_return', 0):.2%}")
    print(f"CAGR: {result.performance_metrics.get('cagr', 0):.2%}")
    print(f"Annualized Volatility: {result.performance_metrics.get('annualized_volatility', 0):.2%}")
    print(f"Sharpe Ratio: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Sortino Ratio: {result.performance_metrics.get('sortino_ratio', 0):.3f}")
    print(f"Max Drawdown: {result.performance_metrics.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio: {result.performance_metrics.get('calmar_ratio', 0):.3f}")
    print(f"Total Trades: {len(result.trades)}")

    # Trade analysis
    if not result.trades.empty:
        print(f"\nTrade Analysis:")
        print(f"Average Trade Size: ${result.trades['value'].mean():.2f}")
        print(f"Total Commission: ${result.trades['commission'].sum():.2f}")

        # Trades by symbol
        trades_by_symbol = result.trades.groupby('symbol').size()
        print(f"\nTrades by Symbol:")
        for symbol, count in trades_by_symbol.items():
            print(f"  {symbol}: {count} trades")

    return result


if __name__ == "__main__":
    result = run_advanced_strategy()

    # Generate comprehensive report
    try:
        from backt.reporting import ReportGenerator, PlotGenerator

        # Generate reports
        report_gen = ReportGenerator("./results")
        files = report_gen.generate_full_report(result, "advanced_strategy")
        print(f"\nGenerated report files: {list(files.keys())}")

        # Create dashboard
        plotter = PlotGenerator()
        plotter.create_performance_dashboard(result, "./results/advanced_strategy_dashboard.png")

    except ImportError as e:
        print(f"Could not generate reports: {e}")