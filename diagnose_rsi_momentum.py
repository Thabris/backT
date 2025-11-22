"""Diagnose RSI Momentum position sizing and trade matching"""
import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backt import Backtester, BacktestConfig
from strategies.momentum import rsi_momentum
import pandas as pd

# Configure backtest
config = BacktestConfig(
    initial_capital=100000,
    start_date='2012-02-01',
    end_date='2012-03-31'
)

# Run backtest
backtester = Backtester(config)
result = backtester.run(
    strategy=rsi_momentum,
    universe=['SPY', 'QQQ', 'TLT', 'GLD'],
    strategy_params={
        'rsi_period': 14,
        'buy_threshold': 60,
        'sell_threshold': 40,
        'buy_tp': 50,
        'sell_tp': 50
    }
)

# Get trades
trades = result.trades

print(f"\nTotal trades: {len(trades)}")
print("\n" + "="*100)
print("ALL TRADES")
print("="*100)

# Show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.max_colwidth', 60)

# Show key columns
cols_to_display = ['symbol', 'side', 'quantity', 'price', 'value', 'meta_signal', 'meta_reason', 'meta_trade_type']
print(trades[cols_to_display].to_string())

# Analyze each symbol
print("\n" + "="*100)
print("POSITION ANALYSIS BY SYMBOL")
print("="*100)

for symbol in ['SPY', 'QQQ', 'TLT', 'GLD']:
    symbol_trades = trades[trades['symbol'] == symbol].copy()

    if len(symbol_trades) == 0:
        print(f"\n{symbol}: No trades")
        continue

    print(f"\n{symbol} Trades ({len(symbol_trades)} total):")
    print(symbol_trades[cols_to_display].to_string())

    # Check if buy and sell volumes match
    buys = symbol_trades[symbol_trades['side'] == 'buy']['quantity'].sum()
    sells = symbol_trades[symbol_trades['side'] == 'sell']['quantity'].sum()

    print(f"\nTotal {symbol} bought: {buys:.6f}")
    print(f"Total {symbol} sold: {sells:.6f}")
    print(f"Difference: {buys - sells:.6f}")
    print(f"Match: {'✓ YES' if abs(buys - sells) < 0.01 else '✗ NO - PROBLEM!'}")

print("\n" + "="*100)
