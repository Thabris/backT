"""
Detailed diagnostic for Kalman filter differences
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only
import pandas as pd

config = BacktestConfig(
    start_date='2022-01-01',
    end_date='2023-12-31',
    initial_capital=100000
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("DETAILED DIAGNOSTIC: Kalman Filter Long Only")
print("="*80)

print("\n[1] Running SPY alone...")
result_spy = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

print(f"\nSPY ALONE Results:")
print(f"  Trades: {len(result_spy.trades)}")
final_equity_spy = result_spy.equity_curve['total_equity'].iloc[-1]
initial_equity_spy = result_spy.equity_curve['total_equity'].iloc[0]
print(f"  Initial equity: ${initial_equity_spy:,.2f}")
print(f"  Final equity: ${final_equity_spy:,.2f}")
print(f"  Total Return: {((final_equity_spy / initial_equity_spy) - 1)*100:.2f}%")

print("\n[2] Running basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

spy_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'SPY']

print(f"\nBASKET Results:")
print(f"  Total trades: {len(result_basket.trades)}")
print(f"  SPY trades: {len(spy_trades_basket)}")
print(f"  QQQ trades: {len(result_basket.trades[result_basket.trades['symbol'] == 'QQQ'])}")
print(f"  GLD trades: {len(result_basket.trades[result_basket.trades['symbol'] == 'GLD'])}")
print(f"  TLT trades: {len(result_basket.trades[result_basket.trades['symbol'] == 'TLT'])}")
final_equity_basket = result_basket.equity_curve['total_equity'].iloc[-1]
initial_equity_basket = result_basket.equity_curve['total_equity'].iloc[0]
print(f"  Initial equity: ${initial_equity_basket:,.2f}")
print(f"  Final equity: ${final_equity_basket:,.2f}")
print(f"  Total Return: {((final_equity_basket / initial_equity_basket) - 1)*100:.2f}%")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

# Compare trade details
print(f"\nTrade Count: SPY alone={len(result_spy.trades)}, SPY basket={len(spy_trades_basket)}")

if len(result_spy.trades) == len(spy_trades_basket):
    print("✓ Trade counts match")
else:
    print(f"✗ Trade counts differ by {len(spy_trades_basket) - len(result_spy.trades)}")

# Compare dates
if len(result_spy.trades) > 0 and len(spy_trades_basket) > 0:
    dates_alone = set(result_spy.trades.index)
    dates_basket = set(spy_trades_basket.index)

    if dates_alone == dates_basket:
        print("✓ Trade dates match")
    else:
        extra = dates_basket - dates_alone
        missing = dates_alone - dates_basket
        if extra:
            print(f"✗ Basket has {len(extra)} extra dates")
        if missing:
            print(f"✗ Basket missing {len(missing)} dates")

# Show all trades side by side
print(f"\n" + "="*80)
print("ALL TRADES COMPARISON")
print("="*80)

if len(result_spy.trades) > 0:
    print("\nSPY ALONE:")
    spy_alone_display = result_spy.trades.reset_index()[['timestamp', 'side', 'quantity', 'price', 'value']]
    for idx, row in spy_alone_display.iterrows():
        print(f"  {idx+1:2d}. {row['timestamp']:%Y-%m-%d} {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f} = ${row['value']:10,.2f}")

if len(spy_trades_basket) > 0:
    print("\nSPY IN BASKET:")
    spy_basket_display = spy_trades_basket.reset_index()[['timestamp', 'side', 'quantity', 'price', 'value']]
    for idx, row in spy_basket_display.iterrows():
        print(f"  {idx+1:2d}. {row['timestamp']:%Y-%m-%d} {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f} = ${row['value']:10,.2f}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if len(result_spy.trades) == len(spy_trades_basket) and set(result_spy.trades.index) == set(spy_trades_basket.index):
    print("\n✓ SUCCESS: Trades are identical (dates and count)")
    print("  Only difference should be position sizes (100% vs 25%)")
else:
    print("\n✗ ISSUE DETECTED: Trades differ between scenarios")
    print("  This indicates the strategy is not trading independently")

print("\n" + "="*80)
