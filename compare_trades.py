"""
Compare SPY trades between solo and basket scenarios to identify differences
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only
import pandas as pd

config = BacktestConfig(
    start_date='2012-01-02',
    end_date='2023-09-30',
    initial_capital=100000
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("TRADE COMPARISON: SPY Alone vs SPY in Basket")
print("="*80)

print("\n[1] Running SPY alone...")
result_spy = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

print(f"\n[2] Running basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

spy_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'SPY']

print(f"\nTrade counts: SPY alone = {len(result_spy.trades)}, SPY in basket = {len(spy_trades_basket)}")

# Get trade dates (timestamp is the index)
dates_alone = set(result_spy.trades.index)
dates_basket = set(spy_trades_basket.index)

# Find differences
only_in_alone = dates_alone - dates_basket
only_in_basket = dates_basket - dates_alone

print("\n" + "="*80)
print("TRADE DATE DIFFERENCES")
print("="*80)

if only_in_alone:
    print(f"\nTrades ONLY in 'SPY alone' scenario ({len(only_in_alone)} trades):")
    for date in sorted(only_in_alone):
        trade = result_spy.trades.loc[date]
        print(f"  {pd.Timestamp(date).date()} - {trade['side']:4s} {trade['quantity']:8.2f} @ ${trade['price']:7.2f}")

if only_in_basket:
    print(f"\nTrades ONLY in 'SPY in basket' scenario ({len(only_in_basket)} trades):")
    for date in sorted(only_in_basket):
        trade = spy_trades_basket.loc[date]
        print(f"  {pd.Timestamp(date).date()} - {trade['side']:4s} {trade['quantity']:8.2f} @ ${trade['price']:7.2f}")

# Show all trades side by side
print("\n" + "="*80)
print("ALL TRADES - SIDE BY SIDE COMPARISON")
print("="*80)

print("\n{:<15} {:<12} {:<8} {:>10} | {:<15} {:<12} {:<8} {:>10}".format(
    "SPY ALONE", "Date", "Side", "Qty", "SPY BASKET", "Date", "Side", "Qty"
))
print("-" * 80)

alone_df = result_spy.trades.reset_index()
basket_df = spy_trades_basket.reset_index()

alone_list = alone_df[['timestamp', 'side', 'quantity']].values.tolist()
basket_list = basket_df[['timestamp', 'side', 'quantity']].values.tolist()

max_len = max(len(alone_list), len(basket_list))

for i in range(max_len):
    left = ""
    right = ""

    if i < len(alone_list):
        t, side, qty = alone_list[i]
        left = f"{i+1:2d}. {pd.Timestamp(t).date()} {side:4s} {qty:9.2f}"
    else:
        left = " " * 40

    if i < len(basket_list):
        t, side, qty = basket_list[i]
        right = f"{i+1:2d}. {pd.Timestamp(t).date()} {side:4s} {qty:9.2f}"
    else:
        right = " " * 40

    # Highlight if dates differ
    marker = ""
    if i < len(alone_list) and i < len(basket_list):
        if alone_list[i][0] != basket_list[i][0]:
            marker = " <-- DIFFERENT"

    print(f"{left:<40} | {right:<40}{marker}")

print("\n" + "="*80)
