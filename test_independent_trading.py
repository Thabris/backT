"""
Test if SPY trades identically when it has the same allocated capital
Scenario 1: SPY alone with $100k
Scenario 2: 4 symbols (SPY, QQQ, GLD, TLT) with $400k total = $100k each
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("TEST: Independent Trading with Equal Capital Allocation")
print("="*80)

print("\n[1] SPY alone with $100,000...")
config_100k = BacktestConfig(
    start_date='2012-01-02',
    end_date='2023-09-30',
    initial_capital=100000
)

result_spy_100k = Backtester(config_100k).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

print(f"    Trades: {len(result_spy_100k.trades)}")
print(f"    Final equity: ${result_spy_100k.equity_curve['total_equity'].iloc[-1]:,.2f}")

print("\n[2] Basket with $400,000 ($100k per symbol)...")
config_400k = BacktestConfig(
    start_date='2012-01-02',
    end_date='2023-09-30',
    initial_capital=400000  # 4x capital for 4 symbols
)

result_basket_400k = Backtester(config_400k).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

spy_trades_400k = result_basket_400k.trades[result_basket_400k.trades['symbol'] == 'SPY']

print(f"    Total trades: {len(result_basket_400k.trades)}")
print(f"    SPY trades: {len(spy_trades_400k)}")
print(f"    Final equity: ${result_basket_400k.equity_curve['total_equity'].iloc[-1]:,.2f}")

# Compare SPY trades
print("\n" + "="*80)
print("SPY TRADE COMPARISON")
print("="*80)

print(f"\nTrade count: $100k alone = {len(result_spy_100k.trades)}, $400k basket = {len(spy_trades_400k)}")

# Get trade dates
dates_100k = set(result_spy_100k.trades.index)
dates_400k = set(spy_trades_400k.index)

# Check if dates match
if dates_100k == dates_400k:
    print("SUCCESS: Trade dates MATCH perfectly!")
else:
    only_100k = dates_100k - dates_400k
    only_400k = dates_400k - dates_100k

    if only_100k:
        print(f"\nTrades ONLY in $100k scenario ({len(only_100k)} trades):")
        for date in sorted(only_100k):
            trade = result_spy_100k.trades.loc[date]
            print(f"  {date.date()} - {trade['side']:4s} {trade['quantity']:8.2f} @ ${trade['price']:7.2f}")

    if only_400k:
        print(f"\nTrades ONLY in $400k scenario ({len(only_400k)} trades):")
        for date in sorted(only_400k):
            trade = spy_trades_400k.loc[date]
            print(f"  {date.date()} - {trade['side']:4s} {trade['quantity']:8.2f} @ ${trade['price']:7.2f}")

# Show side-by-side comparison
print("\n" + "="*80)
print("SIDE-BY-SIDE: SPY Trades")
print("="*80)

print("\n{:<15} {:>10} {:>10} | {:<15} {:>10} {:>10} | {:<8}".format(
    "Date ($100k)", "Side", "Quantity", "Date ($400k)", "Side", "Quantity", "Match"
))
print("-" * 80)

trades_100k = result_spy_100k.trades.reset_index()
trades_400k = spy_trades_400k.reset_index()

max_len = max(len(trades_100k), len(trades_400k))

for i in range(max_len):
    left_date = ""
    left_side = ""
    left_qty = ""
    right_date = ""
    right_side = ""
    right_qty = ""
    match = ""

    if i < len(trades_100k):
        left_date = trades_100k.iloc[i]['timestamp'].strftime('%Y-%m-%d')
        left_side = trades_100k.iloc[i]['side']
        left_qty = f"{trades_100k.iloc[i]['quantity']:.2f}"

    if i < len(trades_400k):
        right_date = trades_400k.iloc[i]['timestamp'].strftime('%Y-%m-%d')
        right_side = trades_400k.iloc[i]['side']
        right_qty = f"{trades_400k.iloc[i]['quantity']:.2f}"

    # Check if they match
    if i < len(trades_100k) and i < len(trades_400k):
        if trades_100k.iloc[i]['timestamp'] == trades_400k.iloc[i]['timestamp']:
            match = "OK"
        else:
            match = "DIFF"

    print(f"{left_date:<15} {left_side:>10} {left_qty:>10} | {right_date:<15} {right_side:>10} {right_qty:>10} | {match:<8}")

# Compare per-symbol metrics
print("\n" + "="*80)
print("PER-SYMBOL METRICS COMPARISON")
print("="*80)

if result_spy_100k.per_symbol_metrics and result_basket_400k.per_symbol_metrics:
    spy_100k_metrics = result_spy_100k.per_symbol_metrics.get('SPY', {})
    spy_400k_metrics = result_basket_400k.per_symbol_metrics.get('SPY', {})

    print(f"\n{'Metric':<25} {'$100k Alone':>15} {'$400k Basket':>15} {'Difference':>15}")
    print("-" * 70)

    metrics_to_compare = [
        ('final_equity', 'Final Equity', '${:,.2f}'),
        ('total_return', 'Total Return', '{:.2%}'),
        ('cagr', 'CAGR', '{:.2%}'),
        ('sharpe_ratio', 'Sharpe Ratio', '{:.3f}'),
        ('sortino_ratio', 'Sortino Ratio', '{:.3f}'),
        ('max_drawdown', 'Max Drawdown', '{:.2%}'),
        ('annualized_volatility', 'Volatility', '{:.2%}'),
        ('win_rate', 'Win Rate', '{:.2%}')
    ]

    for key, label, fmt in metrics_to_compare:
        val_100k = spy_100k_metrics.get(key, 0)
        val_400k = spy_400k_metrics.get(key, 0)
        diff = val_400k - val_100k if key != 'final_equity' else 0

        val_100k_str = fmt.format(val_100k)
        val_400k_str = fmt.format(val_400k)
        diff_str = fmt.format(diff) if key != 'final_equity' else 'N/A'

        print(f"{label:<25} {val_100k_str:>15} {val_400k_str:>15} {diff_str:>15}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if dates_100k == dates_400k:
    print("\nSUCCESS: SPY trades identically when given $100k in both scenarios!")
    print("  This confirms symbols trade independently.")

    # Check if metrics are similar
    if result_spy_100k.per_symbol_metrics and result_basket_400k.per_symbol_metrics:
        sharpe_100k = spy_100k_metrics.get('sharpe_ratio', 0)
        sharpe_400k = spy_400k_metrics.get('sharpe_ratio', 0)
        sharpe_diff = abs(sharpe_100k - sharpe_400k)

        if sharpe_diff < 0.01:
            print("  SUCCESS: Sharpe ratios match perfectly!")
        else:
            print(f"  WARNING: Sharpe ratios differ slightly: {sharpe_diff:.3f}")
            print("    (This could be due to minor rounding differences)")
else:
    print("\nISSUE: Trades still differ even with equal capital allocation!")
    print("  This indicates a deeper problem in the strategy or execution engine.")

print("\n" + "="*80)
