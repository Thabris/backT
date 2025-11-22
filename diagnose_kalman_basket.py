"""
Diagnose Kalman filter basket issue using built-in mock data
"""
from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only

# Use mock data with fixed seed for reproducibility
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000,
    use_mock_data=True,
    mock_seed=42,
    mock_scenario='normal'
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*70)
print("TESTING SPY ALONE")
print("="*70)

result_spy_alone = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

print(f"\nSPY Alone Results:")
print(f"  Total trades: {len(result_spy_alone.trades)}")
print(f"  Buy trades: {len(result_spy_alone.trades[result_spy_alone.trades['side'] == 'buy'])}")
print(f"  Sell trades: {len(result_spy_alone.trades[result_spy_alone.trades['side'] == 'sell'])}")
print(f"  Final equity: ${result_spy_alone.equity_curve['total_equity'].iloc[-1]:,.2f}")

print("\n" + "="*70)
print("TESTING SPY IN BASKET (SPY, QQQ, TLT)")
print("="*70)

# Same config, same seed - mock data will be consistent
result_basket = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'TLT'],
    strategy_params=params
)

spy_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'SPY']
qqq_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'QQQ']
tlt_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'TLT']

print(f"\nBasket Results:")
print(f"  Total trades: {len(result_basket.trades)}")
print(f"  SPY trades: {len(spy_trades_basket)} (was {len(result_spy_alone.trades)} alone)")
print(f"  QQQ trades: {len(qqq_trades_basket)}")
print(f"  TLT trades: {len(tlt_trades_basket)}")
print(f"  Final equity: ${result_basket.equity_curve['total_equity'].iloc[-1]:,.2f}")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

extra_trades = len(spy_trades_basket) - len(result_spy_alone.trades)
print(f"\nSPY has {extra_trades:+d} trades in basket vs. alone")

if extra_trades > 10:
    print("\nPROBLEM: SPY is getting MANY extra trades in the basket!")
    print("   This indicates excessive rebalancing when other symbols enter/exit.")
    print("\n   Root cause: Including HOLD positions in rebalancing causes")
    print("   SPY to trade every time QQQ or TLT changes position.")
    print("\n   Example:")
    print("   - Day 1: SPY enters → 100% weight")
    print("   - Day 5: QQQ enters → SPY rebalanced to 50% (EXTRA TRADE)")
    print("   - Day 10: TLT enters → SPY rebalanced to 33% (EXTRA TRADE)")
    print("   - Day 15: QQQ exits → SPY rebalanced to 50% (EXTRA TRADE)")
    print("\n   Solution needed: Don't rebalance existing positions when")
    print("   other symbols enter/exit. Only rebalance on new signals.")

elif extra_trades < -10:
    print("\n⚠️  PROBLEM: SPY is MISSING trades in the basket!")
    print("   This indicates signals are being blocked or positions")
    print("   are not being included in the portfolio properly.")

else:
    print("\n✓  Trade count is similar - rebalancing issue may be fixed")

# Compare trade dates
if len(result_spy_alone.trades) > 0 and len(spy_trades_basket) > 0:
    spy_alone_dates = set(result_spy_alone.trades.index)
    spy_basket_dates = set(spy_trades_basket.index)

    extra_dates = spy_basket_dates - spy_alone_dates
    missing_dates = spy_alone_dates - spy_basket_dates

    if extra_dates:
        print(f"\n   📊 SPY traded on {len(extra_dates)} extra dates (likely rebalancing)")
        print(f"      First few extra dates: {sorted(list(extra_dates))[:5]}")

    if missing_dates:
        print(f"\n   📊 SPY missing trades on {len(missing_dates)} dates")
        print(f"      First few missing dates: {sorted(list(missing_dates))[:5]}")

# Show first few trades for comparison
print("\n" + "="*70)
print("TRADE COMPARISON (First 5 trades)")
print("="*70)

print("\nSPY ALONE:")
if len(result_spy_alone.trades) > 0:
    display = result_spy_alone.trades.reset_index()[['timestamp', 'side', 'quantity', 'price']].head(5)
    print(display.to_string(index=False))

print("\nSPY IN BASKET:")
if len(spy_trades_basket) > 0:
    display = spy_trades_basket.reset_index()[['timestamp', 'side', 'quantity', 'price']].head(5)
    print(display.to_string(index=False))

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print(f"\nIf SPY has {extra_trades} extra trades, the current approach")
print("of including HOLD positions in rebalancing is causing issues.")
print("\nThe strategy needs to differentiate between:")
print("  1. NEW entry signals → rebalance portfolio")
print("  2. HOLD signals → keep existing weight, don't trigger rebalancing")
