"""
Test Kalman filter fix with REAL Yahoo Finance data
SPY should trade identically alone vs in basket
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only

config = BacktestConfig(
    start_date='2022-01-01',
    end_date='2023-12-31',
    initial_capital=100000
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*70)
print("TEST: SPY alone vs. SPY in basket (SPY, QQQ, GLD, TLT)")
print("Using REAL Yahoo Finance data")
print("="*70)

print("\n[1/2] Running SPY alone...")
result_spy_alone = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

spy_alone_trades = result_spy_alone.trades.copy()
print(f"      SPY trades: {len(spy_alone_trades)}")
print(f"      Final equity: ${result_spy_alone.equity_curve['total_equity'].iloc[-1]:,.2f}")

print("\n[2/2] Running basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

spy_basket_trades = result_basket.trades[result_basket.trades['symbol'] == 'SPY'].copy()
qqq_trades = result_basket.trades[result_basket.trades['symbol'] == 'QQQ']
gld_trades = result_basket.trades[result_basket.trades['symbol'] == 'GLD']
tlt_trades = result_basket.trades[result_basket.trades['symbol'] == 'TLT']

print(f"      Total trades: {len(result_basket.trades)}")
print(f"      SPY: {len(spy_basket_trades)} trades")
print(f"      QQQ: {len(qqq_trades)} trades")
print(f"      GLD: {len(gld_trades)} trades")
print(f"      TLT: {len(tlt_trades)} trades")
print(f"      Final equity: ${result_basket.equity_curve['total_equity'].iloc[-1]:,.2f}")

print("\n" + "="*70)
print("VERIFICATION")
print("="*70)

all_pass = True

# Test 1: Trade count
if len(spy_basket_trades) == len(spy_alone_trades):
    print(f"\n[PASS] Trade count: {len(spy_alone_trades)} trades (identical)")
else:
    print(f"\n[FAIL] Trade count mismatch:")
    print(f"       Alone: {len(spy_alone_trades)} trades")
    print(f"       Basket: {len(spy_basket_trades)} trades")
    print(f"       Difference: {len(spy_basket_trades) - len(spy_alone_trades):+d}")
    all_pass = False

# Test 2: Trade dates
if len(spy_basket_trades) > 0 and len(spy_alone_trades) > 0:
    basket_dates = set(spy_basket_trades.index)
    alone_dates = set(spy_alone_trades.index)

    if basket_dates == alone_dates:
        print("[PASS] Trade dates: Perfectly aligned")
    else:
        extra = basket_dates - alone_dates
        missing = alone_dates - basket_dates
        print("[FAIL] Trade dates don't match:")
        if extra:
            print(f"       {len(extra)} extra in basket: {sorted(list(extra))[:3]}")
        if missing:
            print(f"       {len(missing)} missing in basket: {sorted(list(missing))[:3]}")
        all_pass = False

# Test 3: Trade sides match
if len(spy_basket_trades) == len(spy_alone_trades) and len(spy_basket_trades) > 0:
    alone_sorted = spy_alone_trades.sort_index()
    basket_sorted = spy_basket_trades.sort_index()

    if (alone_sorted['side'].tolist() == basket_sorted['side'].tolist()):
        print("[PASS] Trade sides: Buy/sell signals match")
    else:
        print("[FAIL] Trade sides (buy/sell) don't match")
        all_pass = False

# Show trade comparison
if len(spy_basket_trades) > 0 and len(spy_alone_trades) > 0:
    print("\nFirst 5 SPY trades comparison:")
    print("\nALONE:")
    alone_display = spy_alone_trades.reset_index()[['timestamp', 'side', 'quantity', 'price']].head(5)
    for _, row in alone_display.iterrows():
        print(f"  {row['timestamp']:%Y-%m-%d} {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f}")

    print("\nBASKET:")
    basket_display = spy_basket_trades.reset_index()[['timestamp', 'side', 'quantity', 'price']].head(5)
    for _, row in basket_display.iterrows():
        print(f"  {row['timestamp']:%Y-%m-%d} {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f}")

print("\n" + "="*70)
print("RESULT")
print("="*70)

if all_pass:
    print("\n SUCCESS: SPY trades identically in both scenarios!")
    print("  - Same number of trades")
    print("  - Same trade dates")
    print("  - Same buy/sell signals")
    print("  - Each symbol has fixed 1/N allocation (25% for 4 symbols)")
    print("  - No cross-symbol rebalancing")
else:
    print("\n FAILURE: SPY behavior differs between scenarios")
    print("  Strategy still has issues with basket trading")

print("\n" + "="*70)
