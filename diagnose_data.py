"""
Diagnose if SPY market data is identical between solo and basket scenarios
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only
import pandas as pd

# Monkey patch the strategy to capture market data snapshots
captured_data = {'alone': [], 'basket': []}

def capture_wrapper(mode):
    """Wrap the strategy to capture market data"""
    def wrapped_strategy(market_data, current_time, positions, context, params):
        # Capture SPY data if present
        if 'SPY' in market_data:
            spy_data = market_data['SPY']
            captured_data[mode].append({
                'time': current_time,
                'n_rows': len(spy_data),
                'first_date': spy_data.index[0] if len(spy_data) > 0 else None,
                'last_date': spy_data.index[-1] if len(spy_data) > 0 else None,
                'close_prices': spy_data['close'].tolist() if len(spy_data) > 0 else []
            })

        # Call original strategy
        return kalman_ma_crossover_long_only(market_data, current_time, positions, context, params)

    return wrapped_strategy

config = BacktestConfig(
    start_date='2015-10-01',  # Focus on period where trades diverge
    end_date='2016-02-01',
    initial_capital=100000,
    verbose=False
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("DATA DIAGNOSIS: Compare SPY market data between scenarios")
print("="*80)

print("\n[1] Running SPY alone...")
result_spy = Backtester(config).run(
    strategy=capture_wrapper('alone'),
    universe=['SPY'],
    strategy_params=params
)

print(f"Captured {len(captured_data['alone'])} snapshots for SPY alone")

print("\n[2] Running basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=capture_wrapper('basket'),
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

print(f"Captured {len(captured_data['basket'])} snapshots for basket")

# Compare data snapshots
print("\n" + "="*80)
print("DATA COMPARISON")
print("="*80)

alone_snapshots = captured_data['alone']
basket_snapshots = captured_data['basket']

print(f"\nTotal snapshots: Alone={len(alone_snapshots)}, Basket={len(basket_snapshots)}")

# Compare first few snapshots
for i in range(min(10, len(alone_snapshots), len(basket_snapshots))):
    alone = alone_snapshots[i]
    basket = basket_snapshots[i]

    time_match = alone['time'] == basket['time']
    n_rows_match = alone['n_rows'] == basket['n_rows']
    dates_match = (alone['first_date'] == basket['first_date'] and
                   alone['last_date'] == basket['last_date'])

    # Compare close prices
    prices_match = alone['close_prices'] == basket['close_prices']

    marker = "OK" if (time_match and n_rows_match and dates_match and prices_match) else "DIFF"

    print(f"\n[{marker}] Snapshot {i+1} - {alone['time'].date()}")
    if not time_match:
        print(f"   Time mismatch: {alone['time']} vs {basket['time']}")
    if not n_rows_match:
        print(f"   Row count: {alone['n_rows']} vs {basket['n_rows']}")
    if not dates_match:
        print(f"   Date range: {alone['first_date']} to {alone['last_date']}")
        print(f"           vs: {basket['first_date']} to {basket['last_date']}")
    if not prices_match and n_rows_match:
        # Find first difference in prices
        for j, (p1, p2) in enumerate(zip(alone['close_prices'], basket['close_prices'])):
            if abs(p1 - p2) > 0.01:
                print(f"   Price diff at index {j}: {p1} vs {p2}")
                break

# Focus on dates around the divergence (2015-10-23)
print("\n" + "="*80)
print("FOCUS ON DIVERGENCE DATE: 2015-10-23")
print("="*80)

target_date = pd.Timestamp('2015-10-23')

for i, snap in enumerate(alone_snapshots):
    if snap['time'] == target_date:
        print(f"\nSPY ALONE on {target_date.date()}:")
        print(f"  Data rows: {snap['n_rows']}")
        print(f"  Date range: {snap['first_date'].date()} to {snap['last_date'].date()}")
        print(f"  Last 5 close prices: {snap['close_prices'][-5:]}")
        break

for i, snap in enumerate(basket_snapshots):
    if snap['time'] == target_date:
        print(f"\nSPY IN BASKET on {target_date.date()}:")
        print(f"  Data rows: {snap['n_rows']}")
        print(f"  Date range: {snap['first_date'].date()} to {snap['last_date'].date()}")
        print(f"  Last 5 close prices: {snap['close_prices'][-5:]}")
        break

print("\n" + "="*80)
