"""
Diagnose Kalman filter signal generation differences
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from backt.signal.kalman import KalmanFilter1D
import pandas as pd
from typing import Dict, Any

# Capture signal details
signal_log = {'alone': {}, 'basket': {}}

def kalman_with_logging(mode):
    """Kalman strategy with detailed logging"""
    def strategy(
        market_data: Dict[str, pd.DataFrame],
        current_time: pd.Timestamp,
        positions: Dict[str, Any],
        context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Dict[str, Dict]:
        Q_fast = params.get('Q_fast', 0.014)
        Q_slow = params.get('Q_slow', 0.0006)
        R = params.get('R', 1.0)
        min_periods = params.get('min_periods', 60)

        orders = {}
        signals = {}
        new_entries = []

        for symbol, data in market_data.items():
            if symbol != 'SPY':  # Only log SPY
                continue

            if len(data) < min_periods:
                continue

            # Compute Kalman filters
            kf_fast = KalmanFilter1D(Q=Q_fast, R=R)
            kf_slow = KalmanFilter1D(Q=Q_slow, R=R)

            fast_values = [kf_fast.update(p) for p in data['close']]
            slow_values = [kf_slow.update(p) for p in data['close']]

            if len(fast_values) < 2 or len(slow_values) < 2:
                continue

            current_fast = fast_values[-1]
            current_slow = slow_values[-1]
            prev_fast = fast_values[-2]
            prev_slow = slow_values[-2]

            # Detect crossovers
            golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

            # Check position
            has_position = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0
            current_qty = positions[symbol].qty if has_position else 0.0

            # Log detailed state
            date_str = current_time.strftime('%Y-%m-%d')
            signal_log[mode][date_str] = {
                'prev_fast': prev_fast,
                'prev_slow': prev_slow,
                'current_fast': current_fast,
                'current_slow': current_slow,
                'golden_cross': golden_cross,
                'death_cross': death_cross,
                'has_position': has_position,
                'current_qty': current_qty,
                'data_len': len(data),
                'last_price': data['close'].iloc[-1]
            }

            # Generate signals
            if golden_cross and not has_position:
                signals[symbol] = 'BUY'
                new_entries.append(symbol)
            elif death_cross and has_position:
                signals[symbol] = 'SELL'
            elif has_position:
                signals[symbol] = 'HOLD_LONG'
            else:
                signals[symbol] = 'NEUTRAL'

        # Position sizing
        total_symbols = len(market_data)
        weight_per_symbol = 1.0 / total_symbols

        # Create orders
        for symbol in new_entries:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_symbol
            }

        # Close positions
        for symbol, signal in signals.items():
            if signal == 'SELL':
                if symbol in positions and hasattr(positions[symbol], 'qty'):
                    if positions[symbol].qty != 0:
                        orders[symbol] = {'action': 'close'}

        context['signals'] = signals
        return orders

    return strategy

config = BacktestConfig(
    start_date='2012-01-02',  # Start from beginning to capture full state
    end_date='2016-02-15',     # Include both divergent trades
    initial_capital=100000,
    verbose=False
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("SIGNAL DIAGNOSIS: Kalman Filter Signal Generation")
print("="*80)

print("\n[1] Running SPY alone...")
result_spy = Backtester(config).run(
    strategy=kalman_with_logging('alone'),
    universe=['SPY'],
    strategy_params=params
)

print(f"Trades: {len(result_spy.trades)}")
if len(result_spy.trades) > 0:
    print("\nSPY Alone Trades:")
    for idx, row in result_spy.trades.iterrows():
        print(f"  {idx.date()} - {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f}")

print("\n[2] Running basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=kalman_with_logging('basket'),
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

if not result_basket.trades.empty:
    spy_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'SPY']
    print(f"SPY trades: {len(spy_trades_basket)}")
    if len(spy_trades_basket) > 0:
        print("\nSPY in Basket Trades:")
        for idx, row in spy_trades_basket.iterrows():
            print(f"  {idx.date()} - {row['side']:4s} {row['quantity']:8.2f} @ ${row['price']:7.2f}")
else:
    print("SPY trades: 0")

# Compare signals
print("\n" + "="*80)
print("SIGNAL COMPARISON")
print("="*80)

dates = sorted(set(signal_log['alone'].keys()) | set(signal_log['basket'].keys()))

print("\n{:<12} {:<12} {:<12} {:<10} {:<10} {:<10} {:<12} {:<12} {:<10} {:<10}".format(
    "Date", "Fast(A)", "Fast(B)", "Slow(A)", "Slow(B)", "GC(A)", "GC(B)", "Pos(A)", "Pos(B)", "Match"
))
print("-" * 130)

for date in dates:
    alone = signal_log['alone'].get(date, {})
    basket = signal_log['basket'].get(date, {})

    if alone and basket:
        fast_a = alone.get('current_fast', 0)
        fast_b = basket.get('current_fast', 0)
        slow_a = alone.get('current_slow', 0)
        slow_b = basket.get('current_slow', 0)
        gc_a = 'YES' if alone.get('golden_cross') else 'no'
        gc_b = 'YES' if basket.get('golden_cross') else 'no'
        pos_a = f"{alone.get('current_qty', 0):.2f}"
        pos_b = f"{basket.get('current_qty', 0):.2f}"

        # Check if signals match
        match = "OK" if (alone.get('golden_cross') == basket.get('golden_cross') and
                        alone.get('death_cross') == basket.get('death_cross') and
                        abs(fast_a - fast_b) < 0.01) else "DIFF"

        print(f"{date:<12} {fast_a:<12.4f} {fast_b:<12.4f} {slow_a:<10.4f} {slow_b:<10.4f} {gc_a:<10} {gc_b:<10} {pos_a:<12} {pos_b:<12} {match}")

print("\n" + "="*80)
print("DETAILED CHECK FOR 2015-10-23")
print("="*80)

target = '2015-10-23'
if target in signal_log['alone'] and target in signal_log['basket']:
    alone = signal_log['alone'][target]
    basket = signal_log['basket'][target]

    print(f"\nSPY ALONE:")
    for k, v in alone.items():
        print(f"  {k}: {v}")

    print(f"\nSPY IN BASKET:")
    for k, v in basket.items():
        print(f"  {k}: {v}")

    print(f"\nDIFFERENCES:")
    for k in alone.keys():
        if alone[k] != basket[k]:
            print(f"  {k}: {alone[k]} vs {basket[k]}")

print("\n" + "="*80)
