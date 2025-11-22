"""
Test per-symbol metrics fix
"""
import logging
logging.getLogger('backt').setLevel(logging.WARNING)

from backt import Backtester, BacktestConfig
from strategies.momentum import kalman_ma_crossover_long_only

config = BacktestConfig(
    start_date='2012-01-02',
    end_date='2023-09-30',  # Before trades go down
    initial_capital=100000
)

params = {'Q_fast': 0.014, 'Q_slow': 0.0006}

print("="*80)
print("TEST: Per-Symbol Metrics Should Match Between Scenarios")
print("="*80)

print("\n[1] SPY alone (2012-2023)...")
result_spy = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY'],
    strategy_params=params
)

spy_equity = result_spy.equity_curve['total_equity']
spy_return = ((spy_equity.iloc[-1] / spy_equity.iloc[0]) - 1) * 100

print(f"    Trades: {len(result_spy.trades)}")
print(f"    Final equity: ${spy_equity.iloc[-1]:,.2f}")
print(f"    Total return: {spy_return:.2f}%")

# Get per-symbol metrics for SPY (single symbol case)
if result_spy.per_symbol_metrics and 'SPY' in result_spy.per_symbol_metrics:
    spy_metrics_alone = result_spy.per_symbol_metrics['SPY']
    print(f"    Sharpe (per-symbol): {spy_metrics_alone.get('sharpe_ratio', 0):.3f}")
    print(f"    Max DD (per-symbol): {spy_metrics_alone.get('max_drawdown', 0)*100:.2f}%")

print("\n[2] SPY in basket (SPY, QQQ, GLD, TLT)...")
result_basket = Backtester(config).run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'GLD', 'TLT'],
    strategy_params=params
)

spy_trades_basket = result_basket.trades[result_basket.trades['symbol'] == 'SPY']

print(f"    Total trades: {len(result_basket.trades)}")
print(f"    SPY trades: {len(spy_trades_basket)}")
print(f"    Portfolio final equity: ${result_basket.equity_curve['total_equity'].iloc[-1]:,.2f}")

# Get per-symbol metrics for SPY in basket
if result_basket.per_symbol_metrics:
    print("\n    Per-Symbol Metrics:")
    for symbol in ['SPY', 'QQQ', 'GLD', 'TLT']:
        if symbol in result_basket.per_symbol_metrics:
            m = result_basket.per_symbol_metrics[symbol]
            print(f"      {symbol}:")
            print(f"        Final equity: ${m.get('final_equity', 0):,.2f}")
            print(f"        Total PnL: ${m.get('total_pnl', 0):,.2f}")
            print(f"        Sharpe: {m.get('sharpe_ratio', 0):.3f}")
            print(f"        Max DD: {m.get('max_drawdown', 0)*100:.2f}%")

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

if result_spy.per_symbol_metrics and result_basket.per_symbol_metrics:
    spy_alone_sharpe = result_spy.per_symbol_metrics['SPY'].get('sharpe_ratio', 0)
    spy_basket_sharpe = result_basket.per_symbol_metrics['SPY'].get('sharpe_ratio', 0)

    print(f"\nSPY Sharpe Ratio:")
    print(f"  Alone: {spy_alone_sharpe:.3f}")
    print(f"  In basket: {spy_basket_sharpe:.3f}")
    print(f"  Difference: {abs(spy_alone_sharpe - spy_basket_sharpe):.3f}")

    if abs(spy_alone_sharpe - spy_basket_sharpe) < 0.05:  # Allow 0.05 tolerance
        print("\n[PASS] Sharpe ratios match (within tolerance)")
    else:
        print("\n[FAIL] Sharpe ratios differ significantly")

print("\n" + "="*80)
