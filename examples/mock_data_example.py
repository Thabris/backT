"""
Mock Data Example for BackT

Demonstrates how to use synthetic data for testing strategies without
relying on external data sources. Perfect for development and testing.

Usage:
    python mock_data_example.py
    python mock_data_example.py --scenario bull
    python mock_data_example.py --scenario bear --seed 123
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
import pandas as pd
from typing import Dict, Any

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BackT components
from backt import Backtester, BacktestConfig, MockDataLoader
from backt.signal import TechnicalIndicators, StrategyHelpers
from backt.utils.config import ExecutionConfig


def simple_momentum_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Simple momentum strategy for testing with mock data

    Args:
        market_data: Dictionary of DataFrames with OHLCV data
        current_time: Current timestamp in backtest
        positions: Current portfolio positions
        context: Strategy state storage
        params: Strategy parameters

    Returns:
        Dictionary of orders to execute
    """

    # Strategy parameters
    lookback_days = params.get('lookback_days', 60)  # 3 months lookback
    min_momentum = params.get('min_momentum', 0.05)  # 5% minimum momentum

    orders = {}
    momentum_scores = {}

    # Calculate momentum for each asset
    for symbol, data in market_data.items():
        if len(data) < lookback_days + 1:
            continue

        try:
            # Calculate simple price momentum
            current_price = data['close'].iloc[-1]
            past_price = data['close'].iloc[-lookback_days]

            if past_price > 0:
                momentum = (current_price / past_price) - 1.0
                momentum_scores[symbol] = momentum

                # Go long if momentum is positive and above threshold
                if momentum > min_momentum:
                    orders[symbol] = {
                        'action': 'target_weight',
                        'weight': 0.25  # Equal weight across up to 4 assets
                    }

        except Exception as e:
            print(f"Warning: Error calculating momentum for {symbol}: {e}")

    # Close positions for assets with poor momentum
    for symbol in market_data.keys():
        if symbol not in [s for s, o in orders.items() if 'target_weight' in o]:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store context for analysis
    context['momentum_scores'] = momentum_scores
    context['signal_count'] = len([o for o in orders.values() if 'target_weight' in o])

    return orders


def run_mock_backtest(
    scenario: str = "normal",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100000.0,
    seed: int = 42
):
    """
    Run backtest using mock data

    Args:
        scenario: Mock market scenario ('normal', 'bull', 'bear', 'volatile')
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital
        seed: Random seed for reproducible results

    Returns:
        BacktestResult object
    """

    # Universe of synthetic assets
    universe = ['SPY', 'QQQ', 'TLT', 'GLD', 'IWM', 'EFA', 'VNQ', 'XLE']

    print(f"\n{'='*60}")
    print(f"BackT Mock Data Example")
    print(f"{'='*60}")
    print(f"Scenario: {scenario.upper()}")
    print(f"Universe: {', '.join(universe)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Random Seed: {seed}")
    print(f"{'='*60}\n")

    # Configure execution
    execution_config = ExecutionConfig(
        spread=0.01,           # 1% spread
        slippage_pct=0.001,   # 0.1% slippage
        commission_per_share=0.0,
        commission_per_trade=0.0
    )

    # Create backtest configuration with mock data enabled
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        data_frequency='1D',
        execution=execution_config,
        verbose=True,
        # Mock data settings
        use_mock_data=True,
        mock_scenario=scenario,
        mock_seed=seed
    )

    # Strategy parameters
    strategy_params = {
        'lookback_days': 60,
        'min_momentum': 0.05
    }

    print("Initializing backtester with mock data...")

    # Run backtest
    backtester = Backtester(config)

    try:
        result = backtester.run(
            strategy=simple_momentum_strategy,
            universe=universe,
            strategy_params=strategy_params
        )

        print("[SUCCESS] Mock data backtest completed!\n")
        return result

    except Exception as e:
        print(f"[ERROR] Backtest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def display_results(result, scenario: str):
    """
    Display backtest results

    Args:
        result: BacktestResult object
        scenario: Market scenario name
    """

    if not result:
        print("[ERROR] No results to display")
        return

    metrics = result.performance_metrics

    print(f"{'='*50}")
    print(f"MOCK DATA BACKTEST RESULTS ({scenario.upper()})")
    print(f"{'='*50}")

    # Key Performance Metrics
    print(f"\n[PERFORMANCE METRICS]")
    print(f"{'-'*30}")
    print(f"Total Return:       {metrics.get('total_return', 0):.2%}")
    print(f"Annualized Return:  {metrics.get('annualized_return', 0):.2%}")
    print(f"Volatility:         {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Maximum Drawdown:   {metrics.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio:       {metrics.get('calmar_ratio', 0):.3f}")

    # Trade Information
    print(f"\n[TRADE INFORMATION]")
    print(f"{'-'*30}")
    print(f"Total Trades:       {metrics.get('total_trades', 0)}")
    print(f"Win Rate:           {metrics.get('win_rate', 0):.1%}")
    print(f"Profit Factor:      {metrics.get('profit_factor', 0):.2f}")

    # Portfolio Information
    print(f"\n[PORTFOLIO INFO]")
    print(f"{'-'*30}")

    # Use performance metrics to calculate portfolio values (more reliable)
    initial_value = 100000.0  # Standard starting value
    total_return = metrics.get('total_return', 0.0)
    final_value = initial_value * (1 + total_return)

    print(f"Start Value:        ${initial_value:,.0f}")
    print(f"Final Value:        ${final_value:,.0f}")
    print(f"Profit/Loss:        ${final_value - initial_value:,.0f}")
    print(f"Return:             {total_return:.2%}")

    # Risk Metrics
    if metrics.get('best_month') is not None:
        print(f"\n[RISK METRICS]")
        print(f"{'-'*30}")
        print(f"Best Month:         {metrics.get('best_month', 0):.2%}")
        print(f"Worst Month:        {metrics.get('worst_month', 0):.2%}")
        print(f"Positive Months:    {metrics.get('positive_months_pct', 0):.1%}")

    print(f"\n{'='*50}")


def compare_scenarios():
    """
    Compare results across different market scenarios
    """

    scenarios = ['normal', 'bull', 'bear', 'volatile']
    results = {}

    print(f"\n{'='*60}")
    print(f"COMPARING MOCK DATA SCENARIOS")
    print(f"{'='*60}")

    for scenario in scenarios:
        print(f"\nRunning {scenario} scenario...")
        result = run_mock_backtest(
            scenario=scenario,
            start_date="2020-01-01",
            end_date="2022-12-31",
            initial_capital=100000,
            seed=42  # Same seed for fair comparison
        )

        if result:
            results[scenario] = result

    # Display comparison
    if results:
        print(f"\n{'='*70}")
        print(f"SCENARIO COMPARISON")
        print(f"{'='*70}")
        print(f"{'Scenario':<12} {'Total Ret':<10} {'Sharpe':<8} {'Max DD':<8} {'Trades':<8}")
        print(f"{'-'*70}")

        for scenario, result in results.items():
            metrics = result.performance_metrics
            print(f"{scenario.capitalize():<12} "
                  f"{metrics.get('total_return', 0):>8.1%} "
                  f"{metrics.get('sharpe_ratio', 0):>7.2f} "
                  f"{metrics.get('max_drawdown', 0):>7.1%} "
                  f"{metrics.get('total_trades', 0):>7.0f}")

        print(f"{'='*70}")


def main():
    """
    Main execution function
    """

    parser = argparse.ArgumentParser(description='BackT Mock Data Example')
    parser.add_argument('--scenario', type=str, default='normal',
                       choices=['normal', 'bull', 'bear', 'volatile'],
                       help='Market scenario to simulate')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all scenarios')

    args = parser.parse_args()

    try:
        if args.compare:
            # Run comparison across scenarios
            compare_scenarios()
        else:
            # Run single scenario
            result = run_mock_backtest(
                scenario=args.scenario,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.capital,
                seed=args.seed
            )

            if result:
                display_results(result, args.scenario)

                # Validation
                total_return = result.performance_metrics.get('total_return', 0.0)
                initial_value = args.capital
                final_value = initial_value * (1 + total_return)
                profit_loss = final_value - initial_value
                if profit_loss > 0:
                    print(f"\n[SUCCESS] Strategy was profitable: +${profit_loss:,.0f}")
                else:
                    print(f"\n[INFO] Strategy had losses: ${profit_loss:,.0f}")

                print(f"\n[MOCK DATA] All data was synthetically generated")
                print(f"[REPRODUCIBLE] Use same seed ({args.seed}) for identical results")

        return True

    except Exception as e:
        print(f"\n[ERROR] Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()