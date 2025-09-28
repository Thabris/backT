"""
Reliable Universe Test for BackT

This example uses symbols that are typically more reliable with Yahoo Finance
and includes better error handling for data loading issues.

Usage:
    python reliable_universe_test.py
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date
import pandas as pd
from typing import Dict, Any, List

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers
from backt.utils.config import ExecutionConfig


def test_data_availability(symbols: List[str], start_date: str, end_date: str) -> List[str]:
    """
    Test which symbols have data available and return working subset

    Args:
        symbols: List of symbols to test
        start_date: Start date for data test
        end_date: End date for data test

    Returns:
        List of symbols with available data
    """
    from backt.data import YahooDataLoader

    loader = YahooDataLoader()
    working_symbols = []

    print("Testing data availability for symbols...")

    for symbol in symbols:
        try:
            # Test single symbol
            data = loader.load([symbol], start_date, end_date)
            if data and symbol in data and len(data[symbol]) > 100:  # Minimum 100 data points
                working_symbols.append(symbol)
                print(f"  ✓ {symbol}: {len(data[symbol])} data points available")
            else:
                print(f"  ✗ {symbol}: Insufficient data")
        except Exception as e:
            print(f"  ✗ {symbol}: Error - {str(e)[:50]}...")

    return working_symbols


def simple_momentum_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Simple momentum strategy with robust implementation

    Args:
        market_data: Dictionary of DataFrames with OHLCV data for each symbol
        current_time: Current timestamp in backtest
        positions: Current portfolio positions
        context: Strategy state storage
        params: Strategy parameters

    Returns:
        Dictionary of orders to execute
    """

    # Strategy parameters
    lookback_days = params.get('lookback_days', 252)  # ~1 year
    skip_days = params.get('skip_days', 5)  # Skip recent days
    min_momentum_threshold = params.get('min_momentum_threshold', 0.0)

    orders = {}
    momentum_scores = {}

    # Calculate momentum for each asset
    for symbol, data in market_data.items():
        if len(data) < lookback_days + skip_days + 1:
            continue

        try:
            # Calculate momentum (total return over lookback period, skipping recent days)
            end_idx = len(data) - skip_days
            start_idx = end_idx - lookback_days

            if start_idx < 0 or end_idx <= start_idx:
                continue

            start_price = data['close'].iloc[start_idx]
            end_price = data['close'].iloc[end_idx - 1]

            if start_price > 0:  # Avoid division by zero
                momentum = (end_price / start_price) - 1.0
                momentum_scores[symbol] = momentum

        except Exception as e:
            print(f"Warning: Error calculating momentum for {symbol}: {e}")
            continue

    # Filter assets with positive momentum above threshold
    positive_momentum_assets = {
        symbol: score for symbol, score in momentum_scores.items()
        if score > min_momentum_threshold
    }

    # Equal weight allocation across selected assets
    if positive_momentum_assets:
        target_weight = 1.0 / len(positive_momentum_assets)

        for symbol in positive_momentum_assets:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': target_weight
            }

    # Close positions in assets not selected
    for symbol in market_data.keys():
        if symbol not in positive_momentum_assets and symbol in positions:
            if hasattr(positions[symbol], 'quantity') and positions[symbol].quantity != 0:
                orders[symbol] = {'action': 'close'}

    # Store information in context
    context['momentum_scores'] = momentum_scores
    context['selected_assets'] = list(positive_momentum_assets.keys())
    context['num_selected'] = len(positive_momentum_assets)

    return orders


def run_reliable_test(
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 100000.0
):
    """
    Run backtest with reliable symbols and robust error handling

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital amount

    Returns:
        BacktestResult object or None if failed
    """

    # More reliable symbol universe (major liquid ETFs)
    candidate_universe = [
        'SPY',   # S&P 500 - most liquid US ETF
        'QQQ',   # NASDAQ 100 - very liquid tech ETF
        'IWM',   # Russell 2000 small cap
        'TLT',   # 20+ Year Treasury
        'SHY',   # 1-3 Year Treasury
        'GLD',   # Gold
        'XLF',   # Financial sector
        'XLE',   # Energy sector
    ]

    print(f"\n{'='*60}")
    print(f"BackT Reliable Universe Test")
    print(f"{'='*60}")
    print(f"Candidate Universe: {', '.join(candidate_universe)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"{'='*60}\n")

    # Test data availability
    working_universe = test_data_availability(candidate_universe, start_date, end_date)

    if len(working_universe) < 2:
        print(f"\n[ERROR] Insufficient working symbols: {working_universe}")
        print("Need at least 2 symbols for meaningful backtest")
        return None

    print(f"\n[SUCCESS] Found {len(working_universe)} working symbols: {', '.join(working_universe)}")

    # Strategy parameters
    strategy_params = {
        'lookback_days': 252,  # 1 year lookback
        'skip_days': 5,        # Skip recent 5 days
        'min_momentum_threshold': 0.0
    }

    # Configure execution with realistic costs
    execution_config = ExecutionConfig(
        spread=0.005,                   # 0.5% bid-ask spread
        slippage_pct=0.0005,           # 0.05% slippage
        commission_per_share=0.0,       # Commission-free ETF trading
        commission_per_trade=0.0        # No flat fee
    )

    # Create backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        data_frequency='1D',            # Daily data
        execution=execution_config,
        verbose=True
    )

    print(f"\nRunning backtest with {len(working_universe)} symbols...")
    print(f"Strategy parameters: {strategy_params}")

    # Initialize backtester
    backtester = Backtester(config)

    try:
        # Run backtest
        result = backtester.run(
            strategy=simple_momentum_strategy,
            universe=working_universe,
            strategy_params=strategy_params
        )

        print("[SUCCESS] Backtest completed!\n")
        return result

    except Exception as e:
        print(f"[ERROR] Backtest failed: {str(e)}")
        return None


def display_simple_metrics(result):
    """
    Display key metrics from backtest result

    Args:
        result: BacktestResult object
    """

    if not result:
        print("[ERROR] No results to display")
        return

    metrics = result.performance_metrics

    print(f"{'='*50}")
    print(f"BACKTEST RESULTS")
    print(f"{'='*50}")

    # Key Performance Metrics
    print(f"\n[KEY METRICS]")
    print(f"{'-'*25}")
    print(f"Total Return:       {metrics.get('total_return', 0):.2%}")
    print(f"Annualized Return:  {metrics.get('annualized_return', 0):.2%}")
    print(f"Volatility:         {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Maximum Drawdown:   {metrics.get('max_drawdown', 0):.2%}")
    print(f"Calmar Ratio:       {metrics.get('calmar_ratio', 0):.3f}")

    # Portfolio Information
    print(f"\n[PORTFOLIO INFO]")
    print(f"{'-'*25}")
    print(f"Final Value:        ${result.portfolio_value:,.0f}")
    print(f"Total Trades:       {metrics.get('total_trades', 0)}")
    print(f"Win Rate:           {metrics.get('win_rate', 0):.1%}")

    # Risk Metrics
    print(f"\n[RISK METRICS]")
    print(f"{'-'*25}")
    print(f"Best Month:         {metrics.get('best_month', 0):.2%}")
    print(f"Worst Month:        {metrics.get('worst_month', 0):.2%}")
    print(f"Positive Months:    {metrics.get('positive_months_pct', 0):.1%}")

    print(f"\n{'='*50}")


def main():
    """
    Main execution function
    """

    parser = argparse.ArgumentParser(description='BackT Reliable Universe Test')
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-12-31',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')

    args = parser.parse_args()

    try:
        # Run test
        result = run_reliable_test(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital
        )

        if result:
            # Display results
            display_simple_metrics(result)

            # Basic validation
            if result.portfolio_value > 0:
                print(f"\n[VALIDATION] Backtest completed successfully!")
                print(f"Portfolio grew from ${args.capital:,.0f} to ${result.portfolio_value:,.0f}")
                return result
            else:
                print(f"\n[WARNING] Backtest completed but portfolio value is zero")
        else:
            print(f"\n[ERROR] Backtest failed - no results generated")

    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

    return None


if __name__ == "__main__":
    result = main()