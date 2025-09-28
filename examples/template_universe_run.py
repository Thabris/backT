"""
Template Universe Run Example for BackT

This example demonstrates a typical BackT workflow with:
- Multi-asset universe (8 ETFs across asset classes)
- Configurable date parameters
- Simple momentum strategy implementation
- Comprehensive metrics output and visualization

Usage:
    python template_universe_run.py

Or customize parameters:
    python template_universe_run.py --start-date 2020-01-01 --end-date 2023-12-31
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date
import pandas as pd
from typing import Dict, Any

# Add parent directory to path to find backt module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers
from backt.utils.config import ExecutionConfig
from backt.reporting import ReportGenerator


def diversified_momentum_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Diversified momentum strategy across multiple asset classes

    Strategy Logic:
    1. Calculate 12-month momentum for each asset (excluding last month)
    2. Select assets with positive momentum
    3. Equal weight allocation across selected assets
    4. Monthly rebalancing

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
    lookback_months = params.get('lookback_months', 12)
    skip_months = params.get('skip_months', 1)
    min_momentum_threshold = params.get('min_momentum_threshold', 0.0)
    max_assets = params.get('max_assets', None)  # None = no limit

    orders = {}
    momentum_scores = {}

    # Calculate momentum for each asset
    for symbol, data in market_data.items():
        if len(data) < lookback_months + skip_months:
            continue

        # Calculate momentum (total return over lookback period, skipping recent months)
        end_idx = len(data) - skip_months
        start_idx = end_idx - lookback_months

        if start_idx < 0 or end_idx <= start_idx:
            continue

        start_price = data['close'].iloc[start_idx]
        end_price = data['close'].iloc[end_idx - 1]

        momentum = (end_price / start_price) - 1.0
        momentum_scores[symbol] = momentum

    # Filter assets with positive momentum above threshold
    positive_momentum_assets = {
        symbol: score for symbol, score in momentum_scores.items()
        if score > min_momentum_threshold
    }

    # Limit number of assets if specified
    if max_assets and len(positive_momentum_assets) > max_assets:
        # Sort by momentum and take top assets
        sorted_assets = sorted(
            positive_momentum_assets.items(),
            key=lambda x: x[1],
            reverse=True
        )
        positive_momentum_assets = dict(sorted_assets[:max_assets])

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
            if positions[symbol].quantity != 0:
                orders[symbol] = {'action': 'close'}

    # Store momentum scores in context for analysis
    context['momentum_scores'] = momentum_scores
    context['selected_assets'] = list(positive_momentum_assets.keys())

    return orders


def run_universe_backtest(
    start_date: str = "2018-01-01",
    end_date: str = "2024-01-01",
    initial_capital: float = 100000.0,
    universe: list = None,
    strategy_params: dict = None
):
    """
    Run backtest on diversified universe with comprehensive output

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Starting capital amount
        universe: List of symbols to trade
        strategy_params: Strategy parameter overrides

    Returns:
        BacktestResult object with comprehensive metrics
    """

    # Default diversified universe (8 major asset classes)
    if universe is None:
        universe = [
            'SPY',  # US Large Cap Stocks
            'EFA',  # Developed International Stocks
            'EEM',  # Emerging Market Stocks
            'TLT',  # Long-term US Treasuries
            'IEF',  # Intermediate US Treasuries
            'VNQ',  # US REITs
            'DBC',  # Commodities
            'GLD'   # Gold
        ]

    # Default strategy parameters
    default_params = {
        'lookback_months': 12,
        'skip_months': 1,
        'min_momentum_threshold': 0.0,
        'max_assets': None
    }

    if strategy_params:
        default_params.update(strategy_params)

    # Configure execution with realistic costs
    execution_config = ExecutionConfig(
        spread=0.01,                    # 1% bid-ask spread
        slippage_pct=0.001,            # 0.1% slippage
        commission_per_share=0.0,       # Commission-free ETF trading
        commission_per_trade=0.0        # No flat fee
    )

    # Create backtest configuration
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        data_frequency='1M',            # Monthly rebalancing
        execution=execution_config,
        verbose=True
    )

    print(f"\n{'='*60}")
    print(f"BackT Universe Backtest Template")
    print(f"{'='*60}")
    print(f"Universe: {', '.join(universe)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.0f}")
    print(f"Strategy: Diversified Momentum")
    print(f"Parameters: {default_params}")
    print(f"{'='*60}\n")

    # Initialize backtester
    backtester = Backtester(config)

    # Run backtest
    print("Loading data and running backtest...")
    try:
        result = backtester.run(
            strategy=diversified_momentum_strategy,
            symbols=universe,
            strategy_params=default_params
        )

        print("✅ Backtest completed successfully!\n")
        return result

    except Exception as e:
        print(f"❌ Backtest failed: {str(e)}")
        raise


def display_comprehensive_metrics(result):
    """
    Display comprehensive metrics from backtest result

    Args:
        result: BacktestResult object
    """

    metrics = result.performance_metrics

    print(f"{'='*60}")
    print(f"COMPREHENSIVE PERFORMANCE METRICS")
    print(f"{'='*60}")

    # Return Metrics
    print(f"\n📈 RETURN METRICS")
    print(f"{'-'*30}")
    print(f"Total Return:           {metrics.get('total_return', 0):.2%}")
    print(f"Annualized Return:      {metrics.get('annualized_return', 0):.2%}")
    print(f"CAGR:                   {metrics.get('cagr', 0):.2%}")

    # Risk Metrics
    print(f"\n⚠️  RISK METRICS")
    print(f"{'-'*30}")
    print(f"Volatility (Annual):    {metrics.get('volatility', 0):.2%}")
    print(f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"Sortino Ratio:          {metrics.get('sortino_ratio', 0):.3f}")
    print(f"Calmar Ratio:           {metrics.get('calmar_ratio', 0):.3f}")

    # Drawdown Analysis
    print(f"\n📉 DRAWDOWN ANALYSIS")
    print(f"{'-'*30}")
    print(f"Maximum Drawdown:       {metrics.get('max_drawdown', 0):.2%}")
    print(f"Avg Drawdown:           {metrics.get('avg_drawdown', 0):.2%}")
    print(f"Max DD Duration:        {metrics.get('max_drawdown_duration', 0)} periods")

    # Trade Statistics
    print(f"\n📊 TRADE STATISTICS")
    print(f"{'-'*30}")
    print(f"Total Trades:           {metrics.get('total_trades', 0)}")
    print(f"Win Rate:               {metrics.get('win_rate', 0):.1%}")
    print(f"Profit Factor:          {metrics.get('profit_factor', 0):.2f}")
    print(f"Average Trade:          {metrics.get('avg_trade_return', 0):.2%}")

    # Portfolio Statistics
    print(f"\n💼 PORTFOLIO STATISTICS")
    print(f"{'-'*30}")
    print(f"Final Portfolio Value:  ${result.portfolio_value:.0f}")
    print(f"Best Month:             {metrics.get('best_month', 0):.2%}")
    print(f"Worst Month:            {metrics.get('worst_month', 0):.2%}")
    print(f"% Positive Months:      {metrics.get('positive_months_pct', 0):.1%}")

    # Risk-Adjusted Metrics
    print(f"\n🎯 RISK-ADJUSTED METRICS")
    print(f"{'-'*30}")
    print(f"Information Ratio:      {metrics.get('information_ratio', 0):.3f}")
    print(f"VaR (95%):              {metrics.get('var_95', 0):.2%}")
    print(f"CVaR (95%):             {metrics.get('cvar_95', 0):.2%}")
    print(f"Beta:                   {metrics.get('beta', 0):.3f}")

    print(f"\n{'='*60}")


def create_summary_table(result):
    """
    Create a summary table of key metrics

    Args:
        result: BacktestResult object

    Returns:
        pandas DataFrame with summary metrics
    """

    metrics = result.performance_metrics

    summary_data = {
        'Metric': [
            'Total Return',
            'Annualized Return',
            'Volatility',
            'Sharpe Ratio',
            'Maximum Drawdown',
            'Calmar Ratio',
            'Win Rate',
            'Total Trades',
            'Final Value'
        ],
        'Value': [
            f"{metrics.get('total_return', 0):.2%}",
            f"{metrics.get('annualized_return', 0):.2%}",
            f"{metrics.get('volatility', 0):.2%}",
            f"{metrics.get('sharpe_ratio', 0):.3f}",
            f"{metrics.get('max_drawdown', 0):.2%}",
            f"{metrics.get('calmar_ratio', 0):.3f}",
            f"{metrics.get('win_rate', 0):.1%}",
            f"{metrics.get('total_trades', 0)}",
            f"${result.portfolio_value:,.0f}"
        ]
    }

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def main():
    """
    Main execution function with command line argument parsing
    """

    parser = argparse.ArgumentParser(description='BackT Universe Template Run')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--lookback', type=int, default=12,
                       help='Momentum lookback months')
    parser.add_argument('--skip', type=int, default=1,
                       help='Skip recent months')
    parser.add_argument('--threshold', type=float, default=0.0,
                       help='Minimum momentum threshold')
    parser.add_argument('--max-assets', type=int, default=None,
                       help='Maximum number of assets to hold')

    args = parser.parse_args()

    # Strategy parameters from command line
    strategy_params = {
        'lookback_months': args.lookback,
        'skip_months': args.skip,
        'min_momentum_threshold': args.threshold,
        'max_assets': args.max_assets
    }

    try:
        # Run backtest
        result = run_universe_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            strategy_params=strategy_params
        )

        # Display results
        display_comprehensive_metrics(result)

        # Create summary table
        summary_table = create_summary_table(result)
        print(f"\n📋 SUMMARY TABLE")
        print(f"{'-'*30}")
        print(summary_table.to_string(index=False))

        # Save results (optional)
        try:
            # Save equity curve
            equity_df = pd.DataFrame({
                'Date': result.equity_curve.index,
                'Portfolio_Value': result.equity_curve.values,
                'Returns': result.returns
            })

            output_file = f"backtest_results_{args.start_date}_{args.end_date}.csv"
            equity_df.to_csv(output_file, index=False)
            print(f"\n💾 Results saved to: {output_file}")

        except Exception as e:
            print(f"\n⚠️  Could not save results: {e}")

        return result

    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        return None


if __name__ == "__main__":
    result = main()