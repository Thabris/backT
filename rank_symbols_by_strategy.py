"""
Rank Symbols by Strategy Performance

Command-line tool to backtest a strategy across all ETF symbols and rank them
by Sortino ratio. Outputs top 20 performers with detailed metrics and correlation matrix.

Usage:
    python rank_symbols_by_strategy.py --start 2020-01-01 --end 2023-12-31 --strategy ma_crossover_long_only

    # With checkpoint (recommended for long runs):
    python rank_symbols_by_strategy.py --start 2020-01-01 --end 2023-12-31 --strategy ma_crossover_long_only --checkpoint results.json

    # Save top symbols as a book for Streamlit:
    python rank_symbols_by_strategy.py --start 2020-01-01 --end 2023-12-31 --strategy macd_crossover --top 10 --save "MACD_Top10_2024"

    # Exclude highly correlated symbols (diversification):
    python rank_symbols_by_strategy.py --start 2020-01-01 --end 2023-12-31 --strategy ma_crossover_long_only --top 20 --exclude 80

    # Exclude high-risk symbols (RECOMMENDED for mean reversion strategies):
    python rank_symbols_by_strategy.py --start 2020-01-01 --end 2023-12-31 --strategy bollinger_mean_reversion --top 10 --exclude-high-risk

Arguments:
    --start             : Start date (YYYY-MM-DD)
    --end               : End date (YYYY-MM-DD)
    --strategy          : Strategy name (e.g., ma_crossover_long_only, kalman_ma_crossover_long_only, etc.)
    --top               : Number of top performers to show (default: 20)
    --capital           : Initial capital per symbol (default: 100000)
    --workers           : Number of parallel workers (default: 1 for reliability, 2-3 for speed)
    --checkpoint        : Checkpoint file for progress saving/resuming (optional but recommended)
    --save              : Save top symbols as a book with this name (optional)
    --exclude           : Exclude correlated symbols above threshold (optional)
                          Formats: "80", "0.8", "80%", "correlation>80%"
                          Keeps best performer from each correlated pair
    --exclude-high-risk : Exclude volatility ETFs and leveraged instruments (RECOMMENDED for mean reversion)

Rate Limiting:
    - Default workers=1 processes symbols sequentially (most reliable)
    - Each request has 1 second delay to avoid Yahoo Finance rate limiting
    - Failed symbols retry with exponential backoff (3, 9, 27 seconds)
    - Use --checkpoint to save progress and resume if interrupted

Correlation Filtering:
    - Use --exclude to filter out highly correlated symbols
    - Threshold: correlation coefficient (0-1 or 0-100%)
    - Algorithm: Iterates through top N symbols (best to worst)
      - For each symbol, checks correlation with all remaining symbols
      - If correlation exceeds threshold, removes the worse performer
      - Final list may have fewer than --top symbols
    - Example: --exclude 80 removes symbols with >80% correlation
    - Useful for building diversified portfolios

Book Saving:
    - Use --save to create a book with the top-ranked symbols
    - Book includes strategy name, parameters, and symbols
    - Metadata includes ranking date, average Sortino, and top performer
    - Books can be loaded in Streamlit for backtesting on different date ranges
    - Saved to: saved_books/<name>.json
    - If --exclude is used, only final filtered symbols are saved

Available Strategies:
    - ma_crossover_long_only
    - ma_crossover_long_short
    - kalman_ma_crossover_long_only
    - kalman_ma_crossover_long_short
    - rsi_mean_reversion
    - macd_crossover
    - stochastic_momentum
    - bollinger_mean_reversion
    - adx_trend_filter
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from backt import Backtester, BacktestConfig
from backt.data.sqlite_loader import SQLiteDataLoader
from backt.utils.books import BookManager, create_book_from_session
from strategies import momentum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
warnings.filterwarnings('ignore')


# ETF Universe - imported from streamlit_backtest_runner.py
ETF_UNIVERSE = {
    "Equities - Broad Market": {
        "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IVV": "S&P 500 (iShares)",
        "VOO": "S&P 500 (Vanguard)", "MDY": "S&P Mid Cap 400", "IJH": "S&P Mid Cap 400 (iShares)",
        "IWM": "Russell 2000 Small Cap", "VB": "Small Cap (Vanguard)", "VGK": "Europe Stocks",
        "IEV": "Europe 350", "EEM": "Emerging Markets", "VWO": "Emerging Markets (Vanguard)",
        "EWJ": "Japan", "FXI": "China Large Cap", "MCHI": "China", "EFA": "Developed ex-US",
    },
    "Equities - Factor/Style": {
        "VLUE": "Value Factor", "IWD": "Value Large Cap", "VTV": "Value (Vanguard)",
        "MTUM": "Momentum Factor", "QUAL": "Quality Factor", "SPHQ": "Quality (Invesco)",
        "USMV": "Low Volatility", "SPLV": "Low Volatility (Invesco)", "IJR": "Small Cap Value",
        "VBK": "Small Cap Growth", "LRGF": "Multifactor Large Cap", "ACWF": "Multifactor Global",
    },
    "Fixed Income": {
        "AGG": "US Aggregate Bond", "BND": "Total Bond Market", "SHY": "1-3 Year Treasury",
        "VGSH": "Short-Term Treasury", "TLT": "20+ Year Treasury", "IEF": "7-10 Year Treasury",
        "TIP": "TIPS Inflation-Protected", "SCHP": "TIPS (Schwab)", "LQD": "Investment Grade Corporate",
        "VCIT": "Intermediate Corporate", "HYG": "High Yield Corporate", "JNK": "High Yield (SPDR)",
        "EMB": "Emerging Market Bond", "VWOB": "Emerging Market Bond (Vanguard)",
    },
    "Commodities": {
        "DBC": "Commodity Index", "COMT": "Commodity Optimum Yield", "GSG": "Commodity Broad",
        "GLD": "Gold", "IAU": "Gold (iShares)", "SLV": "Silver", "USO": "Crude Oil WTI",
        "BNO": "Brent Crude", "DBA": "Agriculture", "DBB": "Industrial Metals",
    },
    "Currencies": {
        "UUP": "US Dollar Bull", "USDU": "US Dollar (WisdomTree)", "FXE": "Euro",
        "FXY": "Japanese Yen", "FXB": "British Pound", "CEW": "Emerging Market Currency",
    },
    "Volatility": {
        "VXX": "VIX Short-Term Futures", "UVXY": "VIX 2x Leveraged", "VIXM": "VIX Mid-Term Futures",
        "TAIL": "Tail Risk Hedge", "SVXY": "Short VIX",
    },
    "Alternative": {
        "RPAR": "Risk Parity ETF", "NTSX": "90/60 Stocks/Bonds", "AOR": "Moderate Allocation",
        "DBMF": "Managed Futures", "KMLM": "Managed Futures (KFA)", "FIG": "Global Macro",
        "GVAL": "Global Value", "COM": "Commodity Trend",
    },
    "Sector": {
        "XLK": "Technology", "VGT": "Technology (Vanguard)", "XLE": "Energy", "VDE": "Energy (Vanguard)",
        "XLF": "Financials", "VFH": "Financials (Vanguard)", "XLV": "Healthcare", "VHT": "Healthcare (Vanguard)",
        "XLI": "Industrials", "VIS": "Industrials (Vanguard)", "XLY": "Consumer Discretionary",
        "VCR": "Consumer Discretionary (Vanguard)", "XLU": "Utilities", "VPU": "Utilities (Vanguard)",
        "XLRE": "Real Estate", "VNQ": "REIT (Vanguard)",
    },
    "Leveraged/Inverse": {
        "SPXL": "S&P 500 3x Bull", "SPXS": "S&P 500 3x Bear", "TQQQ": "Nasdaq 100 3x Bull",
        "SQQQ": "Nasdaq 100 3x Bear", "TMF": "Treasury 3x Bull", "TMV": "Treasury 3x Bear",
        "UGL": "Gold 2x Bull", "UCO": "Crude Oil 2x Bull", "SCO": "Crude Oil 2x Bear",
    }
}


# High-risk symbols that should be excluded from most strategies
# These instruments have extreme volatility and can cause catastrophic losses when shorted
HIGH_RISK_SYMBOLS = {
    # Volatility ETFs - extreme decay/spikes, unlimited short risk
    'VXX', 'UVXY', 'VIXM', 'SVXY', 'VIXY',
    # Leveraged/Inverse - 2x-3x leverage, can blow up accounts
    'SPXL', 'SPXS', 'TQQQ', 'SQQQ', 'TMF', 'TMV',
    'UGL', 'UCO', 'SCO', 'SOXL', 'SOXS',
}


# Strategy mapping
STRATEGY_MAP = {
    'ma_crossover_long_only': momentum.ma_crossover_long_only,
    'ma_crossover_long_short': momentum.ma_crossover_long_short,
    'kalman_ma_crossover_long_only': momentum.kalman_ma_crossover_long_only,
    'kalman_ma_crossover_long_short': momentum.kalman_ma_crossover_long_short,
    'rsi_mean_reversion': momentum.rsi_mean_reversion,
    'macd_crossover': momentum.macd_crossover,
    'stochastic_momentum': momentum.stochastic_momentum,
    'bollinger_mean_reversion': momentum.bollinger_mean_reversion,
    'adx_trend_filter': momentum.adx_trend_filter,
}


# Default strategy parameters
DEFAULT_PARAMS = {
    'ma_crossover_long_only': {'fast_ma': 20, 'slow_ma': 50, 'min_periods': 60},
    'ma_crossover_long_short': {'fast_ma': 20, 'slow_ma': 50, 'min_periods': 60},
    'kalman_ma_crossover_long_only': {'Q_fast': 0.014, 'Q_slow': 0.0006, 'R': 1.0, 'min_periods': 60},
    'kalman_ma_crossover_long_short': {'Q_fast': 0.01, 'Q_slow': 0.001, 'R': 1.0, 'min_periods': 60},
    'rsi_mean_reversion': {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70, 'min_periods': 30},
    'macd_crossover': {'fast_period': 11, 'slow_period': 20, 'signal_period': 10, 'allow_short': False, 'min_periods': 35},
    'stochastic_momentum': {'period': 14, 'oversold': 20, 'overbought': 80, 'min_periods': 30},
    'bollinger_mean_reversion': {'bb_period': 20, 'bb_std': 2.0, 'min_periods': 30},
    'adx_trend_filter': {'adx_period': 14, 'adx_threshold': 25, 'allow_short': False, 'min_periods': 30},
}


def get_all_symbols(exclude_high_risk: bool = False) -> List[str]:
    """
    Get all unique symbols from the ETF universe

    Args:
        exclude_high_risk: If True, filter out volatility ETFs and leveraged instruments

    Returns:
        Sorted list of symbols
    """
    symbols = []
    for category_etfs in ETF_UNIVERSE.values():
        symbols.extend(category_etfs.keys())

    unique_symbols = sorted(list(set(symbols)))

    if exclude_high_risk:
        unique_symbols = [s for s in unique_symbols if s not in HIGH_RISK_SYMBOLS]

    return unique_symbols


def run_single_symbol_backtest(
    symbol: str,
    strategy_func,
    strategy_params: Dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
    max_retries: int = 3
) -> Tuple[str, Dict]:
    """
    Run backtest for a single symbol with retry logic

    Returns:
        Tuple of (symbol, metrics_dict) or (symbol, None) if failed
    """
    for attempt in range(max_retries):
        try:
            # Add delay to avoid rate limiting (always, not just on retry)
            time.sleep(1.0)  # 1 second delay between each request (increased from 500ms)

            if attempt > 0:
                wait_time = 3 ** attempt  # Exponential backoff: 3, 9, 27 seconds
                time.sleep(wait_time)

            # Create config
            config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                verbose=False
            )

            # Create data loader - using SQLite database (no rate limiting!)
            data_loader = SQLiteDataLoader("market_data.db")

            # Create backtester
            backtester = Backtester(config, data_loader=data_loader)

            # Run backtest on single symbol
            result = backtester.run(
                strategy=strategy_func,
                universe=[symbol],
                strategy_params=strategy_params
            )

            # Extract metrics
            metrics = result.performance_metrics

            # Add symbol and trade count
            metrics['symbol'] = symbol
            metrics['total_trades'] = len(result.trades)

            # Get equity curve for correlation later
            metrics['equity_curve'] = result.equity_curve['total_equity'] if not result.equity_curve.empty else None

            return (symbol, metrics)

        except Exception as e:
            # On last retry, give up
            if attempt == max_retries - 1:
                return (symbol, None)
            # Otherwise, will retry after delay
            continue

    # Failed after all retries
    return (symbol, None)


def run_parallel_backtests(
    symbols: List[str],
    strategy_func,
    strategy_params: Dict,
    start_date: str,
    end_date: str,
    initial_capital: float,
    max_workers: int = 10,
    checkpoint_file: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Run backtests in parallel across all symbols with checkpoint support

    Args:
        checkpoint_file: Path to save/load progress (enables resume on failure)

    Returns:
        Dict mapping symbol to metrics
    """
    results = {}

    # Load checkpoint if exists
    if checkpoint_file and Path(checkpoint_file).exists():
        try:
            with open(checkpoint_file, 'r') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} results from checkpoint: {checkpoint_file}")
            # Remove already completed symbols
            symbols = [s for s in symbols if s not in results]
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    total = len(symbols)
    if total == 0:
        print("\nAll symbols already completed!")
        return results

    completed = 0

    print(f"\nRunning backtests on {total} symbols...")
    print(f"Using {max_workers} parallel workers")
    if checkpoint_file:
        print(f"Checkpoint file: {checkpoint_file}")
    print("-" * 80)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with staggered start to avoid overwhelming Yahoo API
        future_to_symbol = {}
        for i, symbol in enumerate(symbols):
            # Add small delay between submissions to spread out requests
            if i > 0 and i % max_workers == 0:
                time.sleep(1)  # Pause every batch of workers

            future = executor.submit(
                run_single_symbol_backtest,
                symbol,
                strategy_func,
                strategy_params,
                start_date,
                end_date,
                initial_capital,
                max_retries=3
            )
            future_to_symbol[future] = symbol

        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol, metrics = future.result()
            completed += 1

            if metrics is not None:
                results[symbol] = metrics
                print(f"[{completed}/{total}] {symbol}: Sortino = {metrics.get('sortino_ratio', 0):.3f}")
            else:
                print(f"[{completed}/{total}] {symbol}: FAILED (no data or error)")

            # Save checkpoint after each completion
            if checkpoint_file:
                try:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(results, f, indent=2, default=str)
                except Exception as e:
                    print(f"Warning: Could not save checkpoint: {e}")

    print("-" * 80)
    print(f"Completed: {len(results)} successful, {total - len(results)} failed\n")

    return results


def create_performance_table(top_symbols: List[Tuple[str, Dict]], top_n: int) -> str:
    """
    Create a formatted performance table for top symbols

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("\n" + "="*131)
    lines.append(f"TOP {top_n} SYMBOLS BY SORTINO RATIO")
    lines.append("="*131)

    # Header
    header = f"{'Rank':<5} {'Symbol':<8} {'Sortino':<9} {'Total Ret':<11} {'CAGR':<9} {'Sharpe':<9} {'Max DD':<9} {'Recov Days':<11} {'Volatility':<11} {'Calmar':<9} {'Trades':<7}"
    lines.append(header)
    lines.append("-"*131)

    # Data rows
    for rank, (symbol, metrics) in enumerate(top_symbols, 1):
        sortino = metrics.get('sortino_ratio', 0)
        total_return = metrics.get('total_return', 0)
        cagr = metrics.get('cagr', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        max_dd = metrics.get('max_drawdown', 0)
        max_recov_days = metrics.get('max_days_to_recover', 0)
        volatility = metrics.get('annualized_volatility', 0)
        calmar = metrics.get('calmar_ratio', 0)
        trades = metrics.get('total_trades', 0)

        # Handle infinite/nan values
        if not np.isfinite(calmar):
            calmar_str = "N/A"
        else:
            calmar_str = f"{calmar:.3f}"

        row = f"{rank:<5} {symbol:<8} {sortino:<9.3f} {total_return:<11.2%} {cagr:<9.2%} {sharpe:<9.3f} {max_dd:<9.2%} {max_recov_days:<11.0f} {volatility:<11.2%} {calmar_str:<9} {trades:<7}"
        lines.append(row)

    lines.append("="*131)

    return "\n".join(lines)


def create_correlation_matrix(top_symbols: List[Tuple[str, Dict]]) -> str:
    """
    Create correlation matrix for top symbols

    Returns:
        Formatted string correlation matrix
    """
    # Extract equity curves
    equity_data = {}
    for symbol, metrics in top_symbols:
        if metrics.get('equity_curve') is not None:
            equity_data[symbol] = metrics['equity_curve']

    if len(equity_data) < 2:
        return "\nInsufficient data for correlation matrix\n"

    # Create DataFrame
    df = pd.DataFrame(equity_data)

    # Calculate returns
    returns = df.pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = returns.corr()

    # Format output
    lines = []
    lines.append("\n" + "="*120)
    lines.append("CORRELATION MATRIX (Returns)")
    lines.append("="*120)

    # Create formatted table
    symbols = corr_matrix.columns.tolist()

    # Header row
    header = "Symbol   " + "  ".join([f"{s:>6}" for s in symbols])
    lines.append(header)
    lines.append("-"*120)

    # Data rows
    for symbol in symbols:
        row_values = [f"{corr_matrix.loc[symbol, s]:>6.3f}" for s in symbols]
        row = f"{symbol:<8} " + "  ".join(row_values)
        lines.append(row)

    lines.append("="*120)

    # Summary statistics
    lines.append("\nCorrelation Summary:")
    lines.append(f"  Average correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
    lines.append(f"  Min correlation:     {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].min():.3f}")
    lines.append(f"  Max correlation:     {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")

    return "\n".join(lines)


def filter_correlated_symbols_with_backfill(
    all_sorted_symbols: List[Tuple[str, Dict]],
    target_count: int,
    correlation_threshold: float
) -> Tuple[List[Tuple[str, Dict]], List[Tuple[str, str, float]], int]:
    """
    Filter out highly correlated symbols with backfilling from lower-ranked symbols

    Algorithm:
    1. Start with top N symbols as candidates
    2. Remove correlated pairs (keep best performer)
    3. If we have < N symbols, add next ranked symbol from the full list
    4. Re-check correlations with new symbol
    5. Repeat until we have N symbols OR exhaust all candidates

    Args:
        all_sorted_symbols: Full list of (symbol, metrics) sorted by Sortino (best first)
        target_count: Target number of symbols to return
        correlation_threshold: Correlation threshold (0-1), e.g., 0.8 for 80%

    Returns:
        Tuple of (filtered_symbols, excluded_pairs, candidates_tested)
        - filtered_symbols: Final list of symbols (up to target_count)
        - excluded_pairs: List of (excluded_symbol, kept_symbol, correlation) tuples
        - candidates_tested: Number of symbols tested (includes backfills)
    """
    if len(all_sorted_symbols) < 2:
        return all_sorted_symbols[:target_count], [], len(all_sorted_symbols)

    # Extract equity curves for correlation calculation
    all_equity_data = {}
    for symbol, metrics in all_sorted_symbols:
        if metrics.get('equity_curve') is not None:
            all_equity_data[symbol] = metrics['equity_curve']

    if len(all_equity_data) < 2:
        return all_sorted_symbols[:target_count], [], len(all_sorted_symbols)

    # Calculate correlation matrix for all symbols
    df = pd.DataFrame(all_equity_data)
    returns = df.pct_change().dropna()
    corr_matrix = returns.corr()

    # Track portfolio and excluded symbols
    portfolio = []  # Final portfolio (symbol, metrics) tuples
    portfolio_symbols = set()  # For fast lookup
    excluded_pairs = []  # Track what was excluded and why
    candidate_index = 0  # Index into all_sorted_symbols

    # Iteratively build portfolio
    while len(portfolio) < target_count and candidate_index < len(all_sorted_symbols):
        candidate_symbol, candidate_metrics = all_sorted_symbols[candidate_index]
        candidate_index += 1

        # Skip if no correlation data
        if candidate_symbol not in corr_matrix.index:
            continue

        # Check correlation with all symbols already in portfolio
        is_correlated = False
        for portfolio_symbol, _ in portfolio:
            if portfolio_symbol not in corr_matrix.index:
                continue

            correlation = corr_matrix.loc[candidate_symbol, portfolio_symbol]

            # If correlated with any portfolio symbol, reject candidate
            if abs(correlation) >= correlation_threshold:
                excluded_pairs.append((candidate_symbol, portfolio_symbol, correlation))
                is_correlated = True
                break  # No need to check other portfolio symbols

        # If not correlated with any portfolio symbol, add to portfolio
        if not is_correlated:
            portfolio.append((candidate_symbol, candidate_metrics))
            portfolio_symbols.add(candidate_symbol)

    return portfolio, excluded_pairs, candidate_index


def save_top_symbols_as_book(
    book_name: str,
    top_symbols: List[Tuple[str, Dict]],
    strategy_name: str,
    strategy_params: Dict,
    start_date: str,
    end_date: str
) -> None:
    """
    Save top-ranked symbols as a book for later use in Streamlit

    Args:
        book_name: Name for the book
        top_symbols: List of (symbol, metrics) tuples
        strategy_name: Strategy name
        strategy_params: Strategy parameters
        start_date: Start date used for ranking
        end_date: End date used for ranking
    """
    # Extract just the symbols
    symbols = [symbol for symbol, _ in top_symbols]

    # Determine strategy module (all current strategies are in MOMENTUM)
    strategy_module = "MOMENTUM"

    # Create tags
    tags = [
        "ranked",
        strategy_name,
        f"top_{len(symbols)}"
    ]

    # Create metadata with ranking information
    metadata = {
        "ranking_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ranking_period": f"{start_date} to {end_date}",
        "ranking_metric": "sortino_ratio",
        "num_symbols": len(symbols),
        "avg_sortino": np.mean([m.get('sortino_ratio', 0) for _, m in top_symbols]),
        "avg_sharpe": np.mean([m.get('sharpe_ratio', 0) for _, m in top_symbols]),
        "avg_return": np.mean([m.get('total_return', 0) for _, m in top_symbols]),
        "top_symbol": top_symbols[0][0] if top_symbols else None,
        "top_sortino": top_symbols[0][1].get('sortino_ratio', 0) if top_symbols else 0
    }

    # Create description
    description = (
        f"Top {len(symbols)} symbols ranked by Sortino ratio using {strategy_name} strategy. "
        f"Ranked on {metadata['ranking_date']} using data from {start_date} to {end_date}. "
        f"Average Sortino: {metadata['avg_sortino']:.3f}, Average Return: {metadata['avg_return']:.2%}"
    )

    # Create book
    book = create_book_from_session(
        name=book_name,
        strategy_module=strategy_module,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        symbols=symbols,
        tags=tags,
        description=description
    )

    # Add custom metadata
    book.metadata.update(metadata)

    # Save book
    manager = BookManager()
    manager.save_book(book)

    print(f"\n{'='*80}")
    print(f"BOOK SAVED: {book_name}")
    print(f"{'='*80}")
    print(f"Strategy:      {strategy_name}")
    print(f"Symbols:       {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"Avg Sortino:   {metadata['avg_sortino']:.3f}")
    print(f"Avg Return:    {metadata['avg_return']:.2%}")
    print(f"Location:      saved_books/{book_name}.json")
    print(f"Tags:          {', '.join(tags)}")
    print(f"\nThis book can now be loaded in Streamlit for backtesting!")
    print(f"{'='*80}\n")


def parse_correlation_threshold(threshold_str: str) -> float:
    """
    Parse correlation threshold from various formats

    Args:
        threshold_str: String like "80", "0.8", "80%", "correlation>80%"

    Returns:
        Float between 0 and 1
    """
    # Remove common prefix patterns
    threshold_str = threshold_str.lower().replace('correlation>', '').replace('corr>', '')

    # Remove percentage sign
    if '%' in threshold_str:
        threshold_str = threshold_str.replace('%', '')
        threshold = float(threshold_str) / 100.0
    else:
        threshold = float(threshold_str)

    # Convert to 0-1 range if given as percentage (e.g., "80" -> 0.8)
    if threshold > 1.0:
        threshold = threshold / 100.0

    # Validate
    if threshold < 0 or threshold > 1:
        raise ValueError(f"Correlation threshold must be between 0 and 1 (or 0-100%), got: {threshold}")

    return threshold


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Rank ETF symbols by strategy performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--strategy', required=True, choices=list(STRATEGY_MAP.keys()),
                       help='Strategy name')
    parser.add_argument('--top', type=int, default=20, help='Number of top performers (default: 20)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (default: 1 for reliability, use 2-3 for speed)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file for saving/resuming progress (optional)')
    parser.add_argument('--save', type=str, default=None, help='Save top symbols as a book with this name (optional)')
    parser.add_argument('--exclude', type=str, default=None, help='Exclude correlated symbols, e.g., "80" or "0.8" for 80%% correlation threshold (optional)')
    parser.add_argument('--exclude-high-risk', action='store_true',
                       help='Exclude high-risk symbols (volatility ETFs, leveraged/inverse ETFs) to prevent catastrophic losses')

    args = parser.parse_args()

    # Validate dates
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
        if start_date >= end_date:
            print("ERROR: Start date must be before end date")
            return
    except ValueError as e:
        print(f"ERROR: Invalid date format. Use YYYY-MM-DD. {e}")
        return

    # Get strategy function and parameters
    strategy_func = STRATEGY_MAP[args.strategy]
    strategy_params = DEFAULT_PARAMS.get(args.strategy, {})

    # Print configuration
    print("\n" + "="*80)
    print("SYMBOL RANKING BY STRATEGY PERFORMANCE")
    print("="*80)
    print(f"Strategy:       {args.strategy}")
    print(f"Period:         {args.start} to {args.end}")
    print(f"Initial Capital: ${args.capital:,.0f}")
    print(f"Top N:          {args.top}")
    print(f"Parameters:     {strategy_params}")
    print("="*80)

    # Get all symbols
    symbols = get_all_symbols(exclude_high_risk=args.exclude_high_risk)
    if args.exclude_high_risk:
        print(f"\nHigh-risk filter: ENABLED (excluding {len(HIGH_RISK_SYMBOLS)} symbols: {', '.join(sorted(HIGH_RISK_SYMBOLS))})")
    print(f"\nTotal symbols to test: {len(symbols)}")

    # Run backtests in parallel
    results = run_parallel_backtests(
        symbols=symbols,
        strategy_func=strategy_func,
        strategy_params=strategy_params,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        max_workers=args.workers,
        checkpoint_file=args.checkpoint
    )

    if not results:
        print("\nERROR: No successful backtests. Check date range and data availability.")
        return

    # Filter out invalid results before ranking
    # Remove symbols with:
    # 1. No trades (0 trades = no signal generated)
    # 2. Invalid metrics (inf, -inf, NaN for Sortino ratio)
    # 3. Zero or negative Sharpe ratio (likely data issues)
    valid_results = {}
    filtered_count = 0

    for symbol, metrics in results.items():
        # Check minimum trade requirement
        if metrics.get('total_trades', 0) == 0:
            filtered_count += 1
            continue

        # Check for valid Sortino ratio
        sortino = metrics.get('sortino_ratio', -np.inf)
        if not np.isfinite(sortino) or sortino <= 0:
            filtered_count += 1
            continue

        # Check for valid Sharpe ratio
        sharpe = metrics.get('sharpe_ratio', -np.inf)
        if not np.isfinite(sharpe):
            filtered_count += 1
            continue

        # Passed all filters
        valid_results[symbol] = metrics

    if not valid_results:
        print("\nERROR: No valid results after filtering. All symbols had 0 trades or invalid metrics.")
        print(f"Total symbols tested: {len(symbols)}")
        print(f"Backtests completed: {len(results)}")
        print(f"Filtered out (no trades or invalid metrics): {filtered_count}")
        return

    print(f"\nFiltered Results:")
    print(f"  Total symbols tested: {len(symbols)}")
    print(f"  Backtests completed: {len(results)}")
    print(f"  Valid results (with trades): {len(valid_results)}")
    print(f"  Filtered out (0 trades or invalid metrics): {filtered_count}")

    # Sort by Sortino ratio (descending)
    sorted_results = sorted(
        valid_results.items(),
        key=lambda x: x[1].get('sortino_ratio', -np.inf),
        reverse=True
    )

    # Apply correlation filtering with backfilling if requested
    excluded_pairs = []
    candidates_tested = 0
    if args.exclude:
        try:
            correlation_threshold = parse_correlation_threshold(args.exclude)
            print(f"\n{'='*80}")
            print(f"CORRELATION FILTERING WITH BACKFILL (Threshold: {correlation_threshold:.0%})")
            print(f"{'='*80}")
            print(f"Target symbols: {args.top}")
            print(f"Available candidates: {len(sorted_results)}")

            # Use backfilling algorithm - pass ALL sorted results
            top_symbols, excluded_pairs, candidates_tested = filter_correlated_symbols_with_backfill(
                sorted_results,
                args.top,
                correlation_threshold
            )

            print(f"Candidates tested: {candidates_tested} (ranks 1-{candidates_tested})")
            print(f"Symbols excluded: {len(excluded_pairs)}")
            print(f"Final portfolio: {len(top_symbols)} symbols")

            if len(top_symbols) < args.top:
                print(f"\n[WARNING] Could not reach target of {args.top} symbols - exhausted all candidates")

            if excluded_pairs:
                print(f"\nExcluded Symbols (by rank order):")
                for excluded, kept, corr in excluded_pairs:
                    print(f"  {excluded:<8} excluded (corr={corr:>6.1%} with {kept} already in portfolio)")
            else:
                print(f"\nNo symbols excluded - all tested symbols had correlation below threshold")

            print(f"{'='*80}\n")
        except ValueError as e:
            print(f"\nERROR: Invalid correlation threshold: {e}")
            return
    else:
        # No filtering - just get top N
        top_n = min(args.top, len(sorted_results))
        top_symbols = sorted_results[:top_n]

    # Create and print performance table
    table = create_performance_table(top_symbols, len(top_symbols))
    print(table)

    # Create and print correlation matrix
    corr_matrix = create_correlation_matrix(top_symbols)
    print(corr_matrix)

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total symbols tested: {len(symbols)}")
    print(f"Backtests completed: {len(results)}")
    print(f"Valid results (with trades): {len(valid_results)}")
    print(f"Filtered out (0 trades or invalid metrics): {filtered_count}")
    if args.exclude:
        print(f"Correlation filtering: {'enabled' if args.exclude else 'disabled'}")
        if candidates_tested > 0:
            print(f"  - Candidates evaluated: {candidates_tested} (ranks 1-{candidates_tested})")
            print(f"  - Symbols excluded: {len(excluded_pairs)}")
            print(f"  - Portfolio size: {len(top_symbols)} (target: {args.top})")
            if len(top_symbols) == args.top:
                print(f"  [OK] Target reached: {args.top} uncorrelated symbols")
            elif len(top_symbols) < args.top:
                print(f"  [WARNING] Partial fill: {len(top_symbols)}/{args.top} (exhausted candidates)")
    else:
        print(f"Top performers shown: {len(top_symbols)}")
    print("="*80 + "\n")

    # Save as book if requested
    if args.save:
        save_top_symbols_as_book(
            book_name=args.save,
            top_symbols=top_symbols,
            strategy_name=args.strategy,
            strategy_params=strategy_params,
            start_date=args.start,
            end_date=args.end
        )


if __name__ == "__main__":
    main()
