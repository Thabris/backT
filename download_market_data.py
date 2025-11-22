"""
Bulk Market Data Download

Downloads historical OHLCV data for all ETF symbols with:
- Conservative rate limiting to avoid Yahoo Finance blocks
- Comprehensive data validation
- Gap detection and reporting
- Progress checkpointing
- Detailed quality reports

Usage:
    python download_market_data.py --years 15
    python download_market_data.py --start 2010-01-01 --end 2025-12-31
    python download_market_data.py --update  # Update only (fetch recent data)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import yfinance as yf
from backt.data.market_data_db import MarketDataDB
import warnings
warnings.filterwarnings('ignore')


# ETF Universe
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


def get_all_symbols() -> List[Tuple[str, str, str]]:
    """
    Get all symbols with metadata

    Returns:
        List of (symbol, name, category) tuples
    """
    symbols = []
    for category, etfs in ETF_UNIVERSE.items():
        for symbol, name in etfs.items():
            symbols.append((symbol, name, category))
    return sorted(symbols, key=lambda x: x[0])


def download_symbol(
    symbol: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    delay: float = 1.0
) -> Tuple[str, pd.DataFrame, List[str]]:
    """
    Download data for a single symbol with retry logic

    Args:
        symbol: Ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_retries: Maximum retry attempts
        delay: Base delay in seconds

    Returns:
        Tuple of (symbol, dataframe, warnings_list)
    """
    warnings = []

    for attempt in range(max_retries):
        try:
            # Rate limiting delay (ALWAYS, not just on retry)
            time.sleep(delay)

            # Additional backoff on retry
            if attempt > 0:
                backoff = 3 ** attempt  # 3, 9, 27 seconds
                time.sleep(backoff)
                warnings.append(f"Retry attempt {attempt + 1}/{max_retries}")

            # Download using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                warnings.append("No data returned")
                continue

            # Validate minimum data points
            if len(df) < 10:
                warnings.append(f"Insufficient data: only {len(df)} days")
                continue

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                warnings.append(f"Missing columns: {missing}")
                continue

            # Add adj_close if not present
            if 'adj close' in df.columns:
                df['adj_close'] = df['adj close']
            elif 'adj_close' not in df.columns:
                df['adj_close'] = df['close']

            # Success!
            return (symbol, df, warnings)

        except Exception as e:
            error_msg = str(e)
            if "No data" in error_msg or "not found" in error_msg.lower():
                warnings.append(f"Symbol not found: {error_msg}")
            else:
                warnings.append(f"Download error: {error_msg}")

    # Failed after all retries
    return (symbol, pd.DataFrame(), warnings)


def bulk_download(
    db: MarketDataDB,
    symbols: List[Tuple[str, str, str]],
    start_date: str,
    end_date: str,
    delay: float = 1.0
) -> Dict[str, Dict]:
    """
    Download data for all symbols with progress tracking

    Returns:
        Dict mapping symbol to result statistics
    """
    results = {}
    total = len(symbols)

    print("\n" + "=" * 80)
    print("BULK MARKET DATA DOWNLOAD")
    print("=" * 80)
    print(f"Symbols:    {total}")
    print(f"Period:     {start_date} to {end_date}")
    print(f"Delay:      {delay}s per request")
    print(f"Database:   {db.db_path}")
    print("=" * 80 + "\n")

    start_time = time.time()

    for i, (symbol, name, category) in enumerate(symbols, 1):
        symbol_start = time.time()

        print(f"[{i:3}/{total}] {symbol:8} ({category:30})", end=" ... ", flush=True)

        # Download
        _, df, warnings = download_symbol(symbol, start_date, end_date, delay=delay)

        if df.empty:
            print(f"FAILED - {'; '.join(warnings[:2])}")
            results[symbol] = {
                'success': False,
                'rows': 0,
                'warnings': warnings,
                'category': category,
                'name': name
            }
            continue

        # Insert into database
        rows_inserted, insert_warnings = db.insert_data(symbol, df, validate=True)

        # Combine warnings
        all_warnings = warnings + insert_warnings

        # Detect gaps
        gaps = db.detect_gaps(symbol)

        # Calculate quality
        quality_score = db._calculate_quality_score(symbol)

        symbol_time = time.time() - symbol_start

        # Print result
        status_parts = [f"{rows_inserted} days"]
        if gaps:
            status_parts.append(f"{len(gaps)} gaps")
        if all_warnings:
            status_parts.append(f"{len(all_warnings)} warnings")
        status_parts.append(f"Q:{quality_score:.1f}%")
        status_parts.append(f"[{symbol_time:.1f}s]")

        print(" ".join(status_parts))

        results[symbol] = {
            'success': True,
            'rows': rows_inserted,
            'warnings': all_warnings,
            'gaps': gaps,
            'quality_score': quality_score,
            'category': category,
            'name': name,
            'time': symbol_time
        }

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results.values() if r['success'])
    failed = total - successful
    total_rows = sum(r['rows'] for r in results.values())

    print(f"Total symbols:      {total}")
    print(f"Successful:         {successful} ({successful/total*100:.1f}%)")
    print(f"Failed:             {failed} ({failed/total*100:.1f}%)")
    print(f"Total data points:  {total_rows:,}")
    print(f"Total time:         {total_time/60:.1f} minutes")
    print(f"Average time/symbol: {total_time/total:.1f}s")

    return results


def print_quality_report(db: MarketDataDB, results: Dict[str, Dict]):
    """Print comprehensive data quality report"""

    print("\n" + "=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    # Symbols with gaps
    symbols_with_gaps = [(s, r) for s, r in results.items()
                         if r['success'] and r.get('gaps')]

    if symbols_with_gaps:
        print(f"\nSymbols with data gaps: {len(symbols_with_gaps)}")
        print("-" * 80)
        for symbol, result in symbols_with_gaps[:10]:  # Top 10
            gaps = result['gaps']
            print(f"{symbol:8} - {len(gaps)} gaps, largest: {max(g[2] for g in gaps)} days")

    # Symbols with warnings
    symbols_with_warnings = [(s, r) for s, r in results.items()
                             if r['success'] and r.get('warnings')]

    if symbols_with_warnings:
        print(f"\nSymbols with warnings: {len(symbols_with_warnings)}")
        print("-" * 80)
        for symbol, result in symbols_with_warnings[:10]:  # Top 10
            warnings = result['warnings']
            print(f"{symbol:8} - {'; '.join(warnings[:2])}")

    # Failed symbols
    failed = [(s, r) for s, r in results.items() if not r['success']]

    if failed:
        print(f"\nFailed downloads: {len(failed)}")
        print("-" * 80)
        for symbol, result in failed:
            print(f"{symbol:8} - {'; '.join(result['warnings'][:2])}")

    # Quality distribution
    quality_scores = [r['quality_score'] for r in results.values() if r['success']]
    if quality_scores:
        avg_quality = sum(quality_scores) / len(quality_scores)
        min_quality = min(quality_scores)
        max_quality = max(quality_scores)

        print(f"\nData Quality Scores:")
        print(f"  Average:  {avg_quality:.1f}%")
        print(f"  Range:    {min_quality:.1f}% - {max_quality:.1f}%")
        print(f"  Excellent (>95%): {sum(1 for q in quality_scores if q > 95)}")
        print(f"  Good (80-95%):    {sum(1 for q in quality_scores if 80 <= q <= 95)}")
        print(f"  Fair (<80%):      {sum(1 for q in quality_scores if q < 80)}")

    print("=" * 80)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Download market data to local database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--years', type=int, help='Download N years of data (alternative to --start)')
    parser.add_argument('--update', action='store_true', help='Update mode: fetch only recent data')
    parser.add_argument('--db', type=str, default='market_data.db', help='Database file path')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds, default: 1.0)')

    args = parser.parse_args()

    # Determine date range
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    if args.update:
        # Update mode: fetch recent data only
        print("UPDATE MODE: Will fetch data from last available date for each symbol")
        start_date = None  # Will be determined per-symbol
    elif args.years:
        start_date = (datetime.now() - timedelta(days=args.years * 365)).strftime('%Y-%m-%d')
    elif args.start:
        start_date = args.start
    else:
        # Default: 15 years
        start_date = (datetime.now() - timedelta(days=15 * 365)).strftime('%Y-%m-%d')

    # Initialize database
    db = MarketDataDB(args.db)

    # Get symbols
    symbols = get_all_symbols()

    if args.update:
        # Update mode: check last date for each symbol
        print("Checking existing data...")
        symbols_to_update = []

        for symbol, name, category in symbols:
            last_date = db.get_last_date(symbol)
            if last_date is None:
                # No data yet, download from start_date
                update_start = start_date if start_date else (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
                symbols_to_update.append((symbol, name, category, update_start))
            else:
                # Has data, fetch from last date + 1 day
                update_start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                if update_start <= end_date:
                    symbols_to_update.append((symbol, name, category, update_start))

        print(f"Need to update: {len(symbols_to_update)} symbols")

        # Download with per-symbol start dates
        results = {}
        total = len(symbols_to_update)

        for i, (symbol, name, category, symbol_start) in enumerate(symbols_to_update, 1):
            print(f"[{i:3}/{total}] {symbol:8} from {symbol_start}", end=" ... ", flush=True)

            _, df, warnings = download_symbol(symbol, symbol_start, end_date, delay=args.delay)

            if df.empty:
                print(f"FAILED - {'; '.join(warnings[:2])}")
                results[symbol] = {'success': False, 'rows': 0, 'warnings': warnings}
                continue

            rows_inserted, insert_warnings = db.insert_data(symbol, df, validate=True)
            print(f"{rows_inserted} new days")

            results[symbol] = {
                'success': True,
                'rows': rows_inserted,
                'warnings': warnings + insert_warnings,
                'gaps': db.detect_gaps(symbol),
                'quality_score': db._calculate_quality_score(symbol),
                'category': category,
                'name': name
            }

    else:
        # Bulk download mode
        results = bulk_download(db, symbols, start_date, end_date, delay=args.delay)

    # Print quality report
    print_quality_report(db, results)

    # Close database
    db.close()

    print(f"\nDatabase saved to: {args.db}")


if __name__ == "__main__":
    main()
