"""
Test ETF Data Availability from Yahoo Finance

Tests all symbols in the ETF universe to identify which ones fail to load data.
This helps clean up the symbol list by removing delisted or unavailable ETFs.

Usage:
    python test_etf_data_availability.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from datetime import datetime, timedelta
from backt.data.loaders import YahooDataLoader
import warnings
warnings.filterwarnings('ignore')


# ETF Universe - same as in rank_symbols_by_strategy.py
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


def test_symbol(symbol: str, loader: YahooDataLoader, start_date: str, end_date: str) -> tuple:
    """
    Test if a symbol can be loaded from Yahoo Finance

    Returns:
        (symbol, success: bool, error_message: str or None, data_points: int)
    """
    try:
        # Load data - loader returns dict {symbol: DataFrame}
        data_dict = loader.load(symbol, start_date, end_date)

        # Extract DataFrame from dict
        if isinstance(data_dict, dict):
            if symbol not in data_dict:
                return (symbol, False, "Symbol not in returned data", 0)
            data = data_dict[symbol]
        else:
            data = data_dict

        # Check if we got valid data
        if data is None:
            return (symbol, False, "No data returned (None)", 0)

        if hasattr(data, 'empty') and data.empty:
            return (symbol, False, "No data returned (empty DataFrame)", 0)

        # Check if we have OHLC columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return (symbol, False, f"Missing columns: {missing_cols}", len(data))

        # Check if we have enough data points (at least 10 days)
        if len(data) < 10:
            return (symbol, False, f"Insufficient data: only {len(data)} days", len(data))

        # Success!
        return (symbol, True, None, len(data))

    except Exception as e:
        error_msg = str(e)
        # Simplify common error messages
        if "No data" in error_msg or "not found" in error_msg.lower():
            error_msg = "Symbol not found"
        elif "Invalid" in error_msg:
            error_msg = "Invalid data format"
        return (symbol, False, error_msg, 0)


def main():
    """Main test execution"""
    print("\n" + "="*80)
    print("ETF DATA AVAILABILITY TEST")
    print("="*80)

    # Test period - recent 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    print(f"Test Period: {start_str} to {end_str}")
    print(f"Testing Yahoo Finance data availability for all ETF symbols...")
    print("="*80 + "\n")

    # Collect all symbols
    all_symbols = []
    symbol_to_category = {}
    symbol_to_description = {}

    for category, etfs in ETF_UNIVERSE.items():
        for symbol, description in etfs.items():
            all_symbols.append(symbol)
            symbol_to_category[symbol] = category
            symbol_to_description[symbol] = description

    all_symbols = sorted(all_symbols)
    total = len(all_symbols)

    print(f"Total symbols to test: {total}\n")

    # Create data loader
    loader = YahooDataLoader()

    # Test each symbol
    success_symbols = []
    failed_symbols = []

    for i, symbol in enumerate(all_symbols, 1):
        symbol_result, success, error, data_points = test_symbol(symbol, loader, start_str, end_str)

        if success:
            success_symbols.append((symbol, data_points))
            print(f"[{i:3}/{total}] {symbol:8} [OK] SUCCESS ({data_points} days)")
        else:
            failed_symbols.append((symbol, error))
            print(f"[{i:3}/{total}] {symbol:8} [X] FAILED - {error}")

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Total symbols tested: {total}")
    print(f"Successful:           {len(success_symbols)} ({len(success_symbols)/total*100:.1f}%)")
    print(f"Failed:               {len(failed_symbols)} ({len(failed_symbols)/total*100:.1f}%)")

    # Failed symbols by category
    if failed_symbols:
        print("\n" + "="*80)
        print("FAILED SYMBOLS BY CATEGORY")
        print("="*80)

        # Group by category
        failed_by_category = {}
        for symbol, error in failed_symbols:
            category = symbol_to_category[symbol]
            if category not in failed_by_category:
                failed_by_category[category] = []
            failed_by_category[category].append((symbol, error))

        for category in sorted(failed_by_category.keys()):
            symbols = failed_by_category[category]
            print(f"\n{category} ({len(symbols)} failed):")
            for symbol, error in symbols:
                description = symbol_to_description[symbol]
                print(f"  {symbol:8} - {description:40} [{error}]")

        # List for easy removal
        print("\n" + "="*80)
        print("SYMBOLS TO REMOVE (Python list format)")
        print("="*80)
        failed_symbol_list = [s for s, _ in failed_symbols]
        print("failed_symbols = [")
        for i in range(0, len(failed_symbol_list), 10):
            chunk = failed_symbol_list[i:i+10]
            print("    " + ", ".join([f'"{s}"' for s in chunk]) + ",")
        print("]")
        print(f"\nTotal to remove: {len(failed_symbol_list)} symbols")

        # Statistics by error type
        print("\n" + "="*80)
        print("FAILURE STATISTICS BY ERROR TYPE")
        print("="*80)

        error_counts = {}
        for symbol, error in failed_symbols:
            if error not in error_counts:
                error_counts[error] = []
            error_counts[error].append(symbol)

        for error in sorted(error_counts.keys(), key=lambda x: len(error_counts[x]), reverse=True):
            symbols = error_counts[error]
            print(f"{error:40} - {len(symbols):3} symbols")
            if len(symbols) <= 5:
                print(f"  Symbols: {', '.join(symbols)}")

    else:
        print("\nAll symbols loaded successfully! [OK]")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
