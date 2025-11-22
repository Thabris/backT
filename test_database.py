"""
Test Market Data Database

Quick test to verify database functionality before full download.
Tests with a small sample of symbols.

Usage:
    python test_database.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backt.data.market_data_db import MarketDataDB
from backt.data.sqlite_loader import SQLiteDataLoader
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time


def test_database_operations():
    """Test basic database operations"""
    print("\n" + "="*80)
    print("TESTING DATABASE OPERATIONS")
    print("="*80)

    # Create test database
    db_path = "test_market_data.db"
    db = MarketDataDB(db_path)

    print(f"\n1. Database created: {db_path}")

    # Test symbols
    test_symbols = ['SPY', 'QQQ', 'TLT']
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"\n2. Downloading test data for {test_symbols}")
    print(f"   Period: {start_date} to {end_date}")

    for symbol in test_symbols:
        print(f"\n   Downloading {symbol}...", end=" ")
        time.sleep(1)  # Rate limiting

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                print("FAILED - No data")
                continue

            # Standardize columns
            df.columns = [col.lower() for col in df.columns]
            if 'adj close' in df.columns:
                df['adj_close'] = df['adj close']
            else:
                df['adj_close'] = df['close']

            # Insert into database
            rows, warnings = db.insert_data(symbol, df, validate=True)

            print(f"OK - {rows} rows inserted")

            if warnings:
                print(f"   Warnings: {len(warnings)}")
                for w in warnings[:3]:
                    print(f"     - {w}")

        except Exception as e:
            print(f"FAILED - {e}")

    print("\n3. Verifying data retrieval...")

    for symbol in test_symbols:
        df = db.get_data(symbol, start_date, end_date)
        if not df.empty:
            print(f"   {symbol:8} - {len(df)} rows, "
                  f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            print(f"   {symbol:8} - NO DATA")

    print("\n4. Checking metadata...")

    for symbol in test_symbols:
        meta = db.get_metadata(symbol)
        if meta:
            print(f"   {symbol:8} - Quality: {meta['data_quality_score']:.1f}%, "
                  f"Records: {meta['total_records']}")

    print("\n5. Detecting gaps...")

    for symbol in test_symbols:
        gaps = db.detect_gaps(symbol)
        if gaps:
            print(f"   {symbol:8} - {len(gaps)} gaps detected")
            for start, end, days in gaps[:3]:
                print(f"             {start} to {end} ({days} days)")
        else:
            print(f"   {symbol:8} - No gaps")

    print("\n6. Testing SQLiteDataLoader...")

    loader = SQLiteDataLoader(db_path)
    try:
        data_dict = loader.load(test_symbols, start_date, end_date)
        print(f"   Loaded {len(data_dict)} symbols:")
        for symbol, df in data_dict.items():
            print(f"     {symbol:8} - {len(df)} rows")
    except Exception as e:
        print(f"   FAILED - {e}")

    loader.close()
    db.close()

    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print(f"\nTest database: {db_path}")
    print("\nIf all tests passed, you can proceed with full download:")
    print("  python download_market_data.py --years 15")
    print("\nTo delete test database:")
    print(f"  del {db_path}")


if __name__ == "__main__":
    test_database_operations()
