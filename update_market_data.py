"""
Update Market Data

Quick script to update the database with recent data.
Fetches only new data since last update for each symbol.

Usage:
    python update_market_data.py
    python update_market_data.py --db custom_db.db
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from download_market_data import main as download_main


def main():
    """Run update"""
    parser = argparse.ArgumentParser(description='Update market data database')
    parser.add_argument('--db', type=str, default='market_data.db',
                       help='Database file path (default: market_data.db)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')

    args = parser.parse_args()

    # Run in update mode
    sys.argv = [
        'download_market_data.py',
        '--update',
        '--db', args.db,
        '--delay', str(args.delay)
    ]

    download_main()


if __name__ == "__main__":
    main()
