"""
SQLite Data Loader for Backtester

Loads market data from local SQLite database instead of Yahoo Finance API.
Provides the same interface as YahooDataLoader for drop-in replacement.

Advantages:
- No rate limiting
- Instant data loading
- Offline capability
- Consistent data across backtests
"""

import pandas as pd
from typing import Union, List, Dict
from backt.data.market_data_db import MarketDataDB
from backt.data.loaders import DataLoader
import logging

logger = logging.getLogger(__name__)


class SQLiteDataLoader(DataLoader):
    """
    Load market data from local SQLite database

    Compatible with backtester's DataLoader interface
    """

    def __init__(self, db_path: str = "market_data.db"):
        """
        Initialize SQLite data loader

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__()
        self.db = MarketDataDB(db_path)
        logger.info(f"SQLite data loader initialized: {db_path}")

    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV data from SQLite database

        Args:
            symbols: Single symbol string or list of symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        # Handle single symbol
        if isinstance(symbols, str):
            symbols = [symbols]

        result = {}

        for symbol in symbols:
            try:
                df = self.db.get_data(symbol, start_date, end_date)

                if df.empty:
                    logger.warning(f"No data found for {symbol} in date range {start_date} to {end_date}")
                    continue

                # Ensure proper column names (lowercase)
                df.columns = [col.lower() for col in df.columns]

                # Ensure timezone-aware index
                if df.index.tz is None:
                    df.index = df.index.tz_localize('America/New_York')

                result[symbol] = df

            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue

        if not result:
            raise ValueError(f"No data returned for symbols: {symbols}")

        return result

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols available in database

        Returns:
            List of symbol strings
        """
        return self.db.get_symbols()

    def get_metadata(self, symbol: str) -> Dict:
        """
        Get metadata for a symbol

        Returns:
            Dict with symbol metadata
        """
        return self.db.get_metadata(symbol) or {}

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate market data from database

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Import required columns from constants
            from backt.utils.constants import REQUIRED_COLUMNS

            # Check required columns
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                logger.warning(f"Missing required columns. Has: {list(data.columns)}, Needs: {REQUIRED_COLUMNS}")
                return False

            # Check for reasonable price values
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (data[col] <= 0).any():
                    logger.warning(f"Non-positive values found in {col}")
                    return False

            # Check high >= low
            if (data['high'] < data['low']).any():
                logger.warning("High < Low found in data")
                return False

            # Check volume is non-negative
            if (data['volume'] < 0).any():
                logger.warning("Negative volume found in data")
                return False

            return True

        except Exception as e:
            logger.error(f"Data validation error: {e}")
            return False

    def close(self):
        """Close database connection"""
        self.db.close()
