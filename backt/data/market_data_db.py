"""
SQLite Database for Market Data Storage

Provides efficient local caching of OHLCV data for ETF symbols with:
- Optimized time-series indexing
- Data consistency validation
- Gap detection and reporting
- Incremental updates

Schema:
    market_data: Core OHLCV data table
    metadata: Symbol information and update tracking
    data_quality: Quality metrics and gap tracking
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketDataDB:
    """SQLite database manager for market data"""

    def __init__(self, db_path: str = "market_data.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = None
        self._connect()
        self._create_schema()

    def _connect(self):
        """Establish database connection with optimizations"""
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode for performance
        )

        # Performance optimizations
        self.conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
        self.conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
        self.conn.execute("PRAGMA cache_size=10000")  # 10MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables

        logger.info(f"Connected to database: {self.db_path}")

    def _create_schema(self):
        """Create database schema with optimized indexes"""
        cursor = self.conn.cursor()

        # Main market data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                adj_close REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)

        # Critical indexes for time-series queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol_date
            ON market_data(symbol, date DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_date
            ON market_data(date DESC)
        """)

        # Metadata table for tracking symbols
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                first_date DATE,
                last_date DATE,
                total_records INTEGER,
                last_updated TIMESTAMP,
                data_quality_score REAL,
                notes TEXT
            )
        """)

        # Data quality tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                check_date DATE NOT NULL,
                missing_days INTEGER,
                gap_count INTEGER,
                largest_gap_days INTEGER,
                ohlc_violations INTEGER,
                volume_zeros INTEGER,
                quality_score REAL,
                FOREIGN KEY (symbol) REFERENCES metadata(symbol)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_symbol
            ON data_quality(symbol, check_date DESC)
        """)

        self.conn.commit()
        logger.info("Database schema created/verified")

    def insert_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        validate: bool = True
    ) -> Tuple[int, List[str]]:
        """
        Insert OHLCV data for a symbol with validation

        Args:
            symbol: Ticker symbol
            data: DataFrame with OHLCV data (columns: open, high, low, close, volume, adj_close)
            validate: Whether to validate data consistency

        Returns:
            Tuple of (rows_inserted, list_of_warnings)
        """
        warnings = []

        if data.empty:
            return 0, ["No data to insert"]

        # Validate OHLCV relationships
        if validate:
            validation_warnings = self._validate_ohlcv(data)
            warnings.extend(validation_warnings)

        # Prepare data for insertion
        data = data.copy()
        data['symbol'] = symbol
        data = data.reset_index()

        # Ensure date column is properly named
        if 'date' not in data.columns:
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
            else:
                data['date'] = data.index

        # Convert to required format
        data['date'] = pd.to_datetime(data['date']).dt.date

        # Select only required columns
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return 0, [f"Missing required columns: {missing_cols}"]

        insert_data = data[required_cols]

        # Insert with conflict handling (replace on duplicate)
        cursor = self.conn.cursor()
        rows_inserted = 0

        for _, row in insert_data.iterrows():
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO market_data
                    (symbol, date, open, high, low, close, volume, adj_close)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['symbol'], row['date'],
                    float(row['open']), float(row['high']),
                    float(row['low']), float(row['close']),
                    int(row['volume']), float(row['adj_close'])
                ))
                rows_inserted += 1
            except Exception as e:
                warnings.append(f"Error inserting row for {row['date']}: {e}")

        self.conn.commit()

        # Update metadata
        self._update_metadata(symbol)

        logger.info(f"Inserted {rows_inserted} rows for {symbol}")
        return rows_inserted, warnings

    def _validate_ohlcv(self, data: pd.DataFrame) -> List[str]:
        """
        Validate OHLCV data consistency

        Returns:
            List of warning messages
        """
        warnings = []

        # Check: High >= Low
        invalid_hl = data[data['high'] < data['low']]
        if len(invalid_hl) > 0:
            warnings.append(f"OHLC violation: {len(invalid_hl)} days where High < Low")

        # Check: High >= Open, Close
        invalid_h = data[(data['high'] < data['open']) | (data['high'] < data['close'])]
        if len(invalid_h) > 0:
            warnings.append(f"OHLC violation: {len(invalid_h)} days where High < Open or Close")

        # Check: Low <= Open, Close
        invalid_l = data[(data['low'] > data['open']) | (data['low'] > data['close'])]
        if len(invalid_l) > 0:
            warnings.append(f"OHLC violation: {len(invalid_l)} days where Low > Open or Close")

        # Check: Volume non-negative
        invalid_vol = data[data['volume'] < 0]
        if len(invalid_vol) > 0:
            warnings.append(f"Volume violation: {len(invalid_vol)} days with negative volume")

        # Check: Zero volumes (warning, not error)
        zero_vol = data[data['volume'] == 0]
        if len(zero_vol) > 0:
            warnings.append(f"Data quality: {len(zero_vol)} days with zero volume")

        return warnings

    def _update_metadata(self, symbol: str):
        """Update metadata for a symbol"""
        cursor = self.conn.cursor()

        # Get statistics
        cursor.execute("""
            SELECT
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as total_records
            FROM market_data
            WHERE symbol = ?
        """, (symbol,))

        result = cursor.fetchone()
        if result:
            first_date, last_date, total_records = result

            # Calculate quality score
            quality_score = self._calculate_quality_score(symbol)

            cursor.execute("""
                INSERT OR REPLACE INTO metadata
                (symbol, first_date, last_date, total_records, last_updated, data_quality_score)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """, (symbol, first_date, last_date, total_records, quality_score))

            self.conn.commit()

    def _calculate_quality_score(self, symbol: str) -> float:
        """
        Calculate data quality score (0-100)

        Factors:
        - Data completeness (gaps)
        - OHLCV consistency
        - Volume data availability
        """
        cursor = self.conn.cursor()

        # Get date range and count
        cursor.execute("""
            SELECT
                MIN(date) as first_date,
                MAX(date) as last_date,
                COUNT(*) as actual_days
            FROM market_data
            WHERE symbol = ?
        """, (symbol,))

        result = cursor.fetchone()
        if not result or not result[0]:
            return 0.0

        first_date, last_date, actual_days = result
        first_date = datetime.strptime(first_date, '%Y-%m-%d')
        last_date = datetime.strptime(last_date, '%Y-%m-%d')

        # Expected trading days (assume ~252 per year)
        calendar_days = (last_date - first_date).days
        expected_days = calendar_days * 252 / 365  # Rough estimate

        completeness_score = min(100, (actual_days / expected_days * 100)) if expected_days > 0 else 0

        return completeness_score

    def get_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV data for a symbol

        Args:
            symbol: Ticker symbol
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all

        Returns:
            DataFrame with OHLCV data indexed by date
        """
        query = """
            SELECT date, open, high, low, close, volume, adj_close
            FROM market_data
            WHERE symbol = ?
        """
        params = [symbol]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date ASC"

        df = pd.read_sql_query(query, self.conn, params=params)

        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

        return df

    def get_symbols(self) -> List[str]:
        """Get list of all symbols in database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT symbol FROM metadata ORDER BY symbol")
        return [row[0] for row in cursor.fetchall()]

    def get_metadata(self, symbol: str) -> Optional[Dict]:
        """Get metadata for a symbol"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT symbol, name, first_date, last_date, total_records,
                   last_updated, data_quality_score, notes
            FROM metadata
            WHERE symbol = ?
        """, (symbol,))

        row = cursor.fetchone()
        if row:
            return {
                'symbol': row[0],
                'name': row[1],
                'first_date': row[2],
                'last_date': row[3],
                'total_records': row[4],
                'last_updated': row[5],
                'data_quality_score': row[6],
                'notes': row[7]
            }
        return None

    def detect_gaps(self, symbol: str) -> List[Tuple[str, str, int]]:
        """
        Detect gaps in data (missing trading days)

        Returns:
            List of (gap_start, gap_end, days_missing) tuples
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT date
            FROM market_data
            WHERE symbol = ?
            ORDER BY date ASC
        """, (symbol,))

        dates = [datetime.strptime(row[0], '%Y-%m-%d') for row in cursor.fetchall()]

        if len(dates) < 2:
            return []

        gaps = []
        for i in range(len(dates) - 1):
            day_diff = (dates[i + 1] - dates[i]).days
            # Gap if more than 3 calendar days (allows for weekends)
            if day_diff > 3:
                # Estimate trading days missed (rough)
                trading_days_missed = int(day_diff * 252 / 365) - 1
                if trading_days_missed > 0:
                    gaps.append((
                        dates[i].strftime('%Y-%m-%d'),
                        dates[i + 1].strftime('%Y-%m-%d'),
                        trading_days_missed
                    ))

        return gaps

    def get_last_date(self, symbol: str) -> Optional[datetime]:
        """Get the last available date for a symbol"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT MAX(date)
            FROM market_data
            WHERE symbol = ?
        """, (symbol,))

        result = cursor.fetchone()
        if result and result[0]:
            return datetime.strptime(result[0], '%Y-%m-%d')
        return None

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
