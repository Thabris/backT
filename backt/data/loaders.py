"""
Data loading interfaces and implementations for BackT

Provides standardized data loading from various sources including
Yahoo Finance, CSV files, and custom data sources.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable
import pandas as pd
from pathlib import Path

from ..utils.types import TimeSeriesData
from ..utils.constants import REQUIRED_COLUMNS, DEFAULT_TIMEZONE
from ..utils.logging_config import LoggerMixin


class DataLoader(ABC, LoggerMixin):
    """Abstract base class for data loaders"""

    @abstractmethod
    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
        """
        Load data for the specified symbols and date range

        Args:
            symbols: Single symbol or list of symbols to load
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters specific to the loader

        Returns:
            DataFrame for single symbol or dict of DataFrames for multiple symbols
        """
        pass

    @abstractmethod
    def validate_data(self, data: TimeSeriesData) -> bool:
        """
        Validate that data meets the required format

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        pass

    def _ensure_required_columns(self, data: TimeSeriesData) -> TimeSeriesData:
        """Ensure data has required OHLCV columns"""
        missing_cols = set(REQUIRED_COLUMNS) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        return data

    def _ensure_datetime_index(self, data: TimeSeriesData) -> TimeSeriesData:
        """Ensure data has datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {e}")

        # Ensure timezone awareness
        if data.index.tz is None:
            data.index = data.index.tz_localize(DEFAULT_TIMEZONE)

        return data

    def resample_data(
        self,
        data: TimeSeriesData,
        frequency: str
    ) -> TimeSeriesData:
        """
        Resample data to specified frequency

        Args:
            data: Input data
            frequency: Target frequency (pandas frequency string)

        Returns:
            Resampled data
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have datetime index for resampling")

        # OHLCV resampling rules
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Only resample columns that exist and have rules
        available_rules = {
            col: rule for col, rule in agg_rules.items()
            if col in data.columns
        }

        # Handle additional columns
        for col in data.columns:
            if col not in available_rules:
                if col in ['adj_close', 'vwap']:
                    available_rules[col] = 'last'
                elif 'price' in col.lower():
                    available_rules[col] = 'last'
                else:
                    available_rules[col] = 'last'  # default to last

        resampled = data.resample(frequency).agg(available_rules)

        # Remove rows with NaN values (gaps in data)
        resampled = resampled.dropna()

        self.logger.info(f"Resampled data from {len(data)} to {len(resampled)} rows")
        return resampled


class YahooDataLoader(DataLoader):
    """Data loader for Yahoo Finance data"""

    def __init__(self, auto_adjust: bool = True, timeout: int = 30):
        self.auto_adjust = auto_adjust
        self.timeout = timeout

    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
        """Load data from Yahoo Finance"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for Yahoo data loading. "
                "Install with: pip install yfinance"
            )

        single_symbol = isinstance(symbols, str)
        if single_symbol:
            symbols = [symbols]

        self.logger.info(f"Loading Yahoo Finance data for {symbols}")

        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=self.auto_adjust,
                timeout=self.timeout,
                **kwargs
            )

            if data.empty:
                raise ValueError(f"No data returned for symbols: {symbols}")

            # Handle single vs multiple symbols
            if len(symbols) == 1:
                # Single symbol - flatten column names
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0].lower() for col in data.columns]
                else:
                    data.columns = [col.lower() for col in data.columns]

                data = self._ensure_datetime_index(data)
                data = self._ensure_required_columns(data)

                if not self.validate_data(data):
                    raise ValueError(f"Invalid data format for {symbols[0]}")

                return data if not single_symbol else data

            else:
                # Multiple symbols - return dict
                result = {}
                for symbol in symbols:
                    try:
                        symbol_data = data.xs(symbol, level=1, axis=1)
                        symbol_data.columns = [col.lower() for col in symbol_data.columns]
                        symbol_data = self._ensure_datetime_index(symbol_data)
                        symbol_data = self._ensure_required_columns(symbol_data)

                        if self.validate_data(symbol_data):
                            result[symbol] = symbol_data
                        else:
                            self.logger.warning(f"Invalid data for {symbol}, skipping")

                    except Exception as e:
                        self.logger.error(f"Error processing {symbol}: {e}")

                return result

        except Exception as e:
            self.logger.error(f"Error loading Yahoo Finance data: {e}")
            raise

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate Yahoo Finance data"""
        try:
            # Check required columns
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                return False

            # Check for reasonable price values
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (data[col] <= 0).any():
                    self.logger.warning(f"Non-positive values found in {col}")
                    return False

            # Check high >= low
            if (data['high'] < data['low']).any():
                self.logger.warning("High < Low found in data")
                return False

            # Check volume is non-negative
            if (data['volume'] < 0).any():
                self.logger.warning("Negative volume found in data")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False


class CSVDataLoader(DataLoader):
    """Data loader for CSV files"""

    def __init__(
        self,
        date_column: str = "date",
        symbol_column: Optional[str] = None,
        **csv_kwargs
    ):
        self.date_column = date_column
        self.symbol_column = symbol_column
        self.csv_kwargs = csv_kwargs

    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        file_path: Optional[str] = None,
        **kwargs
    ) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
        """
        Load data from CSV file(s)

        Args:
            symbols: Symbol(s) to load
            start_date: Start date
            end_date: End date
            file_path: Path to CSV file (for single symbol) or directory (for multiple)
            **kwargs: Additional parameters
        """
        if file_path is None:
            raise ValueError("file_path must be provided for CSV loading")

        path = Path(file_path)
        single_symbol = isinstance(symbols, str)

        if single_symbol:
            symbols = [symbols]

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if path.is_file():
            # Single file
            data = self._load_single_csv(path, symbols, start_dt, end_dt)
            return data if not single_symbol else data[symbols[0]]

        elif path.is_dir():
            # Directory with multiple files
            result = {}
            for symbol in symbols:
                symbol_file = path / f"{symbol}.csv"
                if symbol_file.exists():
                    try:
                        symbol_data = self._load_single_csv(
                            symbol_file, [symbol], start_dt, end_dt
                        )[symbol]
                        result[symbol] = symbol_data
                    except Exception as e:
                        self.logger.error(f"Error loading {symbol}: {e}")
                else:
                    self.logger.warning(f"File not found for {symbol}: {symbol_file}")

            return result

        else:
            raise ValueError(f"Invalid file path: {file_path}")

    def _load_single_csv(
        self,
        file_path: Path,
        symbols: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict[str, TimeSeriesData]:
        """Load data from a single CSV file"""
        try:
            data = pd.read_csv(file_path, **self.csv_kwargs)

            # Set date index
            if self.date_column in data.columns:
                data[self.date_column] = pd.to_datetime(data[self.date_column])
                data.set_index(self.date_column, inplace=True)
            else:
                # Assume first column is date
                data.index = pd.to_datetime(data.index)

            # Ensure datetime index
            data = self._ensure_datetime_index(data)

            # Filter by date range
            data = data[(data.index >= start_date) & (data.index <= end_date)]

            # Handle multiple symbols in one file
            if self.symbol_column and self.symbol_column in data.columns:
                result = {}
                for symbol in symbols:
                    symbol_data = data[data[self.symbol_column] == symbol].copy()
                    symbol_data = symbol_data.drop(columns=[self.symbol_column])

                    # Standardize column names
                    symbol_data.columns = [col.lower() for col in symbol_data.columns]
                    symbol_data = self._ensure_required_columns(symbol_data)

                    if self.validate_data(symbol_data):
                        result[symbol] = symbol_data

                return result

            else:
                # Single symbol file
                data.columns = [col.lower() for col in data.columns]
                data = self._ensure_required_columns(data)

                if self.validate_data(data):
                    return {symbols[0]: data}
                else:
                    raise ValueError("Data validation failed")

        except Exception as e:
            self.logger.error(f"Error loading CSV {file_path}: {e}")
            raise

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Validate CSV data format"""
        try:
            # Check required columns
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                missing = set(REQUIRED_COLUMNS) - set(data.columns)
                self.logger.error(f"Missing required columns: {missing}")
                return False

            # Check for numeric data
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    self.logger.error(f"Column {col} is not numeric")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"CSV validation error: {e}")
            return False


class CustomDataLoader(DataLoader):
    """Custom data loader that accepts user-provided function"""

    def __init__(self, load_function: Callable):
        self.load_function = load_function

    def load(
        self,
        symbols: Union[str, List[str]],
        start_date: str,
        end_date: str,
        **kwargs
    ) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
        """Load data using custom function"""
        try:
            data = self.load_function(symbols, start_date, end_date, **kwargs)

            # Validate and process the returned data
            if isinstance(data, pd.DataFrame):
                data = self._ensure_datetime_index(data)
                data = self._ensure_required_columns(data)
                if not self.validate_data(data):
                    raise ValueError("Custom data validation failed")
                return data

            elif isinstance(data, dict):
                processed_data = {}
                for symbol, symbol_data in data.items():
                    symbol_data = self._ensure_datetime_index(symbol_data)
                    symbol_data = self._ensure_required_columns(symbol_data)
                    if self.validate_data(symbol_data):
                        processed_data[symbol] = symbol_data
                    else:
                        self.logger.warning(f"Validation failed for {symbol}")
                return processed_data

            else:
                raise ValueError("Custom loader must return DataFrame or dict of DataFrames")

        except Exception as e:
            self.logger.error(f"Custom data loading error: {e}")
            raise

    def validate_data(self, data: TimeSeriesData) -> bool:
        """Basic validation for custom data"""
        try:
            # Check required columns
            if not all(col in data.columns for col in REQUIRED_COLUMNS):
                return False

            # Check datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                return False

            return True

        except Exception:
            return False