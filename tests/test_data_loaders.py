"""
Tests for data loading module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from backt.data.loaders import YahooDataLoader, CSVDataLoader, DataLoader
from backt.utils.constants import REQUIRED_COLUMNS


class TestDataLoader:
    """Test base DataLoader functionality"""

    def test_ensure_required_columns(self):
        """Test ensuring required columns exist"""
        # Create test data with missing columns
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.1, 2.1, 3.1]
        })

        loader = YahooDataLoader()

        with pytest.raises(ValueError, match="Missing required columns"):
            loader._ensure_required_columns(data)

    def test_ensure_datetime_index(self):
        """Test ensuring datetime index"""
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'high': [1.2, 2.2, 3.2],
            'low': [0.9, 1.9, 2.9],
            'close': [1.1, 2.1, 3.1],
            'volume': [100, 200, 300]
        }, index=['2020-01-01', '2020-01-02', '2020-01-03'])

        loader = YahooDataLoader()
        result = loader._ensure_datetime_index(data)

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None  # Should be timezone-aware

    def test_resample_data(self):
        """Test data resampling"""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100 + 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.randint(1000, 10000, 10)
        }, index=dates)

        loader = YahooDataLoader()
        resampled = loader.resample_data(data, '2D')

        assert len(resampled) == 5  # 10 days resampled to 2-day periods
        assert all(col in resampled.columns for col in REQUIRED_COLUMNS)


class TestYahooDataLoader:
    """Test Yahoo Finance data loader"""

    def test_validate_data_valid(self):
        """Test data validation with valid data"""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

        loader = YahooDataLoader()
        assert loader.validate_data(data) is True

    def test_validate_data_invalid_prices(self):
        """Test data validation with invalid prices"""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [100, 101, -102, 103, 104],  # Negative price
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

        loader = YahooDataLoader()
        assert loader.validate_data(data) is False

    def test_validate_data_high_low_inconsistent(self):
        """Test data validation with high < low"""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [95, 106, 107, 108, 109],  # High < Low for first row
            'low': [105, 96, 97, 98, 99],
            'close': [103, 104, 105, 106, 107],
            'volume': [1000, 1100, 1200, 1300, 1400]
        }, index=dates)

        loader = YahooDataLoader()
        assert loader.validate_data(data) is False

    @patch('yfinance.download')
    def test_load_single_symbol(self, mock_download):
        """Test loading single symbol"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2020-01-01', periods=3))

        mock_download.return_value = mock_data

        loader = YahooDataLoader()
        result = loader.load('AAPL', '2020-01-01', '2020-01-03')

        assert isinstance(result, pd.DataFrame)
        assert all(col in result.columns for col in REQUIRED_COLUMNS)


class TestCSVDataLoader:
    """Test CSV data loader"""

    def test_csv_loader_initialization(self):
        """Test CSV loader initialization"""
        loader = CSVDataLoader(
            date_column='Date',
            symbol_column='Symbol'
        )

        assert loader.date_column == 'Date'
        assert loader.symbol_column == 'Symbol'

    def test_validate_data_missing_columns(self):
        """Test CSV validation with missing columns"""
        data = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.1, 2.1, 3.1]
            # Missing high, low, volume
        })

        loader = CSVDataLoader()
        assert loader.validate_data(data) is False

    def test_validate_data_non_numeric(self):
        """Test CSV validation with non-numeric data"""
        data = pd.DataFrame({
            'open': ['a', 'b', 'c'],  # Non-numeric
            'high': [1.2, 2.2, 3.2],
            'low': [0.9, 1.9, 2.9],
            'close': [1.1, 2.1, 3.1],
            'volume': [100, 200, 300]
        })

        loader = CSVDataLoader()
        assert loader.validate_data(data) is False