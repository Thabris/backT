"""
Data normalization utilities for BackT

Provides functions to clean, normalize, and standardize market data
from various sources into consistent formats.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from ..utils.types import TimeSeriesData
from ..utils.constants import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, DEFAULT_TIMEZONE
from ..utils.logging_config import LoggerMixin


class DataNormalizer(LoggerMixin):
    """Normalizes and cleans market data from various sources"""

    def __init__(
        self,
        remove_outliers: bool = True,
        outlier_method: str = "iqr",
        outlier_threshold: float = 3.0,
        fill_missing: bool = True,
        missing_method: str = "forward_fill"
    ):
        """
        Initialize data normalizer

        Args:
            remove_outliers: Whether to remove statistical outliers
            outlier_method: Method for outlier detection ('iqr', 'zscore')
            outlier_threshold: Threshold for outlier detection
            fill_missing: Whether to fill missing values
            missing_method: Method for filling missing values
        """
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.fill_missing = fill_missing
        self.missing_method = missing_method

    def normalize(
        self,
        data: Union[TimeSeriesData, Dict[str, TimeSeriesData]]
    ) -> Union[TimeSeriesData, Dict[str, TimeSeriesData]]:
        """
        Normalize market data

        Args:
            data: Single DataFrame or dict of DataFrames to normalize

        Returns:
            Normalized data in same format as input
        """
        if isinstance(data, pd.DataFrame):
            return self._normalize_single(data)
        elif isinstance(data, dict):
            return {symbol: self._normalize_single(df) for symbol, df in data.items()}
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")

    def _normalize_single(self, data: TimeSeriesData) -> TimeSeriesData:
        """Normalize a single DataFrame"""
        self.logger.info(f"Normalizing data with {len(data)} rows")

        # Create a copy to avoid modifying original
        normalized = data.copy()

        # Ensure proper datetime index
        normalized = self._ensure_datetime_index(normalized)

        # Standardize column names
        normalized = self._standardize_columns(normalized)

        # Remove duplicates
        normalized = self._remove_duplicates(normalized)

        # Sort by date
        normalized = normalized.sort_index()

        # Validate OHLC relationships
        normalized = self._validate_ohlc(normalized)

        # Handle missing values
        if self.fill_missing:
            normalized = self._fill_missing_values(normalized)

        # Remove outliers
        if self.remove_outliers:
            normalized = self._remove_outliers(normalized)

        # Final validation
        if not self._validate_normalized_data(normalized):
            raise ValueError("Data failed final validation after normalization")

        self.logger.info(f"Normalization complete: {len(normalized)} rows remaining")
        return normalized

    def _ensure_datetime_index(self, data: TimeSeriesData) -> TimeSeriesData:
        """Ensure data has proper datetime index"""
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                raise ValueError(f"Cannot convert index to datetime: {e}")

        # Ensure timezone awareness
        if data.index.tz is None:
            data.index = data.index.tz_localize(DEFAULT_TIMEZONE)
        elif str(data.index.tz) != DEFAULT_TIMEZONE:
            data.index = data.index.tz_convert(DEFAULT_TIMEZONE)

        return data

    def _standardize_columns(self, data: TimeSeriesData) -> TimeSeriesData:
        """Standardize column names to lowercase"""
        column_mapping = {}
        for col in data.columns:
            standardized = col.lower().strip()
            # Handle common variations
            if standardized in ['adj close', 'adjusted_close', 'adj_close']:
                standardized = 'adj_close'
            elif standardized in ['vol', 'volume']:
                standardized = 'volume'
            column_mapping[col] = standardized

        data = data.rename(columns=column_mapping)
        return data

    def _remove_duplicates(self, data: TimeSeriesData) -> TimeSeriesData:
        """Remove duplicate timestamps, keeping the last occurrence"""
        initial_len = len(data)
        data = data[~data.index.duplicated(keep='last')]
        final_len = len(data)

        if final_len < initial_len:
            self.logger.warning(f"Removed {initial_len - final_len} duplicate timestamps")

        return data

    def _validate_ohlc(self, data: TimeSeriesData) -> TimeSeriesData:
        """Validate and fix OHLC relationships"""
        price_cols = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in price_cols):
            return data

        # Identify rows with invalid OHLC relationships
        invalid_high = data['high'] < np.maximum(
            np.maximum(data['open'], data['close']),
            np.maximum(data['high'], data['low'])
        )
        invalid_low = data['low'] > np.minimum(
            np.minimum(data['open'], data['close']),
            np.minimum(data['high'], data['low'])
        )

        invalid_rows = invalid_high | invalid_low

        if invalid_rows.any():
            num_invalid = invalid_rows.sum()
            self.logger.warning(f"Found {num_invalid} rows with invalid OHLC relationships")

            # Fix by setting high = max(O,H,L,C) and low = min(O,H,L,C)
            for idx in data[invalid_rows].index:
                ohlc_values = [
                    data.loc[idx, 'open'],
                    data.loc[idx, 'high'],
                    data.loc[idx, 'low'],
                    data.loc[idx, 'close']
                ]
                data.loc[idx, 'high'] = max(ohlc_values)
                data.loc[idx, 'low'] = min(ohlc_values)

        return data

    def _fill_missing_values(self, data: TimeSeriesData) -> TimeSeriesData:
        """Fill missing values using specified method"""
        if data.isnull().any().any():
            self.logger.info("Filling missing values")

            if self.missing_method == "forward_fill":
                data = data.fillna(method='ffill')
            elif self.missing_method == "backward_fill":
                data = data.fillna(method='bfill')
            elif self.missing_method == "interpolate":
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                data[numeric_cols] = data[numeric_cols].interpolate()
            elif self.missing_method == "drop":
                data = data.dropna()
            else:
                raise ValueError(f"Unknown missing value method: {self.missing_method}")

            # Drop any remaining NaN rows
            data = data.dropna()

        return data

    def _remove_outliers(self, data: TimeSeriesData) -> TimeSeriesData:
        """Remove statistical outliers from price data"""
        price_columns = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_columns if col in data.columns]

        if not available_price_cols:
            return data

        initial_len = len(data)

        for col in available_price_cols:
            if self.outlier_method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR

                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)

            elif self.outlier_method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > self.outlier_threshold

            else:
                raise ValueError(f"Unknown outlier method: {self.outlier_method}")

            if outliers.any():
                self.logger.warning(f"Removing {outliers.sum()} outliers from {col}")
                data = data[~outliers]

        final_len = len(data)
        if final_len < initial_len:
            self.logger.info(f"Removed {initial_len - final_len} total outlier rows")

        return data

    def _validate_normalized_data(self, data: TimeSeriesData) -> bool:
        """Final validation of normalized data"""
        try:
            # Check required columns
            missing_required = set(REQUIRED_COLUMNS) - set(data.columns)
            if missing_required:
                self.logger.error(f"Missing required columns after normalization: {missing_required}")
                return False

            # Check for empty data
            if data.empty:
                self.logger.error("Data is empty after normalization")
                return False

            # Check for infinite values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if np.isinf(data[numeric_cols]).any().any():
                self.logger.error("Infinite values found in normalized data")
                return False

            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in data.columns and (data[col] <= 0).any():
                    self.logger.error(f"Non-positive values found in {col}")
                    return False

            # Check for negative volume
            if 'volume' in data.columns and (data['volume'] < 0).any():
                self.logger.error("Negative volume found")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False

    def adjust_for_splits_and_dividends(
        self,
        data: TimeSeriesData,
        splits: Optional[pd.Series] = None,
        dividends: Optional[pd.Series] = None
    ) -> TimeSeriesData:
        """
        Adjust prices for stock splits and dividends

        Args:
            data: OHLCV data
            splits: Series of split ratios indexed by date
            dividends: Series of dividend amounts indexed by date

        Returns:
            Adjusted data
        """
        adjusted = data.copy()

        # Adjust for splits
        if splits is not None and not splits.empty:
            self.logger.info("Adjusting for stock splits")
            price_cols = ['open', 'high', 'low', 'close']

            for split_date, split_ratio in splits.items():
                if split_ratio != 1.0:  # Only apply if there's actually a split
                    # Adjust prices before split date
                    mask = adjusted.index < split_date
                    for col in price_cols:
                        if col in adjusted.columns:
                            adjusted.loc[mask, col] /= split_ratio

                    # Adjust volume after split date
                    if 'volume' in adjusted.columns:
                        volume_mask = adjusted.index >= split_date
                        adjusted.loc[volume_mask, 'volume'] *= split_ratio

        # Adjust for dividends
        if dividends is not None and not dividends.empty:
            self.logger.info("Adjusting for dividends")
            price_cols = ['open', 'high', 'low', 'close']

            for div_date, div_amount in dividends.items():
                if div_amount > 0:  # Only apply if there's actually a dividend
                    # Adjust prices before ex-dividend date
                    mask = adjusted.index < div_date
                    for col in price_cols:
                        if col in adjusted.columns:
                            adjusted.loc[mask, col] -= div_amount

        return adjusted