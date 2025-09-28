"""
Mock Data Generator for BackT

Generates realistic synthetic market data for testing and development.
Supports various market scenarios and asset types.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings

from .loaders import DataLoader
from ..utils.types import TimeSeriesData


class MockDataGenerator:
    """
    Generates realistic synthetic market data for backtesting
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize mock data generator

        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate_price_series(
        self,
        start_date: str,
        end_date: str,
        initial_price: float = 100.0,
        annual_return: float = 0.08,
        annual_volatility: float = 0.16,
        frequency: str = '1D'
    ) -> pd.DataFrame:
        """
        Generate realistic price series using geometric Brownian motion

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            initial_price: Starting price
            annual_return: Expected annual return (drift)
            annual_volatility: Annual volatility
            frequency: Data frequency ('1D' for daily)

        Returns:
            DataFrame with OHLCV data
        """

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
        n_periods = len(date_range)

        if n_periods == 0:
            raise ValueError(f"No data points generated for period {start_date} to {end_date}")

        # Calculate time step (daily = 1/252 years)
        if frequency == '1D':
            dt = 1.0 / 252.0
        elif frequency == '1H':
            dt = 1.0 / (252.0 * 24.0)
        else:
            dt = 1.0 / 252.0  # Default to daily

        # Generate random returns using geometric Brownian motion
        random_shocks = np.random.normal(0, 1, n_periods)
        returns = (annual_return - 0.5 * annual_volatility**2) * dt + \
                 annual_volatility * np.sqrt(dt) * random_shocks

        # Calculate cumulative prices
        log_prices = np.log(initial_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)

        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, close_prices)):
            # Generate realistic OHLC from close price
            daily_vol = annual_volatility / np.sqrt(252)
            intraday_range = close * daily_vol * np.random.uniform(0.5, 2.0)

            # Generate O, H, L around close price
            open_price = close * (1 + np.random.normal(0, daily_vol * 0.5))
            high_price = max(open_price, close) + np.random.uniform(0, intraday_range)
            low_price = min(open_price, close) - np.random.uniform(0, intraday_range)

            # Ensure OHLC relationships are maintained
            high_price = max(high_price, open_price, close)
            low_price = min(low_price, open_price, close)

            # Generate volume (realistic trading volume)
            base_volume = 1000000
            volume_multiplier = 1 + 0.5 * abs(returns[i]) + np.random.uniform(-0.3, 0.3)
            volume = int(base_volume * volume_multiplier)

            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close, 2),
                'volume': volume
            })

        # Create DataFrame
        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'date'

        return df

    def generate_correlated_assets(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        correlations: Optional[Dict[str, Dict[str, float]]] = None,
        asset_configs: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple correlated asset price series

        Args:
            symbols: List of asset symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            correlations: Optional correlation matrix between assets
            asset_configs: Optional configurations for each asset

        Returns:
            Dictionary of DataFrames with OHLCV data for each symbol
        """

        # Default asset configurations
        default_configs = {
            'SPY': {'annual_return': 0.10, 'annual_volatility': 0.16, 'initial_price': 400},
            'QQQ': {'annual_return': 0.12, 'annual_volatility': 0.20, 'initial_price': 350},
            'TLT': {'annual_return': 0.04, 'annual_volatility': 0.08, 'initial_price': 120},
            'GLD': {'annual_return': 0.06, 'annual_volatility': 0.18, 'initial_price': 180},
            'IWM': {'annual_return': 0.09, 'annual_volatility': 0.22, 'initial_price': 200},
            'EFA': {'annual_return': 0.08, 'annual_volatility': 0.18, 'initial_price': 70},
            'VNQ': {'annual_return': 0.07, 'annual_volatility': 0.20, 'initial_price': 90},
            'XLE': {'annual_return': 0.05, 'annual_volatility': 0.25, 'initial_price': 80},
        }

        # Use provided configs or defaults
        configs = asset_configs or {}

        # Generate independent series first
        asset_data = {}
        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
        n_periods = len(date_range)

        # Generate correlated random shocks if correlations specified
        if correlations and len(symbols) > 1:
            # Create correlation matrix
            n_assets = len(symbols)
            corr_matrix = np.eye(n_assets)

            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if sym1 in correlations and sym2 in correlations[sym1]:
                        corr_matrix[i, j] = correlations[sym1][sym2]
                    elif sym2 in correlations and sym1 in correlations[sym2]:
                        corr_matrix[i, j] = correlations[sym2][sym1]

            # Generate correlated random shocks
            independent_shocks = np.random.normal(0, 1, (n_periods, n_assets))
            try:
                L = np.linalg.cholesky(corr_matrix)
                correlated_shocks = independent_shocks @ L.T
            except np.linalg.LinAlgError:
                # Fallback to independent shocks if correlation matrix is invalid
                correlated_shocks = independent_shocks
        else:
            # Independent shocks
            correlated_shocks = np.random.normal(0, 1, (n_periods, len(symbols)))

        # Generate price series for each asset
        for i, symbol in enumerate(symbols):
            config = configs.get(symbol, default_configs.get(symbol, {
                'annual_return': 0.08,
                'annual_volatility': 0.16,
                'initial_price': 100
            }))

            # Use correlated shocks for this asset
            shocks = correlated_shocks[:, i] if len(symbols) > 1 else correlated_shocks.flatten()

            asset_data[symbol] = self._generate_series_with_shocks(
                date_range=date_range,
                shocks=shocks,
                **config
            )

        return asset_data

    def _generate_series_with_shocks(
        self,
        date_range: pd.DatetimeIndex,
        shocks: np.ndarray,
        initial_price: float = 100.0,
        annual_return: float = 0.08,
        annual_volatility: float = 0.16
    ) -> pd.DataFrame:
        """
        Generate price series with given random shocks

        Args:
            date_range: Date range for the series
            shocks: Array of random shocks
            initial_price: Starting price
            annual_return: Expected annual return
            annual_volatility: Annual volatility

        Returns:
            DataFrame with OHLCV data
        """

        n_periods = len(date_range)
        dt = 1.0 / 252.0  # Daily time step

        # Calculate returns using provided shocks
        returns = (annual_return - 0.5 * annual_volatility**2) * dt + \
                 annual_volatility * np.sqrt(dt) * shocks

        # Calculate cumulative prices
        log_prices = np.log(initial_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)

        # Generate OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(date_range, close_prices)):
            daily_vol = annual_volatility / np.sqrt(252)
            intraday_range = close * daily_vol * np.random.uniform(0.5, 2.0)

            open_price = close * (1 + np.random.normal(0, daily_vol * 0.5))
            high_price = max(open_price, close) + np.random.uniform(0, intraday_range)
            low_price = min(open_price, close) - np.random.uniform(0, intraday_range)

            high_price = max(high_price, open_price, close)
            low_price = min(low_price, open_price, close)

            base_volume = 1000000
            volume_multiplier = 1 + 0.5 * abs(returns[i]) + np.random.uniform(-0.3, 0.3)
            volume = int(base_volume * volume_multiplier)

            data.append({
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close, 2),
                'volume': volume
            })

        df = pd.DataFrame(data, index=date_range)
        df.index.name = 'date'

        return df


class MockDataLoader(DataLoader):
    """
    Mock data loader that generates synthetic market data
    """

    def __init__(self,
                 scenario: str = 'normal',
                 seed: Optional[int] = 42,
                 **kwargs):
        """
        Initialize mock data loader

        Args:
            scenario: Market scenario ('normal', 'bull', 'bear', 'volatile')
            seed: Random seed for reproducible results
            **kwargs: Additional parameters
        """
        super().__init__()
        self.scenario = scenario
        self.generator = MockDataGenerator(seed=seed)
        self.seed = seed

    def load(self,
             symbols: Union[str, List[str]],
             start_date: str,
             end_date: str,
             **kwargs) -> TimeSeriesData:
        """
        Generate mock data for specified symbols and date range

        Args:
            symbols: Symbol or list of symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            **kwargs: Additional parameters

        Returns:
            Dictionary of DataFrames with OHLCV data
        """

        if isinstance(symbols, str):
            symbols = [symbols]

        # Define scenario parameters
        scenario_configs = self._get_scenario_configs()
        config = scenario_configs.get(self.scenario, scenario_configs['normal'])

        # Generate correlations for realistic market behavior
        correlations = self._get_realistic_correlations(symbols)

        # Generate correlated asset data
        try:
            data = self.generator.generate_correlated_assets(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                correlations=correlations,
                asset_configs=config.get('asset_configs')
            )

            # Apply scenario-specific modifications
            if self.scenario == 'bear':
                data = self._apply_bear_market(data)
            elif self.scenario == 'volatile':
                data = self._apply_high_volatility(data)

            return data

        except Exception as e:
            raise RuntimeError(f"Failed to generate mock data: {str(e)}")

    def validate_data(self, data: TimeSeriesData) -> bool:
        """
        Validate mock data format

        Args:
            data: DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return False

            # Check for reasonable price values
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (data[col] <= 0).any():
                    return False

            # Check OHLC relationships
            if (data['high'] < data['low']).any():
                return False
            if (data['high'] < data['open']).any() or (data['high'] < data['close']).any():
                return False
            if (data['low'] > data['open']).any() or (data['low'] > data['close']).any():
                return False

            # Check volume is non-negative
            if (data['volume'] < 0).any():
                return False

            # Check datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                return False

            return True

        except Exception:
            return False

    def _get_scenario_configs(self) -> Dict:
        """Get configuration for different market scenarios"""

        return {
            'normal': {
                'asset_configs': {
                    'SPY': {'annual_return': 0.10, 'annual_volatility': 0.16, 'initial_price': 400},
                    'QQQ': {'annual_return': 0.12, 'annual_volatility': 0.20, 'initial_price': 350},
                    'TLT': {'annual_return': 0.04, 'annual_volatility': 0.08, 'initial_price': 120},
                    'GLD': {'annual_return': 0.06, 'annual_volatility': 0.18, 'initial_price': 180},
                    'IWM': {'annual_return': 0.09, 'annual_volatility': 0.22, 'initial_price': 200},
                    'EFA': {'annual_return': 0.08, 'annual_volatility': 0.18, 'initial_price': 70},
                    'VNQ': {'annual_return': 0.07, 'annual_volatility': 0.20, 'initial_price': 90},
                    'XLE': {'annual_return': 0.05, 'annual_volatility': 0.25, 'initial_price': 80},
                }
            },
            'bull': {
                'asset_configs': {
                    'SPY': {'annual_return': 0.18, 'annual_volatility': 0.14, 'initial_price': 400},
                    'QQQ': {'annual_return': 0.22, 'annual_volatility': 0.18, 'initial_price': 350},
                    'TLT': {'annual_return': 0.02, 'annual_volatility': 0.06, 'initial_price': 120},
                    'GLD': {'annual_return': 0.08, 'annual_volatility': 0.16, 'initial_price': 180},
                    'IWM': {'annual_return': 0.20, 'annual_volatility': 0.20, 'initial_price': 200},
                    'EFA': {'annual_return': 0.15, 'annual_volatility': 0.16, 'initial_price': 70},
                    'VNQ': {'annual_return': 0.12, 'annual_volatility': 0.18, 'initial_price': 90},
                    'XLE': {'annual_return': 0.10, 'annual_volatility': 0.22, 'initial_price': 80},
                }
            },
            'bear': {
                'asset_configs': {
                    'SPY': {'annual_return': -0.05, 'annual_volatility': 0.25, 'initial_price': 400},
                    'QQQ': {'annual_return': -0.08, 'annual_volatility': 0.30, 'initial_price': 350},
                    'TLT': {'annual_return': 0.08, 'annual_volatility': 0.10, 'initial_price': 120},
                    'GLD': {'annual_return': 0.12, 'annual_volatility': 0.20, 'initial_price': 180},
                    'IWM': {'annual_return': -0.10, 'annual_volatility': 0.30, 'initial_price': 200},
                    'EFA': {'annual_return': -0.06, 'annual_volatility': 0.25, 'initial_price': 70},
                    'VNQ': {'annual_return': -0.08, 'annual_volatility': 0.28, 'initial_price': 90},
                    'XLE': {'annual_return': -0.15, 'annual_volatility': 0.35, 'initial_price': 80},
                }
            },
            'volatile': {
                'asset_configs': {
                    'SPY': {'annual_return': 0.08, 'annual_volatility': 0.30, 'initial_price': 400},
                    'QQQ': {'annual_return': 0.10, 'annual_volatility': 0.35, 'initial_price': 350},
                    'TLT': {'annual_return': 0.04, 'annual_volatility': 0.15, 'initial_price': 120},
                    'GLD': {'annual_return': 0.06, 'annual_volatility': 0.25, 'initial_price': 180},
                    'IWM': {'annual_return': 0.07, 'annual_volatility': 0.35, 'initial_price': 200},
                    'EFA': {'annual_return': 0.06, 'annual_volatility': 0.30, 'initial_price': 70},
                    'VNQ': {'annual_return': 0.05, 'annual_volatility': 0.32, 'initial_price': 90},
                    'XLE': {'annual_return': 0.03, 'annual_volatility': 0.40, 'initial_price': 80},
                }
            }
        }

    def _get_realistic_correlations(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Generate realistic correlations between assets"""

        # Define realistic correlations based on asset classes
        correlation_map = {
            ('SPY', 'QQQ'): 0.85,   # High correlation between US equity indices
            ('SPY', 'IWM'): 0.75,   # US large cap vs small cap
            ('SPY', 'EFA'): 0.65,   # US vs international developed
            ('SPY', 'TLT'): -0.25,  # Stocks vs bonds (negative)
            ('SPY', 'GLD'): 0.10,   # Stocks vs gold (low)
            ('SPY', 'VNQ'): 0.60,   # Stocks vs REITs
            ('QQQ', 'TLT'): -0.30,  # Tech stocks vs bonds
            ('TLT', 'GLD'): 0.20,   # Bonds vs gold
            ('EFA', 'GLD'): 0.15,   # International stocks vs gold
        }

        # Build correlation dictionary
        correlations = {}
        for symbol1 in symbols:
            correlations[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    # Look for correlation in map
                    corr_key = (symbol1, symbol2)
                    reverse_key = (symbol2, symbol1)

                    if corr_key in correlation_map:
                        correlations[symbol1][symbol2] = correlation_map[corr_key]
                    elif reverse_key in correlation_map:
                        correlations[symbol1][symbol2] = correlation_map[reverse_key]
                    else:
                        # Default low correlation for unspecified pairs
                        correlations[symbol1][symbol2] = 0.05

        return correlations

    def _apply_bear_market(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply bear market characteristics (additional drawdowns)"""

        modified_data = {}
        for symbol, df in data.items():
            # Create a copy
            modified_df = df.copy()

            # Add occasional sharp drawdowns
            n_periods = len(df)
            crash_periods = np.random.choice(n_periods, size=max(1, n_periods // 100), replace=False)

            for crash_day in crash_periods:
                if crash_day < len(modified_df):
                    # Sharp one-day drop
                    drop_factor = np.random.uniform(0.85, 0.95)
                    modified_df.iloc[crash_day:, ['open', 'high', 'low', 'close']] *= drop_factor

            modified_data[symbol] = modified_df

        return modified_data

    def _apply_high_volatility(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Apply high volatility characteristics"""

        modified_data = {}
        for symbol, df in data.items():
            modified_df = df.copy()

            # Add volatility spikes
            vol_multiplier = 1 + 0.5 * np.random.random(len(df))
            price_cols = ['open', 'high', 'low', 'close']

            for col in price_cols:
                daily_changes = modified_df[col].pct_change().fillna(0)
                amplified_changes = daily_changes * vol_multiplier
                modified_df[col] = modified_df[col].iloc[0] * (1 + amplified_changes).cumprod()

            modified_data[symbol] = modified_df

        return modified_data