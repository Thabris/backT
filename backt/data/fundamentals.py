"""
Fundamental Data Loader for BackT

Loads fundamental financial data using FinancialModelingPrep (FMP) API
with fallback to yfinance for quality-based and value-based trading
strategies (AQR-style).
"""

import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Literal
from datetime import datetime, timedelta
from ..utils.logging_config import LoggerMixin


class FundamentalsLoader(LoggerMixin):
    """
    Loads fundamental data for stocks using FinancialModelingPrep API
    with automatic fallback to yfinance if FMP unavailable.
    """

    def __init__(
        self,
        cache_ttl: int = 86400,
        fmp_api_key: Optional[str] = None,
        data_source: Literal['fmp', 'yfinance', 'auto'] = 'auto'
    ):
        """
        Initialize fundamentals loader

        Parameters:
        -----------
        cache_ttl : int
            Cache time-to-live in seconds (default: 24 hours)
        fmp_api_key : str, optional
            FinancialModelingPrep API key. If not provided, will check
            environment variable FMP_API_KEY
        data_source : str
            Data source preference: 'fmp', 'yfinance', or 'auto'
            - 'fmp': Use FMP only (fail if unavailable)
            - 'yfinance': Use yfinance only
            - 'auto': Try FMP first, fallback to yfinance (default)
        """
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = cache_ttl
        self._cache_timestamps: Dict[str, datetime] = {}
        self._data_source = data_source

        # Get FMP API key from parameter or environment
        self._fmp_api_key = fmp_api_key or os.environ.get('FMP_API_KEY')

        # FMP API base URL
        self._fmp_base_url = 'https://financialmodelingprep.com/api/v3'

        # Track which source we're using
        if self._data_source == 'auto':
            if self._fmp_api_key:
                self.logger.info("FMP API key found - will use FMP with yfinance fallback")
            else:
                self.logger.info("No FMP API key - using yfinance only")
        elif self._data_source == 'fmp':
            if not self._fmp_api_key:
                self.logger.warning("FMP selected but no API key provided!")

        # API call counter for free tier monitoring (250/day limit)
        self._fmp_api_calls_today = 0
        self._fmp_call_date = datetime.now().date()

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache:
            return False
        if symbol not in self._cache_timestamps:
            return False
        age = datetime.now() - self._cache_timestamps[symbol]
        return age.total_seconds() < self._cache_ttl

    def _fmp_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Any]:
        """
        Make a request to FMP API with rate limiting and error handling

        Parameters:
        -----------
        endpoint : str
            API endpoint (e.g., '/ratios/AAPL')
        params : dict, optional
            Additional query parameters

        Returns:
        --------
        JSON response or None if request failed
        """
        if not self._fmp_api_key:
            return None

        # Check daily rate limit (250 calls/day for free tier)
        today = datetime.now().date()
        if today != self._fmp_call_date:
            self._fmp_api_calls_today = 0
            self._fmp_call_date = today

        if self._fmp_api_calls_today >= 250:
            self.logger.warning("FMP API daily limit reached (250 calls)")
            return None

        # Build URL
        url = f"{self._fmp_base_url}{endpoint}"
        params = params or {}
        params['apikey'] = self._fmp_api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            self._fmp_api_calls_today += 1

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                self.logger.warning("FMP API rate limit exceeded")
                return None
            else:
                self.logger.error(f"FMP API error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            self.logger.error(f"FMP API request failed: {str(e)}")
            return None

    def _get_fundamentals_from_fmp(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data from FinancialModelingPrep API

        Returns standardized fundamentals dict using FMP data
        """
        # Get financial ratios (most comprehensive single endpoint)
        ratios = self._fmp_request(f'/ratios/{symbol}', {'limit': 1})
        if not ratios or len(ratios) == 0:
            return None

        # Get key metrics for additional data
        key_metrics = self._fmp_request(f'/key-metrics/{symbol}', {'limit': 1})
        key_metrics_data = key_metrics[0] if key_metrics and len(key_metrics) > 0 else {}

        # Get company profile for metadata
        profile = self._fmp_request(f'/profile/{symbol}')
        profile_data = profile[0] if profile and len(profile) > 0 else {}

        # Latest ratios
        latest = ratios[0]

        # Map FMP data to our standardized format
        fundamentals = {
            # Profitability Metrics
            'roe': latest.get('returnOnEquity'),
            'roa': latest.get('returnOnAssets'),
            'profit_margin': latest.get('netProfitMargin'),
            'gross_margin': latest.get('grossProfitMargin'),
            'operating_margin': latest.get('operatingProfitMargin'),
            'ebitda_margin': None,  # Not in ratios endpoint

            # Growth Metrics (from key metrics if available)
            'revenue_growth': key_metrics_data.get('revenuePerShareGrowth'),
            'earnings_growth': None,  # Would need income statement comparison

            # Safety/Leverage Metrics
            'debt_to_equity': latest.get('debtEquityRatio'),
            'current_ratio': latest.get('currentRatio'),
            'quick_ratio': latest.get('quickRatio'),
            'beta': profile_data.get('beta'),
            'total_debt': latest.get('totalDebt'),
            'total_cash': None,  # Would need balance sheet

            # Payout/Capital Allocation
            'free_cashflow': key_metrics_data.get('freeCashFlowPerShare'),
            'operating_cashflow': key_metrics_data.get('operatingCashFlowPerShare'),

            # Value Metrics
            'price_to_book': latest.get('priceToBookRatio'),
            'trailing_pe': latest.get('priceEarningsRatio'),
            'forward_pe': None,  # Not in ratios
            'peg_ratio': latest.get('priceEarningsToGrowthRatio'),
            'book_value': key_metrics_data.get('bookValuePerShare'),

            # Size
            'market_cap': profile_data.get('mktCap'),
            'enterprise_value': key_metrics_data.get('enterpriseValue'),
            'total_revenue': None,  # Would need income statement

            # Metadata
            'symbol': symbol,
            'sector': profile_data.get('sector'),
            'industry': profile_data.get('industry'),
            'data_source': 'fmp',
            'date': latest.get('date')
        }

        return fundamentals

    def _get_fundamentals_from_yfinance(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data from yfinance (fallback method)

        Returns standardized fundamentals dict using yfinance data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key metrics
            fundamentals = {
                # Profitability Metrics
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'gross_margin': info.get('grossMargins'),
                'operating_margin': info.get('operatingMargins'),
                'ebitda_margin': info.get('ebitdaMargins'),

                # Growth Metrics
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),

                # Safety/Leverage Metrics
                'debt_to_equity': info.get('debtToEquity'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'beta': info.get('beta'),
                'total_debt': info.get('totalDebt'),
                'total_cash': info.get('totalCash'),

                # Payout/Capital Allocation
                'free_cashflow': info.get('freeCashflow'),
                'operating_cashflow': info.get('operatingCashflow'),

                # Value Metrics
                'price_to_book': info.get('priceToBook'),
                'trailing_pe': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'book_value': info.get('bookValue'),

                # Size
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'total_revenue': info.get('totalRevenue'),

                # Metadata
                'symbol': symbol,
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'data_source': 'yfinance'
            }

            return fundamentals

        except Exception as e:
            self.logger.error(f"Error loading fundamentals from yfinance for {symbol}: {str(e)}")
            return None

    def get_fundamentals(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a symbol

        Tries FMP first (if configured), falls back to yfinance

        Parameters:
        -----------
        symbol : str
            Stock ticker symbol
        use_cache : bool
            Whether to use cached data if available

        Returns:
        --------
        Dict with fundamental metrics or None if data unavailable
        """
        # Check cache
        if use_cache and self._is_cache_valid(symbol):
            return self._cache[symbol]

        fundamentals = None

        # Try data sources based on configuration
        if self._data_source == 'yfinance':
            fundamentals = self._get_fundamentals_from_yfinance(symbol)

        elif self._data_source == 'fmp':
            fundamentals = self._get_fundamentals_from_fmp(symbol)
            if fundamentals is None:
                self.logger.error(f"FMP data unavailable for {symbol} (fmp-only mode)")

        else:  # auto mode
            # Try FMP first if API key available
            if self._fmp_api_key:
                fundamentals = self._get_fundamentals_from_fmp(symbol)
                if fundamentals is not None:
                    self.logger.debug(f"Loaded {symbol} from FMP")

            # Fallback to yfinance
            if fundamentals is None:
                fundamentals = self._get_fundamentals_from_yfinance(symbol)
                if fundamentals is not None:
                    self.logger.debug(f"Loaded {symbol} from yfinance (fallback)")

        # Cache the result
        if fundamentals is not None and use_cache:
            self._cache[symbol] = fundamentals
            self._cache_timestamps[symbol] = datetime.now()

        return fundamentals

    def get_multiple_fundamentals(self, symbols: list) -> Dict[str, Dict]:
        """
        Get fundamentals for multiple symbols

        Parameters:
        -----------
        symbols : list
            List of ticker symbols

        Returns:
        --------
        Dict mapping symbol to fundamentals dict
        """
        results = {}
        for symbol in symbols:
            data = self.get_fundamentals(symbol)
            if data is not None:
                results[symbol] = data
        return results

    def get_api_usage_stats(self) -> Dict[str, Any]:
        """
        Get API usage statistics for monitoring

        Returns:
        --------
        Dict with usage stats including:
        - fmp_calls_today: Number of FMP API calls today
        - fmp_calls_remaining: Remaining calls for today (out of 250)
        - cache_size: Number of cached symbols
        - data_source: Current data source configuration
        """
        return {
            'fmp_calls_today': self._fmp_api_calls_today,
            'fmp_calls_remaining': max(0, 250 - self._fmp_api_calls_today),
            'cache_size': len(self._cache),
            'data_source': self._data_source,
            'has_fmp_key': self._fmp_api_key is not None
        }

    def clear_cache(self):
        """Clear the fundamentals cache"""
        self._cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Fundamentals cache cleared")


def calculate_quality_score(fundamentals: Dict[str, Any]) -> Optional[float]:
    """
    Calculate AQR-style composite quality score

    Quality = Average of Profitability, Growth, Safety, and Payout scores

    Parameters:
    -----------
    fundamentals : dict
        Dictionary of fundamental metrics

    Returns:
    --------
    Quality score (0-1) or None if insufficient data
    """
    scores = []

    # Profitability (higher = better)
    prof_metrics = []
    if fundamentals.get('roe') is not None and fundamentals['roe'] > 0:
        prof_metrics.append(fundamentals['roe'])
    if fundamentals.get('roa') is not None and fundamentals['roa'] > 0:
        prof_metrics.append(fundamentals['roa'])
    if fundamentals.get('gross_margin') is not None:
        prof_metrics.append(fundamentals['gross_margin'])
    if fundamentals.get('operating_margin') is not None:
        prof_metrics.append(fundamentals['operating_margin'])

    if prof_metrics:
        scores.append(np.mean(prof_metrics))

    # Growth (higher = better)
    growth_metrics = []
    if fundamentals.get('revenue_growth') is not None:
        growth_metrics.append(fundamentals['revenue_growth'])
    if fundamentals.get('earnings_growth') is not None:
        growth_metrics.append(fundamentals['earnings_growth'])

    if growth_metrics:
        scores.append(np.mean(growth_metrics))

    # Safety (lower debt, higher liquidity = better)
    # Normalize to 0-1 scale where 1 is safest
    safety_score = 0
    safety_count = 0

    # Low debt is good - inverse and cap
    if fundamentals.get('debt_to_equity') is not None:
        d2e = fundamentals['debt_to_equity']
        # Cap at 200%, then inverse
        safety_score += max(0, 1 - min(d2e / 200, 1))
        safety_count += 1

    # High current ratio is good
    if fundamentals.get('current_ratio') is not None:
        cr = fundamentals['current_ratio']
        # Healthy is >= 1.5, cap at 3
        safety_score += min(cr / 1.5, 1)
        safety_count += 1

    # Low beta is safer (but not a quality metric in original AQR, added for safety)
    if fundamentals.get('beta') is not None:
        beta = abs(fundamentals['beta'])
        # Lower beta is safer, cap at 2
        safety_score += max(0, 1 - min(beta / 2, 1))
        safety_count += 1

    if safety_count > 0:
        scores.append(safety_score / safety_count)

    # Payout/Capital Allocation (positive FCF = good)
    if fundamentals.get('free_cashflow') is not None and fundamentals.get('market_cap') is not None:
        if fundamentals['market_cap'] > 0:
            fcf_yield = fundamentals['free_cashflow'] / fundamentals['market_cap']
            # Normalize: 5% FCF yield = good, 10%+ = excellent
            payout_score = min(fcf_yield / 0.10, 1)
            scores.append(max(0, payout_score))

    # Return average of all components
    if len(scores) >= 2:  # Need at least 2 components
        return np.mean(scores)
    return None


def calculate_value_score(fundamentals: Dict[str, Any]) -> Optional[float]:
    """
    Calculate AQR-style value score

    Value = Average of Book/Price, Earnings/Price, and Cashflow/Price

    Parameters:
    -----------
    fundamentals : dict
        Dictionary of fundamental metrics

    Returns:
    --------
    Value score or None if insufficient data
    """
    scores = []

    # Book-to-Price (inverse of P/B)
    if fundamentals.get('price_to_book') is not None and fundamentals['price_to_book'] > 0:
        book_to_price = 1.0 / fundamentals['price_to_book']
        scores.append(book_to_price)

    # Earnings-to-Price (inverse of P/E)
    if fundamentals.get('trailing_pe') is not None and fundamentals['trailing_pe'] > 0:
        earnings_to_price = 1.0 / fundamentals['trailing_pe']
        scores.append(earnings_to_price)

    # Cashflow-to-Price
    if (fundamentals.get('free_cashflow') is not None and
        fundamentals.get('market_cap') is not None and
        fundamentals['market_cap'] > 0):
        cf_to_price = fundamentals['free_cashflow'] / fundamentals['market_cap']
        scores.append(cf_to_price)

    # Return average of available metrics
    if len(scores) >= 2:  # Need at least 2 value metrics
        return np.mean(scores)
    return None
