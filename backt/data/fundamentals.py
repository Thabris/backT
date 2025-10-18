"""
Fundamental Data Loader for BackT

Loads fundamental financial data using yfinance for quality-based
and value-based trading strategies (AQR-style).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from ..utils.logging_config import LoggerMixin


class FundamentalsLoader(LoggerMixin):
    """Loads fundamental data for stocks using yfinance"""

    def __init__(self, cache_ttl: int = 86400):
        """
        Initialize fundamentals loader

        Parameters:
        -----------
        cache_ttl : int
            Cache time-to-live in seconds (default: 24 hours)
        """
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = cache_ttl

    def get_fundamentals(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for a symbol

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
        if use_cache and symbol in self._cache:
            return self._cache[symbol]

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
                'industry': info.get('industry')
            }

            # Cache the result
            if use_cache:
                self._cache[symbol] = fundamentals

            return fundamentals

        except Exception as e:
            self.logger.error(f"Error loading fundamentals for {symbol}: {str(e)}")
            return None

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

    def clear_cache(self):
        """Clear the fundamentals cache"""
        self._cache.clear()
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
