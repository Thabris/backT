"""
Technical indicators for BackT strategies

Common technical analysis indicators for use in trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


class TechnicalIndicators:
    """Collection of technical analysis indicators"""

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period).mean()

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return pd.DataFrame({
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        })

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)

        Returns DataFrame with:
        - macd: MACD line (fast EMA - slow EMA)
        - signal: Signal line (EMA of MACD)
        - histogram: MACD histogram (MACD - Signal)
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator

        Returns DataFrame with:
        - k: %K line (fast stochastic)
        - d: %D line (slow stochastic, signal line)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()

        # Fast %K
        k_fast = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # Slow %K (smoothed)
        k = k_fast.rolling(window=smooth_k).mean()

        # %D (signal line)
        d = k.rolling(window=smooth_d).mean()

        return pd.DataFrame({
            'k': k,
            'd': d
        })

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
        """
        ADX (Average Directional Index) - measures trend strength

        Returns DataFrame with:
        - adx: ADX line (trend strength, 0-100)
        - plus_di: +DI (positive directional indicator)
        - minus_di: -DI (negative directional indicator)

        ADX > 25: Strong trend
        ADX < 20: Weak trend/ranging
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm < minus_dm)] = 0
        minus_dm[(minus_dm < plus_dm)] = 0

        # Directional Indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        ATR (Average True Range) - measures volatility
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()