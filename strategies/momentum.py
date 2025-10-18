"""
Momentum-based Trading Strategies

Collection of momentum and technical indicator strategies including:
- Traditional MA crossovers (long-only and long-short)
- Kalman-enhanced MA crossovers (long-only and long-short)
- RSI mean reversion (overbought/oversold)
- MACD crossover (trend following)
- Stochastic oscillator (momentum)
- Bollinger Bands mean reversion
- ADX trend strength filter

All strategies follow the standard BackT strategy signature.
"""

from typing import Dict, Any
import pandas as pd
from backt.signal import TechnicalIndicators
from backt.signal.kalman import KalmanFilter1D


def ma_crossover_long_only(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Traditional Moving Average Crossover Strategy (Long-Only)

    Logic:
    - Calculate short-term (fast) and long-term (slow) moving averages
    - LONG when fast MA crosses above slow MA (golden cross)
    - CASH when fast MA crosses below slow MA (death cross)
    - Equal weight allocation across selected positions

    Parameters:
    -----------
    fast_ma : int, default=20
        Short-term moving average period
    slow_ma : int, default=50
        Long-term moving average period
    min_periods : int, default=60
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_ma': 20, 'slow_ma': 50, 'max_position_size': 0.25}
    >>> result = backtester.run(
    ...     strategy=ma_crossover_long_only,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    fast_ma = params.get('fast_ma', 20)
    slow_ma = params.get('slow_ma', 50)
    min_periods = params.get('min_periods', 60)
    max_position_size = params.get('max_position_size', 0.25)

    orders = {}
    signals = {}
    long_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate moving averages
            fast_mavg = TechnicalIndicators.sma(data['close'], fast_ma)
            slow_mavg = TechnicalIndicators.sma(data['close'], slow_ma)

            # Get current and previous values
            if len(fast_mavg) < 2 or len(slow_mavg) < 2:
                continue

            current_fast = fast_mavg.iloc[-1]
            current_slow = slow_mavg.iloc[-1]
            prev_fast = fast_mavg.iloc[-2]
            prev_slow = slow_mavg.iloc[-2]

            # Detect crossovers
            golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

            # Generate signals - LONG ONLY
            if golden_cross:
                signals[symbol] = 'BUY'
                long_positions.append(symbol)
            elif death_cross:
                signals[symbol] = 'SELL'
                # Don't add to long_positions (go to cash)
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
                long_positions.append(symbol)
            else:
                signals[symbol] = 'NEUTRAL'
                # Stay in cash

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        # Equal weight allocation, but respect max position size
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Create LONG orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions for assets with no signal
    for symbol in market_data.keys():
        if symbol not in long_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state for analysis
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['total_positions'] = total_positions
    context['position_weight'] = weight_per_position if total_positions > 0 else 0

    return orders


def ma_crossover_long_short(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Traditional Moving Average Crossover Strategy (Long-Short)

    Logic:
    - Calculate short-term (fast) and long-term (slow) moving averages
    - LONG when fast MA crosses above slow MA (golden cross)
    - SHORT when fast MA crosses below slow MA (death cross)
    - Equal weight allocation across selected positions (long and short)

    Key Differences from Long-Only:
    - Goes SHORT instead of cash on death crosses
    - Can profit from declining markets
    - Higher potential returns but also higher risk

    Parameters:
    -----------
    fast_ma : int, default=20
        Short-term moving average period
    slow_ma : int, default=50
        Long-term moving average period
    min_periods : int, default=60
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_ma': 20, 'slow_ma': 50, 'max_position_size': 0.25}
    >>> result = backtester.run(
    ...     strategy=ma_crossover_long_short,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    fast_ma = params.get('fast_ma', 20)
    slow_ma = params.get('slow_ma', 50)
    min_periods = params.get('min_periods', 60)
    max_position_size = params.get('max_position_size', 0.25)

    orders = {}
    signals = {}
    long_positions = []
    short_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate moving averages
            fast_mavg = TechnicalIndicators.sma(data['close'], fast_ma)
            slow_mavg = TechnicalIndicators.sma(data['close'], slow_ma)

            # Get current and previous values
            if len(fast_mavg) < 2 or len(slow_mavg) < 2:
                continue

            current_fast = fast_mavg.iloc[-1]
            current_slow = slow_mavg.iloc[-1]
            prev_fast = fast_mavg.iloc[-2]
            prev_slow = slow_mavg.iloc[-2]

            # Detect crossovers
            golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

            # Generate signals - KEY DIFFERENCE: SHORT on death cross
            if golden_cross:
                signals[symbol] = 'BUY'
                long_positions.append(symbol)
            elif death_cross:
                signals[symbol] = 'SELL_SHORT'
                short_positions.append(symbol)
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
                long_positions.append(symbol)
            else:
                signals[symbol] = 'HOLD_SHORT'
                short_positions.append(symbol)

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        # Equal weight allocation, but respect max position size
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Create LONG orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position  # Positive weight = LONG
            }

        # Create SHORT orders
        for symbol in short_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position  # Negative weight = SHORT
            }

    # Close positions for assets with no signal
    for symbol in market_data.keys():
        if symbol not in (long_positions + short_positions):
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state for analysis
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['short_positions'] = short_positions
    context['num_long'] = len(long_positions)
    context['num_short'] = len(short_positions)
    context['total_positions'] = total_positions
    context['position_weight'] = weight_per_position if total_positions > 0 else 0

    return orders


def kalman_ma_crossover_long_only(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Kalman-Enhanced Moving Average Crossover Strategy (Long-Only)

    Logic:
    - Use Kalman filters instead of traditional moving averages for denoising
    - Two filters with different Q parameters act as fast/slow signals
    - LONG when fast Kalman > slow Kalman (trend confirmation)
    - CASH when fast Kalman < slow Kalman (downtrend)
    - Equal weight allocation across selected positions

    Key Advantages of Kalman Filters:
    - Superior noise reduction without lag
    - Adaptive to changing market conditions
    - No lookback window needed (real-time adaptation)
    - Smoother signals with fewer whipsaws

    Parameters:
    -----------
    Q_fast : float, default=0.01
        Process noise for fast filter (more responsive)
    Q_slow : float, default=0.001
        Process noise for slow filter (smoother)
    R : float, default=1.0
        Measurement noise (standard)
    min_periods : int, default=60
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'Q_fast': 0.01, 'Q_slow': 0.001, 'R': 1.0}
    >>> result = backtester.run(
    ...     strategy=kalman_ma_crossover_long_only,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    Q_fast = params.get('Q_fast', 0.01)
    Q_slow = params.get('Q_slow', 0.001)
    R = params.get('R', 1.0)
    min_periods = params.get('min_periods', 60)
    max_position_size = params.get('max_position_size', 0.25)

    # Initialize Kalman filters in context if not already present
    if 'kalman_filters' not in context:
        context['kalman_filters'] = {}

    orders = {}
    signals = {}
    long_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Reset and compute Kalman filters
            # (In production, you'd maintain state, but for backtesting we recompute)
            kf_fast = KalmanFilter1D(Q=Q_fast, R=R)
            kf_slow = KalmanFilter1D(Q=Q_slow, R=R)

            fast_values = [kf_fast.update(p) for p in data['close']]
            slow_values = [kf_slow.update(p) for p in data['close']]

            # Get current and previous Kalman estimates
            if len(fast_values) < 2 or len(slow_values) < 2:
                continue

            current_fast = fast_values[-1]
            current_slow = slow_values[-1]
            prev_fast = fast_values[-2]
            prev_slow = slow_values[-2]

            # Detect crossovers using Kalman-filtered values
            golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

            # Generate signals - LONG ONLY
            if golden_cross:
                signals[symbol] = 'BUY'
                long_positions.append(symbol)
            elif death_cross:
                signals[symbol] = 'SELL'
                # Go to cash
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
                long_positions.append(symbol)
            else:
                signals[symbol] = 'NEUTRAL'
                # Stay in cash

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        # Equal weight allocation, but respect max position size
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Create LONG orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions for assets with no signal
    for symbol in market_data.keys():
        if symbol not in long_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state for analysis
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['total_positions'] = total_positions
    context['position_weight'] = weight_per_position if total_positions > 0 else 0

    return orders


def kalman_ma_crossover_long_short(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Kalman-Enhanced Moving Average Crossover Strategy (Long-Short)

    Logic:
    - Use Kalman filters instead of traditional moving averages for denoising
    - Two filters with different Q parameters act as fast/slow signals
    - LONG when fast Kalman > slow Kalman (trend confirmation)
    - SHORT when fast Kalman < slow Kalman (downtrend confirmation)
    - Equal weight allocation across selected positions (long and short)

    Key Advantages of Kalman Filters:
    - Superior noise reduction without lag
    - Adaptive to changing market conditions
    - No lookback window needed (real-time adaptation)
    - Smoother signals with fewer whipsaws

    Parameters:
    -----------
    Q_fast : float, default=0.01
        Process noise for fast filter (more responsive)
    Q_slow : float, default=0.001
        Process noise for slow filter (smoother)
    R : float, default=1.0
        Measurement noise (standard)
    min_periods : int, default=60
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'Q_fast': 0.01, 'Q_slow': 0.001, 'R': 1.0}
    >>> result = backtester.run(
    ...     strategy=kalman_ma_crossover_long_short,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    Q_fast = params.get('Q_fast', 0.01)
    Q_slow = params.get('Q_slow', 0.001)
    R = params.get('R', 1.0)
    min_periods = params.get('min_periods', 60)
    max_position_size = params.get('max_position_size', 0.25)

    # Initialize Kalman filters in context if not already present
    if 'kalman_filters' not in context:
        context['kalman_filters'] = {}

    orders = {}
    signals = {}
    long_positions = []
    short_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Reset and compute Kalman filters
            kf_fast = KalmanFilter1D(Q=Q_fast, R=R)
            kf_slow = KalmanFilter1D(Q=Q_slow, R=R)

            fast_values = [kf_fast.update(p) for p in data['close']]
            slow_values = [kf_slow.update(p) for p in data['close']]

            # Get current and previous Kalman estimates
            if len(fast_values) < 2 or len(slow_values) < 2:
                continue

            current_fast = fast_values[-1]
            current_slow = slow_values[-1]
            prev_fast = fast_values[-2]
            prev_slow = slow_values[-2]

            # Detect crossovers using Kalman-filtered values
            golden_cross = (prev_fast <= prev_slow) and (current_fast > current_slow)
            death_cross = (prev_fast >= prev_slow) and (current_fast < current_slow)

            # Generate signals - SHORT on death cross, LONG on golden cross
            if golden_cross:
                signals[symbol] = 'BUY'
                long_positions.append(symbol)
            elif death_cross:
                signals[symbol] = 'SELL_SHORT'
                short_positions.append(symbol)
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
                long_positions.append(symbol)
            else:
                signals[symbol] = 'HOLD_SHORT'
                short_positions.append(symbol)

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        # Equal weight allocation, but respect max position size
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Create LONG orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position  # Positive weight = LONG
            }

        # Create SHORT orders
        for symbol in short_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position  # Negative weight = SHORT
            }

    # Close positions for assets with no signal
    for symbol in market_data.keys():
        if symbol not in (long_positions + short_positions):
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state for analysis
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['short_positions'] = short_positions
    context['num_long'] = len(long_positions)
    context['num_short'] = len(short_positions)
    context['total_positions'] = total_positions
    context['position_weight'] = weight_per_position if total_positions > 0 else 0

    return orders
# New strategies to add to momentum.py


def rsi_mean_reversion(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    RSI Mean Reversion Strategy

    Logic:
    - BUY when RSI < oversold_threshold (asset is oversold)
    - SELL when RSI > overbought_threshold (asset is overbought)
    - Equal weight allocation across selected positions

    Parameters:
    -----------
    rsi_period : int, default=14
        RSI calculation period
    oversold_threshold : float, default=30
        RSI level considered oversold (buy signal)
    overbought_threshold : float, default=70
        RSI level considered overbought (sell signal)
    min_periods : int, default=30
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'rsi_period': 14, 'oversold_threshold': 30, 'overbought_threshold': 70}
    >>> result = backtester.run(
    ...     strategy=rsi_mean_reversion,
    ...     universe=['SPY', 'QQQ', 'TLT'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    rsi_period = params.get('rsi_period', 14)
    oversold = params.get('oversold_threshold', 30)
    overbought = params.get('overbought_threshold', 70)
    min_periods = params.get('min_periods', 30)
    max_position_size = params.get('max_position_size', 0.25)

    orders = {}
    signals = {}
    long_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate RSI
            rsi = TechnicalIndicators.rsi(data['close'], rsi_period)

            if len(rsi) < 2:
                continue

            current_rsi = rsi.iloc[-1]

            # Generate signals based on RSI levels
            if current_rsi < oversold:
                signals[symbol] = 'OVERSOLD_BUY'
                long_positions.append(symbol)
            elif current_rsi > overbought:
                signals[symbol] = 'OVERBOUGHT_SELL'
                # Exit position
            else:
                # Check if we have a position to hold
                if symbol in positions and hasattr(positions[symbol], 'quantity'):
                    if positions[symbol].quantity > 0:
                        # Hold position until overbought
                        signals[symbol] = 'HOLD'
                        long_positions.append(symbol)
                    else:
                        signals[symbol] = 'NEUTRAL'
                else:
                    signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = min(1.0 / total_positions, max_position_size)

        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions not in long_positions
    for symbol in market_data.keys():
        if symbol not in long_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['total_positions'] = total_positions

    return orders


def macd_crossover(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    MACD Crossover Strategy (Trend Following)

    Logic:
    - BUY when MACD line crosses above signal line (bullish crossover)
    - SELL when MACD line crosses below signal line (bearish crossover)
    - Can trade long-only or long-short based on allow_short parameter

    Parameters:
    -----------
    fast_period : int, default=12
        Fast EMA period for MACD
    slow_period : int, default=26
        Slow EMA period for MACD
    signal_period : int, default=9
        Signal line EMA period
    min_periods : int, default=35
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)
    allow_short : bool, default=False
        If True, go short on bearish crossover; if False, go to cash

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'allow_short': True}
    >>> result = backtester.run(
    ...     strategy=macd_crossover,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    fast_period = params.get('fast_period', 12)
    slow_period = params.get('slow_period', 26)
    signal_period = params.get('signal_period', 9)
    min_periods = params.get('min_periods', 35)
    max_position_size = params.get('max_position_size', 0.25)
    allow_short = params.get('allow_short', False)

    orders = {}
    signals = {}
    long_positions = []
    short_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate MACD
            macd_data = TechnicalIndicators.macd(
                data['close'],
                fast=fast_period,
                slow=slow_period,
                signal=signal_period
            )

            if len(macd_data) < 2:
                continue

            current_macd = macd_data['macd'].iloc[-1]
            current_signal = macd_data['signal'].iloc[-1]
            prev_macd = macd_data['macd'].iloc[-2]
            prev_signal = macd_data['signal'].iloc[-2]

            # Detect crossovers
            bullish_cross = (prev_macd <= prev_signal) and (current_macd > current_signal)
            bearish_cross = (prev_macd >= prev_signal) and (current_macd < current_signal)

            # Generate signals
            if bullish_cross:
                signals[symbol] = 'MACD_BUY'
                long_positions.append(symbol)
            elif bearish_cross:
                if allow_short:
                    signals[symbol] = 'MACD_SELL_SHORT'
                    short_positions.append(symbol)
                else:
                    signals[symbol] = 'MACD_SELL'
            elif current_macd > current_signal:
                signals[symbol] = 'MACD_HOLD_LONG'
                long_positions.append(symbol)
            elif current_macd < current_signal and allow_short:
                signals[symbol] = 'MACD_HOLD_SHORT'
                short_positions.append(symbol)
            else:
                signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Long orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

        # Short orders
        for symbol in short_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position
            }

    # Close positions not in active positions
    active_positions = long_positions + short_positions
    for symbol in market_data.keys():
        if symbol not in active_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['short_positions'] = short_positions
    context['total_positions'] = total_positions

    return orders


def stochastic_momentum(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Stochastic Oscillator Momentum Strategy

    Logic:
    - BUY when %K crosses above %D in oversold territory (< 20)
    - SELL when %K crosses below %D in overbought territory (> 80)
    - Equal weight allocation across selected positions

    Parameters:
    -----------
    period : int, default=14
        Lookback period for stochastic calculation
    smooth_k : int, default=3
        Smoothing period for %K
    smooth_d : int, default=3
        Smoothing period for %D
    oversold : float, default=20
        Oversold level
    overbought : float, default=80
        Overbought level
    min_periods : int, default=30
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'period': 14, 'oversold': 20, 'overbought': 80}
    >>> result = backtester.run(
    ...     strategy=stochastic_momentum,
    ...     universe=['SPY', 'QQQ', 'IWM'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    period = params.get('period', 14)
    smooth_k = params.get('smooth_k', 3)
    smooth_d = params.get('smooth_d', 3)
    oversold = params.get('oversold', 20)
    overbought = params.get('overbought', 80)
    min_periods = params.get('min_periods', 30)
    max_position_size = params.get('max_position_size', 0.25)

    orders = {}
    signals = {}
    long_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate Stochastic
            stoch = TechnicalIndicators.stochastic(
                data['high'],
                data['low'],
                data['close'],
                period=period,
                smooth_k=smooth_k,
                smooth_d=smooth_d
            )

            if len(stoch) < 2:
                continue

            current_k = stoch['k'].iloc[-1]
            current_d = stoch['d'].iloc[-1]
            prev_k = stoch['k'].iloc[-2]
            prev_d = stoch['d'].iloc[-2]

            # Detect crossovers
            bullish_cross = (prev_k <= prev_d) and (current_k > current_d) and (current_k < oversold)
            bearish_cross = (prev_k >= prev_d) and (current_k < current_d) and (current_k > overbought)

            # Generate signals
            if bullish_cross:
                signals[symbol] = 'STOCH_OVERSOLD_BUY'
                long_positions.append(symbol)
            elif bearish_cross:
                signals[symbol] = 'STOCH_OVERBOUGHT_SELL'
            elif current_k > current_d and current_k < overbought:
                # Hold long position if still bullish
                signals[symbol] = 'STOCH_HOLD'
                long_positions.append(symbol)
            else:
                signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = min(1.0 / total_positions, max_position_size)

        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions not in long_positions
    for symbol in market_data.keys():
        if symbol not in long_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['total_positions'] = total_positions

    return orders


def bollinger_mean_reversion(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    Bollinger Bands Mean Reversion Strategy

    Logic:
    - BUY when price touches or crosses below lower band (oversold)
    - SELL when price touches or crosses above upper band (overbought)
    - Exit when price returns to middle band

    Parameters:
    -----------
    bb_period : int, default=20
        Bollinger Bands period
    bb_std : float, default=2.0
        Number of standard deviations for bands
    min_periods : int, default=30
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'bb_period': 20, 'bb_std': 2.0}
    >>> result = backtester.run(
    ...     strategy=bollinger_mean_reversion,
    ...     universe=['SPY', 'QQQ', 'TLT'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    bb_period = params.get('bb_period', 20)
    bb_std = params.get('bb_std', 2.0)
    min_periods = params.get('min_periods', 30)
    max_position_size = params.get('max_position_size', 0.25)

    orders = {}
    signals = {}
    long_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate Bollinger Bands
            bb = TechnicalIndicators.bollinger_bands(data['close'], period=bb_period, std_dev=bb_std)

            if len(bb) < 2:
                continue

            current_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-2]

            lower_band = bb['lower'].iloc[-1]
            upper_band = bb['upper'].iloc[-1]
            middle_band = bb['middle'].iloc[-1]

            prev_lower = bb['lower'].iloc[-2]
            prev_upper = bb['upper'].iloc[-2]

            # Detect band touches/crosses
            touches_lower = prev_price > prev_lower and current_price <= lower_band
            touches_upper = prev_price < prev_upper and current_price >= upper_band
            near_middle = abs(current_price - middle_band) / middle_band < 0.01  # Within 1% of middle

            # Generate signals
            if touches_lower:
                signals[symbol] = 'BB_LOWER_BUY'
                long_positions.append(symbol)
            elif touches_upper or near_middle:
                signals[symbol] = 'BB_EXIT'
            else:
                # Hold position if we have one
                if symbol in positions and hasattr(positions[symbol], 'quantity'):
                    if positions[symbol].quantity > 0:
                        signals[symbol] = 'BB_HOLD'
                        long_positions.append(symbol)
                    else:
                        signals[symbol] = 'NEUTRAL'
                else:
                    signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = min(1.0 / total_positions, max_position_size)

        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions not in long_positions
    for symbol in market_data.keys():
        if symbol not in long_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['total_positions'] = total_positions

    return orders


def adx_trend_filter(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Any],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    """
    ADX Trend Strength Filter with Directional Trading

    Logic:
    - Only trade when ADX > threshold (strong trend)
    - BUY when +DI > -DI (bullish trend)
    - SELL/SHORT when -DI > +DI (bearish trend)
    - Exit when ADX falls below threshold (weak trend)

    Parameters:
    -----------
    adx_period : int, default=14
        ADX calculation period
    adx_threshold : float, default=25
        Minimum ADX for trend strength (25 = strong trend)
    min_periods : int, default=30
        Minimum data points required
    max_position_size : float, default=0.25
        Maximum weight per position (0.0 to 1.0)
    allow_short : bool, default=False
        If True, go short in bearish trends; if False, go to cash

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'adx_period': 14, 'adx_threshold': 25, 'allow_short': True}
    >>> result = backtester.run(
    ...     strategy=adx_trend_filter,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    adx_period = params.get('adx_period', 14)
    adx_threshold = params.get('adx_threshold', 25)
    min_periods = params.get('min_periods', 30)
    max_position_size = params.get('max_position_size', 0.25)
    allow_short = params.get('allow_short', False)

    orders = {}
    signals = {}
    long_positions = []
    short_positions = []

    # Analyze each asset
    for symbol, data in market_data.items():
        if len(data) < min_periods:
            continue

        try:
            # Calculate ADX
            adx_data = TechnicalIndicators.adx(
                data['high'],
                data['low'],
                data['close'],
                period=adx_period
            )

            if len(adx_data) < 2:
                continue

            current_adx = adx_data['adx'].iloc[-1]
            plus_di = adx_data['plus_di'].iloc[-1]
            minus_di = adx_data['minus_di'].iloc[-1]

            # Check for strong trend
            strong_trend = current_adx > adx_threshold

            # Generate signals based on trend strength and direction
            if strong_trend:
                if plus_di > minus_di:
                    signals[symbol] = 'ADX_LONG_TREND'
                    long_positions.append(symbol)
                elif minus_di > plus_di:
                    if allow_short:
                        signals[symbol] = 'ADX_SHORT_TREND'
                        short_positions.append(symbol)
                    else:
                        signals[symbol] = 'ADX_BEARISH'
                else:
                    signals[symbol] = 'ADX_NEUTRAL'
            else:
                signals[symbol] = 'ADX_WEAK_TREND'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        weight_per_position = min(1.0 / total_positions, max_position_size)

        # Long orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

        # Short orders
        for symbol in short_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position
            }

    # Close positions not in active positions
    active_positions = long_positions + short_positions
    for symbol in market_data.keys():
        if symbol not in active_positions:
            if symbol in positions and hasattr(positions[symbol], 'quantity'):
                if positions[symbol].quantity != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['short_positions'] = short_positions
    context['total_positions'] = total_positions

    return orders
