"""
Momentum-based Trading Strategies

Collection of moving average crossover strategies including:
- Traditional MA crossovers (long-only and long-short)
- Kalman-enhanced MA crossovers (long-only and long-short)

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
