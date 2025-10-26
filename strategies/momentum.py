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
from backt.signal import TechnicalIndicators, StrategyHelpers
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
    - Each symbol gets equal allocation (1/N) of portfolio value

    Parameters:
    -----------
    fast_ma : int, default=20
        Short-term moving average period
    slow_ma : int, default=50
        Long-term moving average period
    min_periods : int, default=60
        Minimum data points required

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_ma': 20, 'slow_ma': 50}
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

    orders = {}
    signals = {}

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

            # Store MA values for logging
            context.setdefault('ma_values', {})[symbol] = {
                'fast_ma': current_fast,
                'slow_ma': current_slow
            }

            # Generate signals - LONG ONLY with reasons
            if golden_cross:
                signals[symbol] = {
                    'action': 'BUY',
                    'reason': f'Golden cross: Fast MA ({current_fast:.2f}) crossed above Slow MA ({current_slow:.2f})'
                }
            elif death_cross:
                signals[symbol] = {
                    'action': 'SELL',
                    'reason': f'Death cross: Fast MA ({current_fast:.2f}) crossed below Slow MA ({current_slow:.2f})'
                }
                # Don't add to long_positions (go to cash)
            elif current_fast > current_slow:
                signals[symbol] = {
                    'action': 'HOLD_LONG',
                    'reason': f'Bullish trend: Fast MA ({current_fast:.2f}) > Slow MA ({current_slow:.2f})'
                }
            else:
                signals[symbol] = {
                    'action': 'NEUTRAL',
                    'reason': f'Bearish trend: Fast MA ({current_fast:.2f}) < Slow MA ({current_slow:.2f})'
                }
                # Stay in cash

        except Exception as e:
            continue

    # Calculate target weight ONCE (stored in context for consistency)
    if 'target_weights' not in context:
        # Calculate equal 1/N weight for each symbol
        # Each symbol gets equal share of portfolio: 100% / N symbols
        n_symbols = len(market_data.keys())
        if n_symbols > 0:
            weight_per_symbol = 1.0 / n_symbols
            context['target_weights'] = {symbol: weight_per_symbol for symbol in market_data.keys()}
        else:
            context['target_weights'] = {}

    # Create orders ONLY on signal changes (entry/exit), NOT on hold
    for symbol in market_data.keys():
        signal_info = signals.get(symbol, {})
        signal_action = signal_info.get('action', 'UNKNOWN')

        # Check current position safely
        if symbol in positions and positions[symbol] is not None:
            current_qty = positions[symbol].qty
        else:
            current_qty = 0.0

        has_position = abs(current_qty) > 1e-8

        # ENTRY SIGNAL - Buy on golden cross
        if signal_action == 'BUY' and not has_position:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': context['target_weights'].get(symbol, 0),
                'meta': {
                    'reason': "Enter long: " + signal_info.get('reason', 'Golden cross'),
                    'signal': signal_action,
                    'fast_ma': context['ma_values'][symbol]['fast_ma'],
                    'slow_ma': context['ma_values'][symbol]['slow_ma']
                }
            }

        # EXIT SIGNAL - Close on death cross
        elif signal_action == 'SELL' and has_position:
            orders[symbol] = {
                'action': 'close',
                'meta': {
                    'reason': "Exit long: " + signal_info.get('reason', 'Death cross'),
                    'signal': signal_action,
                    'fast_ma': context['ma_values'][symbol]['fast_ma'],
                    'slow_ma': context['ma_values'][symbol]['slow_ma']
                }
            }

        # HOLD SIGNAL - No order, let position drift naturally
        # (HOLD_LONG - do nothing)

    # Store strategy state for analysis
    context['signals'] = signals

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
    - Each symbol gets equal allocation (1/N) of portfolio value

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

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_ma': 20, 'slow_ma': 50}
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

    orders = {}
    signals = {}

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
            elif death_cross:
                signals[symbol] = 'SELL_SHORT'
                short_positions.append(symbol)
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
            else:
                signals[symbol] = 'HOLD_SHORT'
                short_positions.append(symbol)

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        # Equal weight allocation (1/N)
        weight_per_position = 1.0 / total_positions

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
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
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
    - Each symbol gets equal allocation (1/N) of portfolio value

    Key Advantages of Kalman Filters:
    - Superior noise reduction without lag
    - Adaptive to changing market conditions
    - No lookback window needed (real-time adaptation)
    - Smoother signals with fewer whipsaws

    Parameters:
    -----------
    Q_fast : float, default=0.014
        Process noise for fast filter (more responsive)
    Q_slow : float, default=0.0006
        Process noise for slow filter (smoother)
    R : float, default=1.0
        Measurement noise (standard)
    min_periods : int, default=60
        Minimum data points required

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'Q_fast': 0.014, 'Q_slow': 0.0006, 'R': 1.0}
    >>> result = backtester.run(
    ...     strategy=kalman_ma_crossover_long_only,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    Q_fast = params.get('Q_fast', 0.014)
    Q_slow = params.get('Q_slow', 0.0006)
    R = params.get('R', 1.0)
    min_periods = params.get('min_periods', 60)

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

            # Check current position
            has_position = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0

            # Signal-based trading: Only enter/exit on crossovers
            if golden_cross and not has_position:
                # Enter long on golden cross (only if not already long)
                signals[symbol] = 'BUY'
                long_positions.append(symbol)
            elif death_cross and has_position:
                # Exit on death cross
                signals[symbol] = 'SELL'
                # Don't add to long_positions - will be closed
            elif has_position:
                # Hold existing position - include in rebalancing for equal weights
                signals[symbol] = 'HOLD_LONG'
                long_positions.append(symbol)  # Include for multi-symbol rebalancing
            else:
                signals[symbol] = 'NEUTRAL'
                # Stay in cash

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        # Equal weight allocation (1/N)
        weight_per_position = 1.0 / total_positions

        # Create LONG orders
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions on death cross (explicit SELL signal)
    for symbol, signal in signals.items():
        if signal == 'SELL':
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
                    orders[symbol] = {'action': 'close'}

    # Store strategy state for analysis
    context['signals'] = signals

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
    - Each symbol gets equal allocation (1/N) of portfolio value

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

    # Initialize Kalman filters in context if not already present
    if 'kalman_filters' not in context:
        context['kalman_filters'] = {}

    orders = {}
    signals = {}

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
            elif death_cross:
                signals[symbol] = 'SELL_SHORT'
                short_positions.append(symbol)
            elif current_fast > current_slow:
                signals[symbol] = 'HOLD_LONG'
            else:
                signals[symbol] = 'HOLD_SHORT'
                short_positions.append(symbol)

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        # Equal weight allocation (1/N)
        weight_per_position = 1.0 / total_positions

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
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
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
    - Each symbol gets equal allocation (1/N) of portfolio value

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

            # Check current position
            has_position = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0

            # Signal-based trading: Only enter on oversold, exit on overbought
            if current_rsi < oversold and not has_position:
                # Enter position when oversold (only if not already holding)
                signals[symbol] = 'OVERSOLD_BUY'
                long_positions.append(symbol)
            elif current_rsi > overbought and has_position:
                # Exit position when overbought
                signals[symbol] = 'OVERBOUGHT_SELL'
                # Don't add to long_positions - will be closed
            elif has_position:
                # Hold position until overbought - include in rebalancing
                signals[symbol] = 'HOLD'
                long_positions.append(symbol)  # Include for multi-symbol rebalancing
            else:
                signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions

        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions on overbought (explicit SELL signal)
    for symbol, signal in signals.items():
        if signal == 'OVERBOUGHT_SELL':
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
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
    - Each symbol gets equal allocation (1/N) of portfolio value

    Parameters:
    -----------
    fast_period : int, default=11
        Fast EMA period for MACD
    slow_period : int, default=20
        Slow EMA period for MACD
    signal_period : int, default=10
        Signal line EMA period
    min_periods : int, default=35
        Minimum data points required
    allow_short : bool, default=False
        If True, go short on bearish crossover; if False, go to cash

    Returns:
    --------
    Dict[str, Dict]
        Dictionary of orders keyed by symbol

    Example:
    --------
    >>> params = {'fast_period': 11, 'slow_period': 20, 'signal_period': 10, 'allow_short': True}
    >>> result = backtester.run(
    ...     strategy=macd_crossover,
    ...     universe=['SPY', 'QQQ'],
    ...     strategy_params=params
    ... )
    """
    # Strategy parameters
    fast_period = params.get('fast_period', 11)
    slow_period = params.get('slow_period', 20)
    signal_period = params.get('signal_period', 10)
    min_periods = params.get('min_periods', 35)
    allow_short = params.get('allow_short', False)

    orders = {}
    signals = {}

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

            # Store signal values for logging
            context.setdefault('macd_values', {})[symbol] = {
                'macd': current_macd,
                'signal': current_signal,
                'histogram': macd_data['histogram'].iloc[-1]
            }

            # Generate signals with reasons
            if bullish_cross:
                signals[symbol] = {
                    'action': 'MACD_BUY',
                    'reason': f'MACD bullish crossover (MACD: {current_macd:.4f} > Signal: {current_signal:.4f})'
                }
            elif bearish_cross:
                if allow_short:
                    signals[symbol] = {
                        'action': 'MACD_SELL_SHORT',
                        'reason': f'MACD bearish crossover (MACD: {current_macd:.4f} < Signal: {current_signal:.4f})'
                    }
                else:
                    signals[symbol] = {
                        'action': 'MACD_SELL',
                        'reason': f'MACD bearish crossover (MACD: {current_macd:.4f} < Signal: {current_signal:.4f})'
                    }
            elif current_macd > current_signal:
                signals[symbol] = {
                    'action': 'MACD_HOLD_LONG',
                    'reason': f'MACD above signal (MACD: {current_macd:.4f} > Signal: {current_signal:.4f})'
                }
            elif current_macd < current_signal and allow_short:
                signals[symbol] = {
                    'action': 'MACD_HOLD_SHORT',
                    'reason': f'MACD below signal (MACD: {current_macd:.4f} < Signal: {current_signal:.4f})'
                }
                short_positions.append(symbol)
            else:
                signals[symbol] = {
                    'action': 'NEUTRAL',
                    'reason': 'No MACD signal'
                }

        except Exception as e:
            continue

    # Calculate target weight ONCE (stored in context for consistency)
    if 'target_weights' not in context:
        # Calculate equal 1/N weight for each symbol
        # Each symbol gets equal share of portfolio: 100% / N symbols
        n_symbols = len(market_data.keys())
        if n_symbols > 0:
            weight_per_symbol = 1.0 / n_symbols
            context['target_weights'] = {symbol: weight_per_symbol for symbol in market_data.keys()}
        else:
            context['target_weights'] = {}

        context['initial_allocation_done'] = True

    # Create orders ONLY on signal changes (entry/exit), NOT on hold
    for symbol in market_data.keys():
        signal_info = signals.get(symbol, {})
        if not isinstance(signal_info, dict):
            continue

        signal_action = signal_info.get('action', 'UNKNOWN')

        # Check current position safely
        if symbol in positions and positions[symbol] is not None:
            current_qty = positions[symbol].qty
        else:
            current_qty = 0.0

        has_position = abs(current_qty) > 1e-8

        # ENTRY SIGNALS - Buy/Short when crossing over
        if signal_action == 'MACD_BUY' and not has_position:
            # Enter long position using target_weight (scales with portfolio growth)
            orders[symbol] = {
                'action': 'target_weight',
                'weight': context['target_weights'].get(symbol, 0),
                'meta': {
                    'reason': "Enter long: " + signal_info.get('reason', 'MACD bullish'),
                    'signal': signal_action,
                    'macd': context['macd_values'][symbol]['macd'],
                    'signal_line': context['macd_values'][symbol]['signal'],
                    'histogram': context['macd_values'][symbol]['histogram']
                }
            }

        elif signal_action == 'MACD_SELL_SHORT' and allow_short and not has_position:
            # Enter short position using target_weight (negative weight for shorts)
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -context['target_weights'].get(symbol, 0),  # Negative for short
                'meta': {
                    'reason': "Enter short: " + signal_info.get('reason', 'MACD bearish'),
                    'signal': signal_action,
                    'macd': context['macd_values'][symbol]['macd'],
                    'signal_line': context['macd_values'][symbol]['signal'],
                    'histogram': context['macd_values'][symbol]['histogram']
                }
            }

        # EXIT SIGNALS - Close position when signal changes
        elif signal_action == 'MACD_SELL' and has_position and current_qty > 0:
            # Exit long position (long-only mode)
            orders[symbol] = {
                'action': 'close',
                'meta': {
                    'reason': "Exit long: " + signal_info.get('reason', 'MACD bearish crossover'),
                    'signal': signal_action,
                    'macd': context['macd_values'][symbol]['macd'],
                    'signal_line': context['macd_values'][symbol]['signal']
                }
            }

        elif signal_action == 'MACD_BUY' and has_position and current_qty < 0:
            # Close short and go long
            orders[symbol] = {
                'action': 'close',
                'meta': {
                    'reason': "Exit short: " + signal_info.get('reason', 'MACD bullish crossover'),
                    'signal': 'EXIT_SHORT',
                    'macd': context['macd_values'][symbol]['macd'],
                    'signal_line': context['macd_values'][symbol]['signal']
                }
            }
            # Will enter long on next iteration

        elif signal_action == 'MACD_SELL_SHORT' and has_position and current_qty > 0 and allow_short:
            # Close long and go short
            orders[symbol] = {
                'action': 'close',
                'meta': {
                    'reason': "Exit long: " + signal_info.get('reason', 'MACD bearish crossover'),
                    'signal': 'EXIT_LONG',
                    'macd': context['macd_values'][symbol]['macd'],
                    'signal_line': context['macd_values'][symbol]['signal']
                }
            }
            # Will enter short on next iteration

        elif signal_action == 'NEUTRAL' and has_position:
            # Exit on neutral signal
            orders[symbol] = {
                'action': 'close',
                'meta': {
                    'reason': "Exit position: " + signal_info.get('reason', 'No signal'),
                    'signal': 'EXIT'
                }
            }

        # HOLD SIGNALS - No order, let position drift naturally
        # (MACD_HOLD_LONG, MACD_HOLD_SHORT - do nothing)

    # Store strategy state
    context['signals'] = signals

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
    - Each symbol gets equal allocation (1/N) of portfolio value

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

            # Check current position
            has_position = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0

            # Signal-based trading: Only enter on bullish cross, exit on bearish cross
            if bullish_cross and not has_position:
                # Enter position on bullish crossover (only if not already holding)
                signals[symbol] = 'STOCH_OVERSOLD_BUY'
                long_positions.append(symbol)
            elif bearish_cross and has_position:
                # Exit position on bearish crossover
                signals[symbol] = 'STOCH_OVERBOUGHT_SELL'
                # Don't add to long_positions - will be closed
            elif has_position:
                # Hold position if still bullish - include in rebalancing
                signals[symbol] = 'STOCH_HOLD'
                long_positions.append(symbol)  # Include for multi-symbol rebalancing
            else:
                signals[symbol] = 'NEUTRAL'

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions

        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position
            }

    # Close positions on bearish crossover (explicit SELL signal)
    for symbol, signal in signals.items():
        if signal == 'STOCH_OVERBOUGHT_SELL':
            if symbol in positions and hasattr(positions[symbol], 'qty'):
                if positions[symbol].qty != 0:
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

    orders = {}
    signals = {}
    long_positions = []
    positions_to_close = []  # Track explicit EXIT signals

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

            # Calculate distance from bands (as percentage)
            lower_distance = (current_price - lower_band) / lower_band
            upper_distance = (upper_band - current_price) / upper_band
            middle_distance = abs(current_price - middle_band) / middle_band

            # Detect band touches/crosses
            # More lenient conditions to generate more trades
            touches_lower = current_price <= lower_band  # Price at or below lower band
            crosses_lower = prev_price > prev_lower and current_price <= lower_band  # Strong signal
            touches_upper = current_price >= upper_band  # Price at or above upper band
            near_middle = middle_distance < 0.01  # Within 1% of middle band

            # Generate signals with detailed information
            has_position = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0

            # Signal-based trading: Only trade on actual signals, not continuous rebalancing
            if (touches_lower or crosses_lower) and not has_position:
                # BUY signal - enter position (only if we don't already have one)
                signals[symbol] = {
                    'action': 'BB_LOWER_BUY',
                    'reason': f"Price ${current_price:.2f} at/below lower band ${lower_band:.2f} ({lower_distance:.2%} below)",
                    'lower_band': lower_band,
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'price': current_price,
                    'is_cross': crosses_lower
                }
                long_positions.append(symbol)
            elif touches_upper and has_position:
                # EXIT signal - price reached upper band (take profit)
                signals[symbol] = {
                    'action': 'BB_EXIT_UPPER',
                    'reason': f"Price ${current_price:.2f} at/above upper band ${upper_band:.2f} - exit overbought",
                    'lower_band': lower_band,
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'price': current_price
                }
                positions_to_close.append(symbol)  # Mark for explicit closing
            elif near_middle and has_position:
                # EXIT signal - price returned to middle (mean reversion complete)
                signals[symbol] = {
                    'action': 'BB_EXIT_MIDDLE',
                    'reason': f"Price ${current_price:.2f} near middle band ${middle_band:.2f} - mean reversion complete",
                    'lower_band': lower_band,
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'price': current_price
                }
                positions_to_close.append(symbol)  # Mark for explicit closing
            else:
                # No signal - either holding or neutral
                if has_position:
                    signals[symbol] = {
                        'action': 'BB_HOLD',
                        'reason': f"Holding position - price ${current_price:.2f} between bands",
                        'price': current_price
                    }
                    # NOTE: Do NOT add to long_positions here!
                    # This prevents continuous rebalancing. Position stays as-is until exit signal.
                else:
                    signals[symbol] = {
                        'action': 'NEUTRAL',
                        'reason': f"No signal - price ${current_price:.2f} between bands",
                        'price': current_price
                    }

        except Exception as e:
            continue

    # Calculate position sizing
    total_positions = len(long_positions)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions

        for symbol in long_positions:
            signal_info = signals.get(symbol, {})
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position,
                'meta': {
                    'reason': signal_info.get('reason', 'Bollinger Bands position'),
                    'signal': signal_info.get('action', 'BB_HOLD'),
                    'strategy': 'bollinger_mean_reversion',
                    'lower_band': signal_info.get('lower_band'),
                    'upper_band': signal_info.get('upper_band'),
                    'middle_band': signal_info.get('middle_band'),
                    'price': signal_info.get('price')
                }
            }

    # Close positions with explicit EXIT signals only
    for symbol in positions_to_close:
        if symbol in positions and hasattr(positions[symbol], 'qty'):
            if positions[symbol].qty != 0:
                signal_info = signals.get(symbol, {})
                orders[symbol] = {
                    'action': 'close',
                    'meta': {
                        'reason': signal_info.get('reason', 'Exit position'),
                        'signal': signal_info.get('action', 'EXIT'),
                        'strategy': 'bollinger_mean_reversion'
                    }
                }

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['positions_to_close'] = positions_to_close
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
    allow_short = params.get('allow_short', False)

    orders = {}
    signals = {}
    long_positions = []
    short_positions = []
    positions_to_close = []

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

            # Check current position
            has_long = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty > 0
            has_short = symbol in positions and hasattr(positions[symbol], 'qty') and positions[symbol].qty < 0

            # Signal-based trading: Only trade on NEW signals, not continuous rebalancing
            if strong_trend:
                if plus_di > minus_di:
                    # Bullish trend
                    if not has_long:
                        # Enter long position (only if we don't already have one)
                        signals[symbol] = 'ADX_LONG_ENTRY'
                        long_positions.append(symbol)
                        if has_short:
                            # Close short before going long
                            positions_to_close.append(symbol)
                    else:
                        # Already long - hold position, include in rebalancing
                        signals[symbol] = 'ADX_LONG_HOLD'
                        long_positions.append(symbol)  # Include for multi-symbol rebalancing
                elif minus_di > plus_di:
                    # Bearish trend
                    if allow_short:
                        if not has_short:
                            # Enter short position (only if we don't already have one)
                            signals[symbol] = 'ADX_SHORT_ENTRY'
                            short_positions.append(symbol)
                            if has_long:
                                # Close long before going short
                                positions_to_close.append(symbol)
                        else:
                            # Already short - hold position, include in rebalancing
                            signals[symbol] = 'ADX_SHORT_HOLD'
                            short_positions.append(symbol)  # Include for multi-symbol rebalancing
                    else:
                        # Bearish but shorts not allowed - close long if we have one
                        signals[symbol] = 'ADX_BEARISH_EXIT'
                        if has_long:
                            positions_to_close.append(symbol)
                else:
                    signals[symbol] = 'ADX_NEUTRAL'
            else:
                # Weak trend - exit all positions
                signals[symbol] = 'ADX_WEAK_TREND_EXIT'
                if has_long or has_short:
                    positions_to_close.append(symbol)

        except Exception as e:
            continue

    # Calculate position sizing for NEW positions only
    total_positions = len(long_positions) + len(short_positions)

    if total_positions > 0:
        weight_per_position = 1.0 / total_positions

        # Long orders (only for NEW positions)
        for symbol in long_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': weight_per_position,
                'meta': {
                    'signal': signals.get(symbol, 'ADX_LONG'),
                    'strategy': 'adx_trend_filter'
                }
            }

        # Short orders (only for NEW positions)
        for symbol in short_positions:
            orders[symbol] = {
                'action': 'target_weight',
                'weight': -weight_per_position,
                'meta': {
                    'signal': signals.get(symbol, 'ADX_SHORT'),
                    'strategy': 'adx_trend_filter'
                }
            }

    # Close positions with explicit EXIT signals only
    for symbol in positions_to_close:
        if symbol in positions and hasattr(positions[symbol], 'qty'):
            if positions[symbol].qty != 0:
                orders[symbol] = {
                    'action': 'close',
                    'meta': {
                        'signal': signals.get(symbol, 'ADX_EXIT'),
                        'strategy': 'adx_trend_filter'
                    }
                }

    # Store strategy state
    context['signals'] = signals
    context['long_positions'] = long_positions
    context['short_positions'] = short_positions
    context['positions_to_close'] = positions_to_close

    return orders
