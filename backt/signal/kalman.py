"""
Kalman Filter implementations for trading signals

Provides efficient Kalman filtering for price series denoising and
trend estimation in algorithmic trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


class KalmanFilter1D:
    """
    1D Kalman filter for univariate time series (e.g., price)

    Uses a simple constant model: x(t) = x(t-1) + w(t)
    where w(t) ~ N(0, Q) is the process noise.

    The filter estimates the true underlying value by balancing:
    - Model predictions (with process noise Q)
    - Observations (with measurement noise R)

    Parameters:
    -----------
    Q : float
        Process noise covariance. Controls how much the state can change.
        - Higher Q (e.g., 0.1): Filter adapts quickly, more responsive
        - Lower Q (e.g., 0.001): Filter assumes stable model, smoother
        Typical range: 0.0001 to 0.1

    R : float
        Measurement noise covariance. Controls trust in observations.
        - Higher R (e.g., 5.0): Assumes noisy data, more smoothing
        - Lower R (e.g., 0.1): Trusts observations, follows closely
        Typical range: 0.01 to 10.0

    initial_value : float, optional
        Initial state estimate. If None, uses first observation.

    initial_error : float
        Initial error covariance. Default 1.0.

    Attributes:
    -----------
    x : float
        Current state estimate (filtered value)
    P : float
        Current error covariance
    Q : float
        Process noise covariance
    R : float
        Measurement noise covariance

    Example:
    --------
    >>> kf = KalmanFilter1D(Q=0.001, R=1.0)
    >>> filtered_values = [kf.update(price) for price in prices]

    Notes:
    ------
    For moving average replacement:
    - Fast MA equivalent: Q=0.01, R=1.0
    - Slow MA equivalent: Q=0.001, R=1.0
    """

    def __init__(
        self,
        Q: float = 0.001,
        R: float = 1.0,
        initial_value: Optional[float] = None,
        initial_error: float = 1.0
    ):
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance

        # State
        self.x = initial_value if initial_value is not None else 0.0
        self.P = initial_error  # Error covariance

        # Track if initialized
        self._initialized = initial_value is not None

    def update(self, measurement: float) -> float:
        """
        Update the filter with a new measurement

        Parameters:
        -----------
        measurement : float
            New observed value (e.g., price)

        Returns:
        --------
        float
            Filtered estimate of the true value
        """
        # Initialize on first measurement
        if not self._initialized:
            self.x = measurement
            self._initialized = True
            return self.x

        # Prediction step
        x_pred = self.x  # State prediction (constant model)
        P_pred = self.P + self.Q  # Error covariance prediction

        # Update step
        # Kalman gain: how much to trust the measurement vs. the prediction
        K = P_pred / (P_pred + self.R)

        # Update state estimate
        innovation = measurement - x_pred  # Measurement residual
        self.x = x_pred + K * innovation

        # Update error covariance
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self, value: Optional[float] = None, error: float = 1.0):
        """
        Reset the filter state

        Parameters:
        -----------
        value : float, optional
            New initial state value. If None, marks as uninitialized.
        error : float
            New initial error covariance
        """
        self.x = value if value is not None else 0.0
        self.P = error
        self._initialized = value is not None


class KalmanFilter2D:
    """
    2D Kalman filter for price and velocity estimation

    State: x = [price, velocity]

    This filter estimates both the price level and its rate of change
    (velocity/momentum) simultaneously. Useful for trend-following.

    Model:
    ------
    price(t) = price(t-1) + velocity(t-1) * dt + w_price
    velocity(t) = velocity(t-1) + w_velocity

    Parameters:
    -----------
    Q_price : float
        Process noise for price. Typical: 0.001
    Q_velocity : float
        Process noise for velocity. Typical: 0.0001
    R : float
        Measurement noise. Typical: 1.0
    dt : float
        Time step. Default 1.0 (daily data)

    Attributes:
    -----------
    x : np.ndarray, shape (2,)
        State estimate [price, velocity]
    P : np.ndarray, shape (2, 2)
        Error covariance matrix

    Example:
    --------
    >>> kf = KalmanFilter2D(Q_price=0.001, Q_velocity=0.0001, R=1.0)
    >>> for price in prices:
    ...     estimated_price, estimated_velocity = kf.update(price)
    ...     # Use velocity as momentum indicator
    """

    def __init__(
        self,
        Q_price: float = 0.001,
        Q_velocity: float = 0.0001,
        R: float = 1.0,
        dt: float = 1.0,
        initial_state: Optional[np.ndarray] = None
    ):
        self.dt = dt

        # State transition matrix (constant velocity model)
        self.A = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ])

        # Observation matrix (we only observe price)
        self.H = np.array([[1.0, 0.0]])

        # Process noise covariance
        self.Q = np.array([
            [Q_price, 0.0],
            [0.0, Q_velocity]
        ])

        # Measurement noise covariance
        self.R = np.array([[R]])

        # State estimate [price, velocity]
        if initial_state is not None:
            self.x = initial_state.copy()
            self._initialized = True
        else:
            self.x = np.zeros(2)
            self._initialized = False

        # Error covariance
        self.P = np.eye(2)

    def update(self, measurement: float) -> tuple:
        """
        Update filter with new price measurement

        Parameters:
        -----------
        measurement : float
            Observed price

        Returns:
        --------
        tuple : (price, velocity)
            Filtered price estimate and velocity estimate
        """
        # Initialize on first measurement
        if not self._initialized:
            self.x[0] = measurement
            self.x[1] = 0.0
            self._initialized = True
            return self.x[0], self.x[1]

        # Prediction step
        x_pred = self.A @ self.x
        P_pred = self.A @ self.P @ self.A.T + self.Q

        # Update step
        # Innovation
        y = measurement - (self.H @ x_pred)[0]

        # Innovation covariance
        S = (self.H @ P_pred @ self.H.T + self.R)[0, 0]

        # Kalman gain
        K = P_pred @ self.H.T / S

        # Update state
        self.x = x_pred + K.flatten() * y

        # Update covariance
        self.P = (np.eye(2) - np.outer(K, self.H)) @ P_pred

        return self.x[0], self.x[1]

    def reset(self, state: Optional[np.ndarray] = None):
        """Reset filter state"""
        if state is not None:
            self.x = state.copy()
            self._initialized = True
        else:
            self.x = np.zeros(2)
            self._initialized = False
        self.P = np.eye(2)


# Helper functions for common use cases

def kalman_filter_series(
    data: Union[pd.Series, np.ndarray],
    Q: float = 0.001,
    R: float = 1.0
) -> Union[pd.Series, np.ndarray]:
    """
    Apply Kalman filter to entire series at once

    Parameters:
    -----------
    data : pd.Series or np.ndarray
        Input time series (e.g., prices)
    Q : float
        Process noise covariance
    R : float
        Measurement noise covariance

    Returns:
    --------
    pd.Series or np.ndarray
        Filtered series (same type as input)

    Example:
    --------
    >>> prices = pd.Series([100, 102, 101, 103, 105])
    >>> filtered = kalman_filter_series(prices, Q=0.001, R=1.0)
    """
    kf = KalmanFilter1D(Q=Q, R=R)

    if isinstance(data, pd.Series):
        filtered = data.copy()
        for i in range(len(data)):
            filtered.iloc[i] = kf.update(data.iloc[i])
        return filtered
    else:
        filtered = np.zeros_like(data)
        for i in range(len(data)):
            filtered[i] = kf.update(data[i])
        return filtered


def kalman_moving_average(
    data: pd.Series,
    Q: float = 0.001,
    R: float = 1.0,
    name: Optional[str] = None
) -> pd.Series:
    """
    Kalman filter as adaptive moving average

    Replaces traditional moving averages with Kalman-filtered version.
    No lookback window needed, adapts in real-time.

    Parameters:
    -----------
    data : pd.Series
        Price series with datetime index
    Q : float
        Process noise. Smaller = smoother (like longer MA period)
        - Q=0.0001: Very smooth (like 200-day MA)
        - Q=0.001: Smooth (like 50-day MA)
        - Q=0.01: Moderate (like 20-day MA)
        - Q=0.1: Responsive (like 5-day MA)
    R : float
        Measurement noise. Usually 1.0.
    name : str, optional
        Name for output series

    Returns:
    --------
    pd.Series
        Kalman-filtered moving average

    Example:
    --------
    >>> df['kf_fast'] = kalman_moving_average(df['close'], Q=0.01, R=1.0)
    >>> df['kf_slow'] = kalman_moving_average(df['close'], Q=0.001, R=1.0)
    >>> df['signal'] = df['kf_fast'] > df['kf_slow']
    """
    result = kalman_filter_series(data, Q=Q, R=R)

    if name:
        result.name = name
    else:
        result.name = f'kalman_Q{Q}_R{R}'

    return result


def kalman_crossover_signals(
    data: pd.Series,
    Q_fast: float = 0.01,
    Q_slow: float = 0.001,
    R: float = 1.0
) -> pd.DataFrame:
    """
    Generate crossover signals using two Kalman filters

    Similar to moving average crossover but with adaptive Kalman filters.

    Parameters:
    -----------
    data : pd.Series
        Price series
    Q_fast : float
        Process noise for fast filter (more responsive)
    Q_slow : float
        Process noise for slow filter (smoother)
    R : float
        Measurement noise for both filters

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: ['fast', 'slow', 'signal']
        signal: 1 (bullish), -1 (bearish), 0 (neutral)

    Example:
    --------
    >>> signals = kalman_crossover_signals(df['close'])
    >>> df['buy'] = signals['signal'] == 1
    >>> df['sell'] = signals['signal'] == -1
    """
    kf_fast = kalman_moving_average(data, Q=Q_fast, R=R, name='fast')
    kf_slow = kalman_moving_average(data, Q=Q_slow, R=R, name='slow')

    df = pd.DataFrame({
        'fast': kf_fast,
        'slow': kf_slow
    }, index=data.index)

    # Generate signals
    df['signal'] = 0
    df.loc[df['fast'] > df['slow'], 'signal'] = 1  # Bullish
    df.loc[df['fast'] < df['slow'], 'signal'] = -1  # Bearish

    return df


def kalman_with_velocity(
    data: pd.Series,
    Q_price: float = 0.001,
    Q_velocity: float = 0.0001,
    R: float = 1.0,
    dt: float = 1.0
) -> pd.DataFrame:
    """
    Estimate both price and velocity using 2D Kalman filter

    Parameters:
    -----------
    data : pd.Series
        Price series
    Q_price, Q_velocity, R, dt : float
        Kalman filter parameters

    Returns:
    --------
    pd.DataFrame
        Columns: ['price', 'velocity']

    Example:
    --------
    >>> result = kalman_with_velocity(df['close'])
    >>> df['kf_price'] = result['price']
    >>> df['kf_velocity'] = result['velocity']
    >>> df['momentum_signal'] = result['velocity'] > 0
    """
    kf = KalmanFilter2D(
        Q_price=Q_price,
        Q_velocity=Q_velocity,
        R=R,
        dt=dt
    )

    prices = np.zeros(len(data))
    velocities = np.zeros(len(data))

    for i, price in enumerate(data):
        p, v = kf.update(price)
        prices[i] = p
        velocities[i] = v

    return pd.DataFrame({
        'price': prices,
        'velocity': velocities
    }, index=data.index)
