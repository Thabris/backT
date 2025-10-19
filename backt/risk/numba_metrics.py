"""
Numba-optimized metric calculations for maximum performance

These JIT-compiled functions are called by MetricsEngine for fast calculations.
On first run, functions compile to native machine code and are cached.
Subsequent calls (like in CPCV or grid optimization) use compiled versions.
"""

import numpy as np
from typing import Tuple

# Optional numba support
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create a no-op decorator if numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


@jit(nopython=True, cache=True)
def calculate_sharpe_ratio_fast(
    returns: np.ndarray,
    risk_free_rate: float,
    periods_per_year: float
) -> float:
    """
    Fast Sharpe ratio calculation using Numba JIT

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Excess returns
    excess_returns = returns - (risk_free_rate / periods_per_year)

    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns)

    if std_excess == 0:
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return sharpe


@jit(nopython=True, cache=True)
def calculate_sortino_ratio_fast(
    returns: np.ndarray,
    risk_free_rate: float,
    periods_per_year: float
) -> float:
    """
    Fast Sortino ratio calculation using Numba JIT

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    mean_return = np.mean(returns)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        return np.inf  # No downside risk

    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    downside_deviation = downside_std * np.sqrt(periods_per_year)
    sortino = (mean_return * periods_per_year - risk_free_rate) / downside_deviation

    return sortino


@jit(nopython=True, cache=True)
def calculate_max_drawdown_fast(equity: np.ndarray) -> Tuple[float, int, int]:
    """
    Fast maximum drawdown calculation using Numba JIT

    Args:
        equity: Array of equity values

    Returns:
        Tuple of (max_drawdown, max_dd_duration, num_drawdown_periods)
    """
    if len(equity) < 2:
        return 0.0, 0, 0

    # Calculate running max manually (np.maximum.accumulate not supported in Numba)
    running_max = np.zeros(len(equity))
    running_max[0] = equity[0]
    for i in range(1, len(equity)):
        running_max[i] = max(running_max[i-1], equity[i])

    # Calculate drawdown
    drawdown = np.zeros(len(equity))
    for i in range(len(equity)):
        if running_max[i] > 0:
            drawdown[i] = (equity[i] - running_max[i]) / running_max[i]
        else:
            drawdown[i] = 0.0

    max_dd = np.min(drawdown)

    # Calculate drawdown durations
    drawdown_periods_list = []
    current_period = 0

    for i in range(len(drawdown)):
        if drawdown[i] < 0:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods_list.append(current_period)
            current_period = 0

    # Add final period if still in drawdown
    if current_period > 0:
        drawdown_periods_list.append(current_period)

    max_dd_duration = max(drawdown_periods_list) if len(drawdown_periods_list) > 0 else 0
    num_dd_periods = len(drawdown_periods_list)

    return max_dd, max_dd_duration, num_dd_periods


@jit(nopython=True, cache=True)
def calculate_var_cvar_fast(
    returns: np.ndarray,
    confidence_level: float
) -> Tuple[float, float]:
    """
    Fast VaR and CVaR calculation using Numba JIT

    Args:
        returns: Array of period returns
        confidence_level: Confidence level (e.g., 0.95 for 95% VaR)

    Returns:
        Tuple of (VaR, CVaR)
    """
    if len(returns) < 2:
        return 0.0, 0.0

    # VaR is the percentile
    var = np.percentile(returns, (1 - confidence_level) * 100)

    # CVaR is the mean of returns below VaR
    tail_returns = returns[returns <= var]

    if len(tail_returns) == 0:
        cvar = var
    else:
        cvar = np.mean(tail_returns)

    return var, cvar


@jit(nopython=True, cache=True)
def calculate_return_stats_fast(
    equity: np.ndarray,
    periods_per_year: float
) -> Tuple[float, float, float, float]:
    """
    Fast calculation of basic return statistics using Numba JIT

    Args:
        equity: Array of equity values
        periods_per_year: Number of periods per year

    Returns:
        Tuple of (total_return, cagr, annualized_vol, best_day, worst_day)
    """
    if len(equity) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    initial = equity[0]
    final = equity[-1]

    # Total return
    total_return = (final - initial) / initial if initial > 0 else 0.0

    # CAGR
    years = len(equity) / periods_per_year
    if years > 0 and initial > 0 and final > 0:
        cagr = (final / initial) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Calculate period returns
    returns = np.zeros(len(equity) - 1)
    for i in range(1, len(equity)):
        if equity[i-1] > 0:
            returns[i-1] = (equity[i] - equity[i-1]) / equity[i-1]
        else:
            returns[i-1] = 0.0

    if len(returns) > 0:
        volatility = np.std(returns) * np.sqrt(periods_per_year)
        best_day = np.max(returns)
        worst_day = np.min(returns)
    else:
        volatility = 0.0
        best_day = 0.0
        worst_day = 0.0

    return total_return, cagr, volatility, best_day, worst_day


@jit(nopython=True, cache=True)
def calculate_calmar_ratio_fast(
    cagr: float,
    max_drawdown: float
) -> float:
    """
    Fast Calmar ratio calculation using Numba JIT

    Args:
        cagr: Compound annual growth rate
        max_drawdown: Maximum drawdown (negative value)

    Returns:
        Calmar ratio (CAGR / abs(max drawdown))
    """
    if max_drawdown >= 0:
        return np.inf

    return cagr / abs(max_drawdown)


@jit(nopython=True, cache=True)
def calculate_win_rate_fast(pnl_array: np.ndarray) -> Tuple[float, float, int, int]:
    """
    Fast win rate and profit factor calculation using Numba JIT

    Args:
        pnl_array: Array of trade PnLs

    Returns:
        Tuple of (win_rate, profit_factor, num_wins, num_losses)
    """
    if len(pnl_array) == 0:
        return 0.0, 0.0, 0, 0

    num_wins = 0
    num_losses = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for pnl in pnl_array:
        if pnl > 0:
            num_wins += 1
            gross_profit += pnl
        elif pnl < 0:
            num_losses += 1
            gross_loss += abs(pnl)

    total_trades = len(pnl_array)
    win_rate = num_wins / total_trades if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

    return win_rate, profit_factor, num_wins, num_losses


# Vectorized versions for batch processing
@jit(nopython=True, cache=True, parallel=True)
def calculate_rolling_sharpe_fast(
    returns: np.ndarray,
    window: int,
    risk_free_rate: float,
    periods_per_year: float
) -> np.ndarray:
    """
    Fast rolling Sharpe ratio calculation using Numba JIT with parallelization

    Args:
        returns: Array of period returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Array of rolling Sharpe ratios
    """
    n = len(returns)
    result = np.zeros(n)

    for i in range(window, n):
        window_returns = returns[i-window:i]
        result[i] = calculate_sharpe_ratio_fast(window_returns, risk_free_rate, periods_per_year)

    return result
