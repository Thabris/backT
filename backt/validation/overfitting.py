"""
Overfitting detection metrics for backtesting

Implements:
- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio (DSR)
- Performance degradation analysis
- Stability metrics across validation paths

Based on research by Marcos Lopez de Prado in "Advances in Financial Machine Learning"
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class OverfittingMetrics:
    """Container for overfitting detection metrics"""
    pbo: float  # Probability of Backtest Overfitting
    dsr: float  # Deflated Sharpe Ratio
    degradation_pct: float  # IS to OOS performance degradation
    sharpe_stability: float  # Consistency of Sharpe across paths
    is_mean_sharpe: float  # Mean in-sample Sharpe
    oos_mean_sharpe: float  # Mean out-of-sample Sharpe
    n_trials: int  # Number of trials/parameter sets tested


def calculate_probability_backtest_overfitting(
    is_performance: np.ndarray,
    oos_performance: np.ndarray,
    metric_name: str = "sharpe_ratio"
) -> float:
    """
    Calculate Probability of Backtest Overfitting (PBO)

    PBO measures the probability that the best in-sample strategy's performance
    is due to overfitting rather than skill.

    Args:
        is_performance: In-sample performance metrics for each parameter set
        oos_performance: Out-of-sample performance for same parameter sets
        metric_name: Name of the metric being evaluated

    Returns:
        PBO value between 0 and 1 (lower is better)
        - PBO < 0.5: Low probability of overfitting (good)
        - PBO > 0.7: High probability of overfitting (bad)

    Formula:
        PBO = P[OOS_performance < median(OOS_performance) | IS_performance = max(IS_performance)]

    Reference:
        Bailey, D.H. and Lopez de Prado, M., 2014. The probability of backtest overfitting.
        Journal of Computational Finance, 20(4), pp.39-69.
    """
    if len(is_performance) != len(oos_performance):
        raise ValueError("IS and OOS performance arrays must have same length")

    if len(is_performance) < 2:
        raise ValueError("Need at least 2 parameter sets to calculate PBO")

    # Rank parameter sets by IS performance
    is_ranks = stats.rankdata(is_performance)

    # Get the logits (IS rank vs OOS performance relationship)
    n = len(is_performance)

    # Calculate median OOS performance
    oos_median = np.median(oos_performance)

    # For each IS rank, check if corresponding OOS is below median
    below_median = oos_performance < oos_median

    # Calculate PBO using the ranking method
    # This counts how many of the top IS performers have below-median OOS performance
    n_top_half = n // 2
    top_is_indices = np.argsort(is_performance)[-n_top_half:]
    n_top_below_median = np.sum(below_median[top_is_indices])

    pbo = n_top_below_median / n_top_half

    return pbo


def calculate_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    returns_skewness: float = 0.0,
    returns_kurtosis: float = 3.0,
    n_observations: int = 252
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR)

    DSR adjusts the Sharpe ratio for the number of trials tested and
    statistical properties of returns, providing a more conservative estimate.

    Args:
        observed_sharpe: Best Sharpe ratio observed
        n_trials: Number of parameter sets / strategies tested
        returns_skewness: Skewness of returns distribution (0 = normal)
        returns_kurtosis: Kurtosis of returns (3 = normal, excess = 0)
        n_observations: Number of return observations (e.g., 252 for daily over 1 year)

    Returns:
        DSR value (typically < observed Sharpe)
        - DSR > 2.0: Strong statistical significance
        - DSR > 1.0: Statistically significant at 84% confidence
        - DSR < 1.0: Not statistically significant

    Reference:
        Bailey, D.H. and Lopez de Prado, M., 2014. The deflated Sharpe ratio:
        correcting for selection bias, backtest overfitting, and non-normality.
        Journal of Portfolio Management, 40(5), pp.94-107.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    if n_observations < 1:
        raise ValueError("n_observations must be at least 1")

    # Expected maximum Sharpe under null hypothesis (no skill)
    # This is the Sharpe you'd expect to see by chance given n_trials
    expected_max_sharpe = _calculate_expected_max_sharpe(n_trials, n_observations)

    # Variance of the estimated Sharpe ratio
    # Adjusted for non-normality (skewness and kurtosis)
    excess_kurtosis = returns_kurtosis - 3.0

    sharpe_variance = (
        1 + (observed_sharpe ** 2) / 2
        - observed_sharpe * returns_skewness
        + (observed_sharpe ** 2) * excess_kurtosis / 4
    ) / (n_observations - 1)

    sharpe_std = np.sqrt(sharpe_variance)

    # Calculate DSR
    dsr = (observed_sharpe - expected_max_sharpe) / sharpe_std

    return dsr


def _calculate_expected_max_sharpe(n_trials: int, n_observations: int) -> float:
    """
    Calculate expected maximum Sharpe ratio under null hypothesis

    Uses Euler-Mascheroni constant approximation for expected value
    of maximum of n_trials standard normal variables.

    Args:
        n_trials: Number of trials (strategies tested)
        n_observations: Number of observations

    Returns:
        Expected maximum Sharpe ratio by chance
    """
    # Euler-Mascheroni constant
    euler_gamma = 0.5772156649

    # Expected value of max of n independent standard normals
    # E[max(Z1,...,Zn)] ≈ sqrt(2 * ln(n) - ln(ln(n)) - ln(4π)) + γ / sqrt(2*ln(n))

    if n_trials == 1:
        return 0.0

    ln_n = np.log(n_trials)
    ln_ln_n = np.log(ln_n)
    ln_4pi = np.log(4 * np.pi)

    expected_max = np.sqrt(2 * ln_n - ln_ln_n - ln_4pi) + euler_gamma / np.sqrt(2 * ln_n)

    # Adjust for number of observations
    # The Sharpe ratio scales with sqrt(T)
    adjusted = expected_max / np.sqrt(n_observations)

    return adjusted


def calculate_performance_degradation(
    is_metrics: Dict[str, float],
    oos_metrics: Dict[str, float],
    primary_metric: str = "sharpe_ratio"
) -> Dict[str, float]:
    """
    Calculate performance degradation from in-sample to out-of-sample

    Args:
        is_metrics: In-sample performance metrics
        oos_metrics: Out-of-sample performance metrics
        primary_metric: Primary metric to focus on

    Returns:
        Dictionary with degradation percentages for each metric
    """
    degradation = {}

    for metric_name in is_metrics.keys():
        if metric_name in oos_metrics:
            is_value = is_metrics[metric_name]
            oos_value = oos_metrics[metric_name]

            if is_value != 0:
                # Calculate percentage degradation
                deg_pct = ((is_value - oos_value) / abs(is_value)) * 100
                degradation[metric_name] = deg_pct
            else:
                degradation[metric_name] = np.nan

    return degradation


def calculate_sharpe_stability(sharpe_values: np.ndarray) -> float:
    """
    Calculate stability metric for Sharpe ratios across validation paths

    Stability is measured as inverse of coefficient of variation.
    Higher values indicate more consistent performance.

    Args:
        sharpe_values: Array of Sharpe ratios from different validation paths

    Returns:
        Stability score (higher is better)
        - > 5.0: Very stable
        - 2.0-5.0: Stable
        - < 2.0: Unstable
    """
    if len(sharpe_values) < 2:
        return np.inf  # Perfect stability with only one value

    mean_sharpe = np.mean(sharpe_values)
    std_sharpe = np.std(sharpe_values, ddof=1)

    if std_sharpe == 0:
        return np.inf  # Perfect stability (no variation)

    if mean_sharpe == 0:
        return 0.0  # Undefined stability

    # Coefficient of variation
    cv = std_sharpe / abs(mean_sharpe)

    # Return inverse CV as stability metric
    stability = 1.0 / cv

    return stability


def analyze_overfitting_comprehensive(
    is_sharpe_ratios: np.ndarray,
    oos_sharpe_ratios: np.ndarray,
    returns_skewness: float = 0.0,
    returns_kurtosis: float = 3.0,
    n_observations: int = 252
) -> OverfittingMetrics:
    """
    Perform comprehensive overfitting analysis

    Args:
        is_sharpe_ratios: In-sample Sharpe ratios for all parameter sets
        oos_sharpe_ratios: Out-of-sample Sharpe ratios for same parameter sets
        returns_skewness: Skewness of returns
        returns_kurtosis: Kurtosis of returns
        n_observations: Number of observations in each backtest

    Returns:
        OverfittingMetrics object with all metrics
    """
    n_trials = len(is_sharpe_ratios)

    # Calculate PBO
    pbo = calculate_probability_backtest_overfitting(
        is_sharpe_ratios, oos_sharpe_ratios
    )

    # Get best IS performance
    best_is_sharpe = np.max(is_sharpe_ratios)

    # Calculate DSR for best strategy
    dsr = calculate_deflated_sharpe_ratio(
        best_is_sharpe,
        n_trials,
        returns_skewness,
        returns_kurtosis,
        n_observations
    )

    # Calculate degradation
    is_mean = np.mean(is_sharpe_ratios)
    oos_mean = np.mean(oos_sharpe_ratios)

    if is_mean != 0:
        degradation_pct = ((is_mean - oos_mean) / abs(is_mean)) * 100
    else:
        degradation_pct = np.nan

    # Calculate stability
    sharpe_stability = calculate_sharpe_stability(oos_sharpe_ratios)

    return OverfittingMetrics(
        pbo=pbo,
        dsr=dsr,
        degradation_pct=degradation_pct,
        sharpe_stability=sharpe_stability,
        is_mean_sharpe=is_mean,
        oos_mean_sharpe=oos_mean,
        n_trials=n_trials
    )


def interpret_overfitting_metrics(metrics: OverfittingMetrics) -> Dict[str, str]:
    """
    Provide human-readable interpretation of overfitting metrics

    Args:
        metrics: OverfittingMetrics object

    Returns:
        Dictionary with interpretations for each metric
    """
    interpretations = {}

    # PBO interpretation
    if metrics.pbo < 0.3:
        interpretations['pbo'] = f"Excellent (PBO={metrics.pbo:.2%}): Very low overfitting risk"
    elif metrics.pbo < 0.5:
        interpretations['pbo'] = f"Good (PBO={metrics.pbo:.2%}): Low overfitting risk"
    elif metrics.pbo < 0.7:
        interpretations['pbo'] = f"Moderate (PBO={metrics.pbo:.2%}): Some overfitting risk"
    else:
        interpretations['pbo'] = f"Poor (PBO={metrics.pbo:.2%}): High overfitting risk"

    # DSR interpretation
    if metrics.dsr > 2.0:
        interpretations['dsr'] = f"Excellent (DSR={metrics.dsr:.2f}): Highly significant performance"
    elif metrics.dsr > 1.0:
        interpretations['dsr'] = f"Good (DSR={metrics.dsr:.2f}): Statistically significant"
    elif metrics.dsr > 0:
        interpretations['dsr'] = f"Weak (DSR={metrics.dsr:.2f}): Not statistically significant"
    else:
        interpretations['dsr'] = f"Poor (DSR={metrics.dsr:.2f}): Negative adjusted performance"

    # Degradation interpretation
    if np.isnan(metrics.degradation_pct):
        interpretations['degradation'] = "N/A: Cannot calculate"
    elif metrics.degradation_pct < 10:
        interpretations['degradation'] = f"Excellent ({metrics.degradation_pct:.1f}%): Minimal degradation"
    elif metrics.degradation_pct < 20:
        interpretations['degradation'] = f"Good ({metrics.degradation_pct:.1f}%): Acceptable degradation"
    elif metrics.degradation_pct < 40:
        interpretations['degradation'] = f"Moderate ({metrics.degradation_pct:.1f}%): Noticeable degradation"
    else:
        interpretations['degradation'] = f"Poor ({metrics.degradation_pct:.1f}%): Severe degradation"

    # Stability interpretation
    if metrics.sharpe_stability > 5.0:
        interpretations['stability'] = f"Excellent ({metrics.sharpe_stability:.1f}): Very consistent"
    elif metrics.sharpe_stability > 2.0:
        interpretations['stability'] = f"Good ({metrics.sharpe_stability:.1f}): Consistent performance"
    elif metrics.sharpe_stability > 1.0:
        interpretations['stability'] = f"Moderate ({metrics.sharpe_stability:.1f}): Some inconsistency"
    else:
        interpretations['stability'] = f"Poor ({metrics.sharpe_stability:.1f}): Highly inconsistent"

    return interpretations
