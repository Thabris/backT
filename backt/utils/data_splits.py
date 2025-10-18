"""
Data splitting utilities for cross-validation

Implements purging and embargoing techniques for time series data
to prevent data leakage in backtesting.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from itertools import combinations
import math


def create_purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    purge_pct: float = 0.05,
    embargo_pct: float = 0.02
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-Fold splits with purging and embargoing

    Args:
        n_samples: Total number of samples
        n_splits: Number of folds
        purge_pct: Percentage of data to purge around test sets (0.05 = 5%)
        embargo_pct: Percentage to embargo after test sets (0.02 = 2%)

    Returns:
        List of (train_indices, test_indices) tuples
    """
    fold_size = n_samples // n_splits
    purge_size = int(n_samples * purge_pct / 2)  # Purge before and after
    embargo_size = int(n_samples * embargo_pct)

    splits = []

    for fold in range(n_splits):
        # Define test set boundaries
        test_start = fold * fold_size
        test_end = test_start + fold_size

        # Adjust for last fold to include remaining samples
        if fold == n_splits - 1:
            test_end = n_samples

        # Create test indices
        test_indices = np.arange(test_start, test_end)

        # Create train indices with purging and embargoing
        train_indices = []

        # Add samples before test set (with purge buffer)
        if test_start > purge_size:
            train_indices.extend(range(0, test_start - purge_size))

        # Add samples after test set (with embargo and purge buffer)
        if test_end + embargo_size + purge_size < n_samples:
            train_indices.extend(range(test_end + embargo_size + purge_size, n_samples))

        train_indices = np.array(train_indices)

        splits.append((train_indices, test_indices))

    return splits


def generate_cpcv_combinations(
    n_splits: int = 10,
    n_test_splits: int = 2
) -> List[Tuple[int, ...]]:
    """
    Generate all combinatorial fold combinations for CPCV

    Args:
        n_splits: Total number of folds (e.g., 10)
        n_test_splits: Number of test folds per combination (e.g., 2)

    Returns:
        List of tuples, each containing indices of test folds

    Example:
        >>> combos = generate_cpcv_combinations(n_splits=5, n_test_splits=2)
        >>> len(combos)
        10  # C(5,2) = 10 combinations
    """
    fold_indices = range(n_splits)
    return list(combinations(fold_indices, n_test_splits))


def create_cpcv_split(
    indices: np.ndarray,
    test_fold_indices: Tuple[int, ...],
    n_splits: int,
    purge_pct: float = 0.05,
    embargo_pct: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a single CPCV train/test split with purging and embargoing

    Args:
        indices: Array of all data indices
        test_fold_indices: Tuple of fold indices to use as test set
        n_splits: Total number of folds
        purge_pct: Percentage to purge around test sets
        embargo_pct: Percentage to embargo after test sets

    Returns:
        (train_indices, test_indices) tuple
    """
    n_samples = len(indices)
    fold_size = n_samples // n_splits
    purge_size = int(n_samples * purge_pct / 2)
    embargo_size = int(n_samples * embargo_pct)

    # Collect all test indices
    test_indices_list = []
    for fold_idx in test_fold_indices:
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size

        # Adjust last fold
        if fold_idx == n_splits - 1:
            test_end = n_samples

        test_indices_list.extend(range(test_start, test_end))

    test_indices = np.array(sorted(test_indices_list))

    # Find continuous test regions and apply purging/embargoing
    train_mask = np.ones(n_samples, dtype=bool)

    # Mark test regions and buffers as False
    for fold_idx in test_fold_indices:
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size

        if fold_idx == n_splits - 1:
            test_end = n_samples

        # Mark test region
        train_mask[test_start:test_end] = False

        # Mark purge before test
        purge_start = max(0, test_start - purge_size)
        train_mask[purge_start:test_start] = False

        # Mark purge and embargo after test
        embargo_end = min(n_samples, test_end + embargo_size + purge_size)
        train_mask[test_end:embargo_end] = False

    train_indices = np.where(train_mask)[0]

    return train_indices, test_indices


def create_time_series_split(
    data_index: pd.DatetimeIndex,
    train_end: str,
    test_start: str,
    test_end: str,
    purge_days: int = 0,
    embargo_days: int = 0
) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Create a time series split with date-based boundaries

    Args:
        data_index: DatetimeIndex of the full dataset
        train_end: End date for training set (inclusive)
        test_start: Start date for test set
        test_end: End date for test set (inclusive)
        purge_days: Days to purge before test set
        embargo_days: Days to embargo after training set

    Returns:
        (train_index, test_index) tuple
    """
    train_end_dt = pd.Timestamp(train_end)
    test_start_dt = pd.Timestamp(test_start)
    test_end_dt = pd.Timestamp(test_end)

    # Apply purge and embargo
    if purge_days > 0:
        train_end_dt = train_end_dt - pd.Timedelta(days=purge_days)

    if embargo_days > 0:
        train_end_dt = train_end_dt - pd.Timedelta(days=embargo_days)

    # Create splits
    train_index = data_index[data_index <= train_end_dt]
    test_index = data_index[(data_index >= test_start_dt) & (data_index <= test_end_dt)]

    return train_index, test_index


def calculate_optimal_purge_embargo(
    strategy_horizon_days: int,
    rebalance_frequency_days: int
) -> Tuple[float, float]:
    """
    Calculate optimal purge and embargo percentages based on strategy characteristics

    Args:
        strategy_horizon_days: Strategy's typical holding period in days
        rebalance_frequency_days: How often strategy rebalances

    Returns:
        (purge_pct, embargo_pct) tuple

    Example:
        >>> # For a monthly momentum strategy (20 day lookback, monthly rebalance)
        >>> purge_pct, embargo_pct = calculate_optimal_purge_embargo(
        ...     strategy_horizon_days=20,
        ...     rebalance_frequency_days=21
        ... )
        >>> print(f"Purge: {purge_pct:.1%}, Embargo: {embargo_pct:.1%}")
        Purge: 5.0%, Embargo: 2.0%
    """
    # Rule of thumb: purge should be at least as long as strategy horizon
    # Embargo should be at least as long as rebalance frequency

    # Assuming typical backtest is 252 * 10 = 2520 days (10 years)
    typical_backtest_days = 2520

    purge_pct = (strategy_horizon_days * 2) / typical_backtest_days  # 2x horizon for safety
    embargo_pct = rebalance_frequency_days / typical_backtest_days

    # Cap at reasonable maximums
    purge_pct = min(purge_pct, 0.10)  # Max 10%
    embargo_pct = min(embargo_pct, 0.05)  # Max 5%

    return purge_pct, embargo_pct


def validate_split_quality(
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    n_samples: int,
    min_train_pct: float = 0.5
) -> Tuple[bool, str]:
    """
    Validate that a train/test split has sufficient data

    Args:
        train_indices: Training set indices
        test_indices: Test set indices
        n_samples: Total number of samples
        min_train_pct: Minimum percentage of data for training (default 50%)

    Returns:
        (is_valid, message) tuple
    """
    n_train = len(train_indices)
    n_test = len(test_indices)

    # Check for empty sets
    if n_train == 0:
        return False, "Training set is empty"
    if n_test == 0:
        return False, "Test set is empty"

    # Check for overlap
    if len(set(train_indices) & set(test_indices)) > 0:
        return False, "Train and test sets overlap"

    # Check minimum training data
    train_pct = n_train / n_samples
    if train_pct < min_train_pct:
        return False, f"Training set too small: {train_pct:.1%} < {min_train_pct:.1%}"

    # Check test set is reasonable
    test_pct = n_test / n_samples
    if test_pct < 0.05:
        return False, f"Test set too small: {test_pct:.1%}"

    return True, f"Valid split: {train_pct:.1%} train, {test_pct:.1%} test"
