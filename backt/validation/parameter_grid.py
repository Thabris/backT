"""
Parameter grid utilities for strategy optimization

Provides tools for generating and managing parameter combinations
for grid search and random search optimization.
"""

from typing import Dict, List, Any, Iterator, Optional
from itertools import product
import numpy as np
from dataclasses import dataclass


@dataclass
class ParameterSet:
    """Single set of parameters with metadata"""
    params: Dict[str, Any]
    index: int
    hash_key: str


class ParameterGrid:
    """
    Generate all combinations of parameters for grid search

    Example:
        >>> grid = ParameterGrid({
        ...     'fast_ma': [10, 20, 30],
        ...     'slow_ma': [50, 100, 200],
        ...     'threshold': [0.0, 0.01]
        ... })
        >>> len(grid)
        18
        >>> for params in grid:
        ...     print(params)
    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize parameter grid

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       Example: {'fast_ma': [10, 20], 'slow_ma': [50, 100]}
        """
        self.param_grid = param_grid
        self.param_names = sorted(param_grid.keys())
        self.param_values = [param_grid[name] for name in self.param_names]
        self._size = None

    def __len__(self) -> int:
        """Return total number of parameter combinations"""
        if self._size is None:
            self._size = np.prod([len(values) for values in self.param_values])
        return int(self._size)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all parameter combinations"""
        for values in product(*self.param_values):
            yield dict(zip(self.param_names, values))

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get parameter set by index"""
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for grid of size {len(self)}")

        # Convert linear index to multi-dimensional index
        indices = []
        remaining = index
        for n_values in reversed([len(v) for v in self.param_values]):
            indices.append(remaining % n_values)
            remaining //= n_values

        indices.reverse()

        # Build parameter dict
        params = {}
        for name, idx in zip(self.param_names, indices):
            params[name] = self.param_grid[name][idx]

        return params

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of all parameter combinations"""
        return list(self)

    def to_dataframe(self):
        """Convert to pandas DataFrame (if pandas available)"""
        try:
            import pandas as pd
            return pd.DataFrame(list(self))
        except ImportError:
            raise ImportError("pandas required for to_dataframe()")

    def sample(self, n: int, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Randomly sample n parameter combinations

        Args:
            n: Number of combinations to sample
            random_state: Random seed for reproducibility

        Returns:
            List of parameter dictionaries
        """
        if n >= len(self):
            return self.to_list()

        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(self), size=n, replace=False)

        return [self[i] for i in indices]


class RandomParameterSampler:
    """
    Sample parameters from distributions for random search

    Example:
        >>> from scipy.stats import uniform, randint
        >>> sampler = RandomParameterSampler({
        ...     'fast_ma': randint(5, 50),
        ...     'slow_ma': randint(50, 200),
        ...     'threshold': uniform(0, 0.05)
        ... }, n_iter=100)
        >>> for params in sampler:
        ...     print(params)
    """

    def __init__(
        self,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize random parameter sampler

        Args:
            param_distributions: Dict of parameter names to scipy distributions
                                or lists of values
            n_iter: Number of parameter combinations to generate
            random_state: Random seed
        """
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def __len__(self) -> int:
        return self.n_iter

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Generate random parameter combinations"""
        for _ in range(self.n_iter):
            params = {}
            for name, distribution in self.param_distributions.items():
                if hasattr(distribution, 'rvs'):
                    # scipy distribution
                    params[name] = distribution.rvs(random_state=self.rng)
                elif isinstance(distribution, list):
                    # List of discrete values
                    params[name] = self.rng.choice(distribution)
                else:
                    raise ValueError(
                        f"Parameter {name} must be a list or scipy distribution"
                    )

            yield params


def validate_parameter_grid(param_grid: Dict[str, List[Any]]) -> None:
    """
    Validate parameter grid format

    Args:
        param_grid: Parameter grid to validate

    Raises:
        ValueError: If grid format is invalid
    """
    if not isinstance(param_grid, dict):
        raise ValueError("param_grid must be a dictionary")

    if not param_grid:
        raise ValueError("param_grid cannot be empty")

    for name, values in param_grid.items():
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise ValueError(
                f"Parameter '{name}' values must be a list, tuple, or array"
            )

        if len(values) == 0:
            raise ValueError(f"Parameter '{name}' has no values")


def estimate_grid_search_time(
    n_combinations: int,
    avg_backtest_time_seconds: float,
    n_cv_folds: int = 5
) -> Dict[str, float]:
    """
    Estimate total time for grid search with cross-validation

    Args:
        n_combinations: Number of parameter combinations
        avg_backtest_time_seconds: Average time for single backtest
        n_cv_folds: Number of cross-validation folds

    Returns:
        Dictionary with time estimates in various units
    """
    total_backtests = n_combinations * n_cv_folds
    total_seconds = total_backtests * avg_backtest_time_seconds

    return {
        'total_backtests': total_backtests,
        'seconds': total_seconds,
        'minutes': total_seconds / 60,
        'hours': total_seconds / 3600,
        'days': total_seconds / 86400,
        'estimated_completion': f"{total_seconds / 3600:.1f} hours"
    }
