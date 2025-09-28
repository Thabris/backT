"""
Risk calculation utilities for BackT

Additional risk calculation and monitoring tools.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from ..utils.logging_config import LoggerMixin


class RiskCalculator(LoggerMixin):
    """Additional risk calculation utilities"""

    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if returns.empty:
            return 0.0
        return np.percentile(returns, confidence_level * 100)

    @staticmethod
    def calculate_portfolio_volatility(
        weights: Dict[str, float],
        covariance_matrix: pd.DataFrame
    ) -> float:
        """Calculate portfolio volatility given weights and covariance matrix"""
        if not weights or covariance_matrix.empty:
            return 0.0

        # Convert weights to array
        symbols = list(weights.keys())
        weight_array = np.array([weights.get(symbol, 0.0) for symbol in symbols])

        # Ensure covariance matrix has the same symbols
        cov_symbols = covariance_matrix.index.tolist()
        aligned_weights = np.array([weights.get(symbol, 0.0) for symbol in cov_symbols])

        # Calculate portfolio variance
        portfolio_variance = np.dot(aligned_weights.T, np.dot(covariance_matrix.values, aligned_weights))
        return np.sqrt(portfolio_variance)