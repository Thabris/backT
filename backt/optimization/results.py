"""
Result classes for parameter optimization
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class ParameterSetResult:
    """Results from evaluating a single parameter set"""

    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    backtest_time_seconds: float
    evaluation_id: int

    # Optional CPCV validation results
    cpcv_result: Optional[Any] = None  # CPCVResult if validated

    def get_metric(self, metric_name: str) -> float:
        """Get specific metric value"""
        return self.metrics.get(metric_name, np.nan)

    def __repr__(self):
        sharpe = self.metrics.get('sharpe_ratio', 0.0)
        return f"ParameterSetResult(params={self.parameters}, sharpe={sharpe:.2f})"


@dataclass
class OptimizationResult:
    """Complete optimization results"""

    # Best parameters found
    best_parameters: Dict[str, Any]
    best_metrics: Dict[str, float]

    # All evaluated parameter sets
    all_results: List[ParameterSetResult]

    # Optimization metadata
    method: str  # 'flaml' or 'grid'
    optimization_metric: str
    total_evaluations: int
    total_time_seconds: float
    start_time: datetime
    end_time: datetime

    # Search space
    param_space: Dict[str, Any]

    # CPCV validation (if performed)
    cpcv_validated: bool = False
    top_k_cpcv_results: Optional[List[Any]] = None

    def get_top_k(self, k: int = 10) -> List[ParameterSetResult]:
        """Get top K parameter sets by optimization metric"""
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x.get_metric(self.optimization_metric),
            reverse=True  # Assume higher is better for most metrics
        )
        return sorted_results[:k]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all results to pandas DataFrame for analysis"""
        data = []
        for result in self.all_results:
            row = {}
            # Add parameters with 'param_' prefix for Grid Search compatibility
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value
            # Add metrics without prefix
            row.update(result.metrics)
            row['evaluation_id'] = result.evaluation_id
            row['backtest_time'] = result.backtest_time_seconds
            data.append(row)

        df = pd.DataFrame(data)
        return df.sort_values(self.optimization_metric, ascending=False)

    def summary(self) -> Dict[str, Any]:
        """Get optimization summary statistics"""
        metric_values = [r.get_metric(self.optimization_metric) for r in self.all_results]

        return {
            'method': self.method,
            'total_evaluations': self.total_evaluations,
            'total_time_seconds': self.total_time_seconds,
            'avg_time_per_eval': self.total_time_seconds / max(self.total_evaluations, 1),
            'best_parameters': self.best_parameters,
            'best_metric_value': self.best_metrics.get(self.optimization_metric, np.nan),
            'metric_mean': np.mean(metric_values),
            'metric_std': np.std(metric_values),
            'metric_min': np.min(metric_values),
            'metric_max': np.max(metric_values),
            'optimization_metric': self.optimization_metric,
            'cpcv_validated': self.cpcv_validated
        }

    # Compatibility properties for Grid Search interface
    @property
    def best_params(self) -> Dict[str, Any]:
        """Alias for best_parameters (Grid Search compatibility)"""
        return self.best_parameters

    @property
    def total_combinations(self) -> int:
        """Alias for total_evaluations (Grid Search compatibility)"""
        return self.total_evaluations

    @property
    def execution_time(self) -> float:
        """Alias for total_time_seconds (Grid Search compatibility)"""
        return self.total_time_seconds

    @property
    def best_metric_value(self) -> float:
        """Best value of the optimization metric"""
        return self.best_metrics.get(self.optimization_metric, np.nan)

    @property
    def cpcv_results(self) -> Optional[List[Any]]:
        """Alias for top_k_cpcv_results (Grid Search compatibility)"""
        if self.cpcv_validated and self.top_k_cpcv_results:
            # Convert to format expected by Grid Search
            return [
                {'params': result.parameters, 'cpcv_result': result.cpcv_result}
                for result in self.all_results
                if hasattr(result, 'cpcv_result') and result.cpcv_result is not None
            ]
        return None


@dataclass
class OptimizationSummary:
    """High-level summary for quick reporting"""

    method: str
    best_parameters: Dict[str, Any]
    best_sharpe: float
    best_cagr: float
    total_evaluations: int
    total_time_minutes: float

    # Validation status
    pbo: Optional[float] = None
    dsr: Optional[float] = None
    validation_passed: Optional[bool] = None

    def __repr__(self):
        return (
            f"OptimizationSummary(\n"
            f"  method={self.method},\n"
            f"  best_params={self.best_parameters},\n"
            f"  sharpe={self.best_sharpe:.2f},\n"
            f"  cagr={self.best_cagr:.2%},\n"
            f"  evaluations={self.total_evaluations},\n"
            f"  time={self.total_time_minutes:.1f}min,\n"
            f"  validated={self.validation_passed}\n"
            f")"
        )
