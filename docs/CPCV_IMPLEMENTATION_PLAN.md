# CPCV Implementation Plan for BackT Framework

## Overview
This document outlines the implementation plan for adding Combinatorial Purged Cross-Validation (CPCV) methodology to the BackT trading backtesting framework for robust time series momentum strategy testing.

## Background Research Summary

### Key Findings from Academic Research (2024-2025)
- CPCV significantly outperforms traditional walk-forward analysis
- Lower Probability of Backtest Overfitting (PBO)
- Superior Deflated Sharpe Ratio (DSR) test statistics
- Better handling of temporal dependencies in financial time series
- Developed by Marcos Lopez de Prado in "Advances in Financial Machine Learning"

### CPCV Advantages over Traditional Methods
- **Multiple Backtest Paths**: 45+ different train/test combinations vs single path
- **Superior Overfitting Detection**: Significantly lower PBO compared to traditional methods
- **Proper Data Leakage Prevention**: Purging and embargoing ensure no information leakage
- **Statistical Significance**: DSR accounts for multiple testing

## Implementation Framework

### Phase 1: Data Preparation & Framework Setup

#### 1.1 Data Structure
```python
# Target data structure for momentum testing
universe = ['SPY', 'EFA', 'EEM', 'TLT', 'VNQ', 'DBC', 'GLD', 'IEF']
data_period = '2010-01-01' to '2024-12-31'  # 15 years
frequency = 'monthly'  # Monthly rebalancing
```

#### 1.2 Time Series Momentum Strategy Definition
```python
def time_series_momentum_strategy(data, lookback_months=12, skip_months=1):
    """
    Pure time series momentum - each asset evaluated independently

    Parameters:
    - lookback_months: momentum calculation period (3, 6, 9, 12)
    - skip_months: skip recent months to avoid reversal (0, 1, 2)
    """
    # Calculate momentum for each asset independently
    # Go long if momentum > 0, cash if momentum <= 0
    # Position size: equal weight among selected assets
```

### Phase 2: CPCV Implementation Framework

#### 2.1 Purging and Embargoing Setup
```python
class CPCVConfig:
    n_splits = 10           # Number of folds
    n_test_splits = 2       # Number of test folds per iteration
    purge_pct = 0.05       # 5% data purging around test sets
    embargo_pct = 0.02     # 2% embargo period after test sets
    min_train_length = 24   # Minimum 24 months training data
```

#### 2.2 Data Leakage Prevention
- **Purging**: Remove 5% of data around test set boundaries
- **Embargoing**: 2% gap between training and test sets
- **Feature lag**: Ensure momentum calculations use only past data

#### 2.3 Combinatorial Path Generation
```python
# Calculate number of backtest paths
n_combinations = math.comb(n_splits, n_test_splits)  # C(10,2) = 45 paths
total_paths = (n_combinations * n_test_splits) / n_splits  # 45 * 2 / 10 = 9 paths
```

### Phase 3: Parameter Optimization Grid

#### 3.1 Parameter Space Definition
```python
parameter_grid = {
    'lookback_months': [3, 6, 9, 12, 18],
    'skip_months': [0, 1, 2],
    'volatility_target': [None, 0.10, 0.15],  # Optional vol targeting
    'momentum_threshold': [0.0, 0.02, 0.05],  # Minimum momentum required
    'max_allocation': [0.5, 1.0]  # Maximum allocation to single asset
}
# Total combinations: 5 × 3 × 3 × 3 × 2 = 270 parameter sets
```

#### 3.2 Optimization Objective
```python
def fitness_function(returns, benchmark_returns):
    """
    Multi-objective fitness combining:
    - Risk-adjusted returns (Sharpe ratio)
    - Maximum drawdown control
    - Stability across paths
    """
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(returns)
    calmar = sharpe / max_dd if max_dd > 0 else 0
    stability = 1 / std(sharpe_across_paths)  # Consistency reward

    return 0.4 * sharpe + 0.3 * calmar + 0.3 * stability
```

### Phase 4: Overfitting Detection Metrics

#### 4.1 Primary Metrics
```python
def calculate_overfitting_metrics(is_results, oos_results):
    """
    Calculate key overfitting detection metrics
    """
    # Probability of Backtest Overfitting (PBO)
    pbo = calculate_pbo(is_results, oos_results)

    # Deflated Sharpe Ratio (DSR)
    dsr = calculate_deflated_sharpe(oos_sharpe, n_trials, skewness, kurtosis)

    # Performance degradation
    is_sharpe = np.mean([r.sharpe for r in is_results])
    oos_sharpe = np.mean([r.sharpe for r in oos_results])
    degradation = (is_sharpe - oos_sharpe) / is_sharpe

    return {'PBO': pbo, 'DSR': dsr, 'degradation': degradation}
```

#### 4.2 Stability Metrics
```python
def calculate_stability_metrics(results_across_paths):
    """
    Measure consistency across different CPCV paths
    """
    sharpe_ratios = [r.sharpe for r in results_across_paths]
    max_drawdowns = [r.max_drawdown for r in results_across_paths]

    return {
        'sharpe_stability': 1 / np.std(sharpe_ratios),
        'drawdown_consistency': np.mean(max_drawdowns),
        'win_rate_variance': np.var([r.win_rate for r in results_across_paths])
    }
```

## BackT Framework Integration

### Current BackT Compatibility Assessment

#### ✅ Strong Compatibility Areas
- **Core Architecture**: Event-driven engine perfect for CPCV implementation
- **Existing Components**: BacktestConfig, YahooDataLoader, MetricsEngine, PortfolioManager
- **Strategy Implementation**: Time series momentum fits naturally into BackT's strategy function pattern

#### 🔧 Required Adaptations

##### 1. Cross-Validation Framework Integration
```python
# New classes needed in backt/validation/
class CPCVValidator:
    """Combinatorial Purged Cross-Validation for BackT"""

class ParameterGrid:
    """Parameter optimization grid management"""

class OverfittingDetector:
    """PBO, DSR, and stability metrics calculation"""
```

##### 2. Batch Backtesting Capability
```python
# Extension to backt/engine/backtester.py
class BatchBacktester(Backtester):
    """Run multiple backtests with different parameters/data splits"""

    def run_parameter_grid(self, strategy, symbols, parameter_grid):
        """Run strategy across parameter combinations"""

    def run_cpcv_validation(self, strategy, symbols, cpcv_config):
        """Execute CPCV validation framework"""
```

##### 3. Data Splitting Utilities
```python
# New module: backt/utils/data_splits.py
def create_purged_splits(data, test_indices, purge_pct, embargo_pct):
    """Create purged and embargoed train/test splits"""

def generate_cpcv_combinations(n_splits, n_test_splits):
    """Generate combinatorial fold combinations"""
```

##### 4. Enhanced Metrics and Reporting
```python
# Extensions to backt/risk/metrics.py
def calculate_deflated_sharpe_ratio(returns, n_trials, skewness, kurtosis):
    """Calculate DSR for multiple testing correction"""

def calculate_probability_backtest_overfitting(is_results, oos_results):
    """Calculate PBO metric"""
```

### Required New Modules

```
backt/
├── validation/              # New module
│   ├── __init__.py
│   ├── cpcv_validator.py   # Main CPCV implementation
│   ├── parameter_grid.py   # Grid search management
│   └── overfitting.py      # PBO, DSR calculations
├── utils/
│   ├── data_splits.py      # New: Data splitting utilities
│   └── validation_utils.py # New: Validation helpers
└── examples/
    └── momentum_cpcv.py    # New: Complete CPCV example
```

### Integration Points with BackT

#### 1. Strategy Definition (No changes needed)
```python
def time_series_momentum_strategy(
    market_data: Dict[str, pd.DataFrame],
    current_time: pd.Timestamp,
    positions: Dict[str, Position],
    context: Dict[str, Any],
    params: Dict[str, Any]
) -> Dict[str, Dict]:
    # Leverage BackT's existing strategy API
    # Use TechnicalIndicators.momentum() if available
    # Return standard order format
```

#### 2. Configuration Extension
```python
@dataclass
class CPCVConfig(BacktestConfig):
    """Extended config for CPCV validation"""
    n_splits: int = 10
    n_test_splits: int = 2
    purge_pct: float = 0.05
    embargo_pct: float = 0.02
    parameter_grid: Dict[str, List] = field(default_factory=dict)
```

#### 3. Results Integration
```python
@dataclass
class CPCVResult(BacktestResult):
    """Extended result with CPCV metrics"""
    pbo_score: float
    deflated_sharpe: float
    stability_metrics: Dict[str, float]
    path_results: List[BacktestResult]
```

## Implementation Workflow

### Phase 5: CPCV Execution Loop
```python
def run_cpcv_momentum_test():
    """
    Main execution loop for CPCV momentum testing
    """

    # 1. Generate all combinatorial fold combinations
    fold_combinations = generate_cpcv_combinations(n_splits=10, n_test_splits=2)

    results = {}
    for param_set in parameter_grid:
        path_results = []

        for fold_combination in fold_combinations:
            # 2. Create purged train/test splits
            train_idx, test_idx = create_purged_splits(
                fold_combination, purge_pct=0.05, embargo_pct=0.02
            )

            # 3. Train strategy on purged training data
            strategy = train_momentum_strategy(data[train_idx], param_set)

            # 4. Test on out-of-sample data
            oos_result = backtest_strategy(strategy, data[test_idx])
            path_results.append(oos_result)

        # 5. Aggregate results across all paths
        results[param_set] = aggregate_path_results(path_results)

    return results
```

### Phase 6: Validation and Robustness Testing

#### Final Out-of-Sample Validation
```python
def final_validation(best_parameters, holdout_data):
    """
    Final validation on completely held-out data (2023-2024)
    """
    # Use best parameters from CPCV on fresh data
    # This data was never seen during optimization

    strategy = build_momentum_strategy(best_parameters)
    final_results = backtest_strategy(strategy, holdout_data)

    # Compare to CPCV expectations
    expectation_gap = compare_to_cpcv_expectations(final_results)

    return final_results, expectation_gap
```

#### Robustness Checks
- **Transaction costs**: Test with various cost assumptions (0.05%, 0.1%, 0.2%)
- **Market regimes**: Analyze performance in different volatility regimes
- **Universe changes**: Test with alternative asset universes
- **Frequency changes**: Test monthly vs quarterly rebalancing

## Expected Outcomes & Success Criteria

### Success Metrics
- **PBO < 0.5**: Low probability of overfitting
- **DSR > 1.0**: Statistically significant out-of-sample performance
- **Degradation < 20%**: Reasonable performance decay from in-sample
- **Sharpe stability > 2.0**: Consistent performance across paths

### Implementation Timeline
1. **Week 1-2**: Data preparation and CPCV framework setup
2. **Week 3-4**: Parameter grid testing and initial results
3. **Week 5**: Overfitting analysis and robustness testing
4. **Week 6**: Final validation and documentation

## Implementation Strategy Options

### Option A: External Wrapper (Faster to implement)
```python
class BackTCPCVFramework:
    """External CPCV framework using BackT as engine"""
    def __init__(self, backt_config):
        self.backtester = Backtester(backt_config)

    def run_cpcv_validation(self, strategy, symbols, cpcv_config):
        # Use BackT for individual backtests
        # Handle CPCV logic externally
```

### Option B: Integrated Enhancement (More robust)
```python
# Extend core BackT classes directly
class Backtester:  # Modified existing class
    def run_cpcv(self, strategy, symbols, cpcv_config):
        # Integrated CPCV capability
```

## Estimated Implementation Effort

### Code Requirements
- **New validation module**: ~500-800 lines of code
- **Extended configuration classes**: ~100 lines
- **Batch execution wrapper**: ~200-300 lines
- **Enhanced metrics calculation**: ~300 lines

**Total estimated addition**: ~1,000-1,500 lines of code, mostly in new modules without touching core BackT functionality.

## Competitive Advantage

This CPCV framework would provide a **significant competitive advantage** - professional-grade validation methodology that's rare in open-source backtesting frameworks.

## Recommendation

Start with Option A (external wrapper) for quick implementation, then migrate to Option B for production use.

---

*This plan was developed based on comprehensive research of academic papers and best practices for momentum strategy optimization and overfitting prevention (2024-2025).*