# BackT Strategy Library

Organized collection of reusable trading strategies by theme.

## Overview

This directory contains ready-to-use trading strategies that can be imported and used in notebooks or scripts. Each module focuses on a specific theme (momentum, mean reversion, etc.) and contains multiple related strategies.

## Available Strategies

### Momentum Strategies (`momentum.py`)

Moving average crossover strategies using traditional and Kalman-enhanced approaches.

#### 1. `ma_crossover_long_only`
Traditional moving average crossover, long positions only.

**Parameters:**
- `fast_ma` (int, default=20): Fast MA period
- `slow_ma` (int, default=50): Slow MA period
- `min_periods` (int, default=60): Minimum data required
- `max_position_size` (float, default=0.25): Max position weight

**Example:**
```python
from strategies import ma_crossover_long_only

params = {
    'fast_ma': 20,
    'slow_ma': 50,
    'max_position_size': 0.25
}

result = backtester.run(
    strategy=ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'IWM'],
    strategy_params=params
)
```

#### 2. `ma_crossover_long_short`
Traditional moving average crossover with short positions on death crosses.

**Parameters:**
- `fast_ma` (int, default=20): Fast MA period
- `slow_ma` (int, default=50): Slow MA period
- `min_periods` (int, default=60): Minimum data required
- `max_position_size` (float, default=0.25): Max position weight

**Example:**
```python
from strategies import ma_crossover_long_short

params = {
    'fast_ma': 8,
    'slow_ma': 20,
    'max_position_size': 0.25
}

config = BacktestConfig(
    ...,
    allow_short=True,  # IMPORTANT: Enable short selling
    max_leverage=2.0
)

result = backtester.run(
    strategy=ma_crossover_long_short,
    universe=['SPY', 'QQQ', 'TLT'],
    strategy_params=params
)
```

#### 3. `kalman_ma_crossover_long_only`
Kalman-enhanced MA crossover, long positions only. Uses adaptive filtering for noise reduction.

**Parameters:**
- `Q_fast` (float, default=0.01): Fast filter process noise (higher = more responsive)
- `Q_slow` (float, default=0.001): Slow filter process noise (lower = smoother)
- `R` (float, default=1.0): Measurement noise
- `min_periods` (int, default=60): Minimum data required
- `max_position_size` (float, default=0.25): Max position weight

**Example:**
```python
from strategies import kalman_ma_crossover_long_only

params = {
    'Q_fast': 0.01,   # Responsive filter
    'Q_slow': 0.001,  # Smooth filter
    'R': 1.0,
    'max_position_size': 0.25
}

result = backtester.run(
    strategy=kalman_ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'IWM'],
    strategy_params=params
)
```

#### 4. `kalman_ma_crossover_long_short`
Kalman-enhanced MA crossover with short positions. Best noise reduction with long-short capability.

**Parameters:**
- `Q_fast` (float, default=0.01): Fast filter process noise
- `Q_slow` (float, default=0.001): Slow filter process noise
- `R` (float, default=1.0): Measurement noise
- `min_periods` (int, default=60): Minimum data required
- `max_position_size` (float, default=0.25): Max position weight

**Example:**
```python
from strategies import kalman_ma_crossover_long_short

params = {
    'Q_fast': 0.01,
    'Q_slow': 0.001,
    'R': 1.0,
    'max_position_size': 0.25
}

config = BacktestConfig(
    ...,
    allow_short=True,
    max_leverage=2.0
)

result = backtester.run(
    strategy=kalman_ma_crossover_long_short,
    universe=['SPY', 'QQQ', 'TLT', 'GLD'],
    strategy_params=params
)
```

## Usage in Notebooks

### Basic Import and Run

```python
# Import BackT components
from backt import Backtester, BacktestConfig
from strategies import ma_crossover_long_only

# Configure backtest
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000.0
)

# Define strategy parameters
params = {
    'fast_ma': 20,
    'slow_ma': 50,
    'max_position_size': 0.25
}

# Run backtest
backtester = Backtester(config)
result = backtester.run(
    strategy=ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'IWM'],
    strategy_params=params
)

# Analyze results
from backt.reporting import PerformanceReport
report = PerformanceReport(result)
report.print_report()
```

### Comparing Multiple Strategies

```python
from strategies import (
    ma_crossover_long_only,
    kalman_ma_crossover_long_only
)

strategies = {
    'Traditional MA': (ma_crossover_long_only, {'fast_ma': 20, 'slow_ma': 50}),
    'Kalman MA': (kalman_ma_crossover_long_only, {'Q_fast': 0.01, 'Q_slow': 0.001})
}

results = {}
for name, (strategy_func, params) in strategies.items():
    result = backtester.run(
        strategy=strategy_func,
        universe=['SPY', 'QQQ'],
        strategy_params=params
    )
    results[name] = result

# Compare performance
import pandas as pd
comparison = pd.DataFrame({
    name: {
        'Total Return': result.equity_curve['total_value'].iloc[-1] / 100000 - 1,
        'Num Trades': len(result.trades)
    }
    for name, result in results.items()
}).T
```

## Strategy Signature

All strategies follow the standard BackT signature:

```python
def strategy_name(
    market_data: Dict[str, pd.DataFrame],  # OHLCV data for each symbol
    current_time: pd.Timestamp,            # Current simulation time
    positions: Dict[str, Position],        # Current positions
    context: Dict[str, Any],               # Persistent strategy state
    params: Dict[str, Any]                 # Strategy parameters
) -> Dict[str, Dict]:                      # Orders to execute
    """Strategy docstring"""
    # Implementation
    return orders
```

## Adding New Strategies

To add a new strategy:

1. Create a new module (e.g., `mean_reversion.py`) or add to existing module
2. Follow the standard strategy signature
3. Include comprehensive docstring with parameters and examples
4. Update `__init__.py` to export the strategy
5. Update this README with usage examples

## Parameter Tuning Guidelines

### Traditional MA Crossover
- **Fast MA**: 10-20 days for active trading, 20-50 for moderate
- **Slow MA**: 50-200 days (200 = long-term trend)
- **Common combinations**: (20, 50), (50, 200), (10, 30)

### Kalman Filter Parameters
- **Q_fast**: 0.01-0.1 (higher = more responsive, like short MA)
- **Q_slow**: 0.0001-0.001 (lower = smoother, like long MA)
- **R**: Usually 1.0 (standard measurement noise)
- **Typical combinations**: (0.01, 0.001), (0.1, 0.01), (0.001, 0.0001)

### Position Sizing
- **Conservative**: max_position_size = 0.10 (10% max per position)
- **Moderate**: max_position_size = 0.25 (25% max per position)
- **Aggressive**: max_position_size = 0.33 (33% max per position)

## Best Practices

1. **Always test with mock data first** before using real market data
2. **Use walk-forward analysis** for parameter optimization
3. **Set appropriate risk limits** in BacktestConfig (max_position_size, max_leverage)
4. **Compare against benchmarks** (e.g., SPY buy-and-hold)
5. **Monitor strategy context** for debugging and analysis

## Performance Considerations

- **Traditional MA**: Fast computation, suitable for large universes
- **Kalman Filter**: Slower due to filter updates, best for < 50 symbols
- **Long-Short**: Requires allow_short=True and appropriate max_leverage

## Future Strategy Themes

Planned strategy modules:
- `mean_reversion.py`: Bollinger Band, RSI, z-score strategies
- `pairs_trading.py`: Statistical arbitrage and cointegration strategies
- `multi_factor.py`: Combined momentum, value, quality factors
- `machine_learning.py`: ML-based signal generation strategies
