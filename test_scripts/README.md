# Test Scripts Directory

This directory is for **temporary test scripts** used during development to verify bug fixes, test new features, and validate changes.

## Purpose

When working on the BackT framework, developers often need to create quick test scripts to:
- Verify bug fixes (e.g., CAGR calculation, Bollinger Bands strategy)
- Test new features before integration
- Reproduce issues reported by users
- Validate performance optimizations
- Debug specific scenarios

These test scripts are different from the formal unit tests in `tests/` directory. They are:
- **Temporary** - Created for immediate testing needs
- **Exploratory** - Used to understand behavior and validate fixes
- **Ad-hoc** - Not meant for CI/CD or automated testing

## Conventions

### File Naming
Use descriptive names that clearly indicate what is being tested:
```
test_cagr_fix.py          # Testing CAGR calculation fix
test_bollinger_fix.py     # Testing Bollinger Bands strategy fixes
test_single_symbol.py     # Testing single symbol backtest
test_performance.py       # Testing performance improvements
diagnose_macd.py          # Diagnosing MACD signal issues
```

### Script Structure
Keep test scripts simple and focused:
```python
"""Brief description of what this script tests"""
from backt import Backtester, BacktestConfig
from strategies.momentum import some_strategy

# Test configuration
config = BacktestConfig(
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=100000.0
)

# Run test
backtester = Backtester(config)
result = backtester.run(strategy=some_strategy, universe=['SPY'])

# Verify expected behavior
print(f"Total Return: {result.metrics['total_return']:.2%}")
assert len(result.trades) > 0, "Expected trades to be generated"
```

### Cleanup
- Test scripts should be **deleted** after the fix/feature is verified and committed
- Do NOT commit test scripts to the repository (they are gitignored by default)
- If a test script contains valuable test cases, convert it to a proper unit test in `tests/`

### When to Move to `tests/`
Consider moving a test script to the formal `tests/` directory if it:
- Tests core functionality that should be verified on every commit
- Contains edge cases that should be continuously validated
- Provides regression testing for important bugs
- Is comprehensive enough to serve as integration testing

## Directory Structure

```
backtester2/
├── tests/                    # Formal unit tests (pytest)
│   ├── test_config.py
│   ├── test_data_loaders.py
│   ├── test_portfolio.py
│   └── test_validation.py
│
├── test_scripts/            # Temporary test scripts (you are here!)
│   ├── README.md           # This file
│   ├── test_*.py           # Your temporary test scripts
│   └── diagnose_*.py       # Diagnostic scripts
│
└── examples/               # Example strategies and tutorials
    ├── simple_strategy.py
    ├── buy_and_hold.py
    └── advanced_strategy.py
```

## Benefits

Using this directory instead of creating test files at the project root:
1. **Cleaner root directory** - Keeps project organized
2. **Easy cleanup** - Can delete entire directory or use wildcards
3. **Clear separation** - Distinguishes temporary tests from permanent unit tests
4. **Git-friendly** - Test scripts are gitignored by default (see `.gitignore`)
5. **Team collaboration** - Everyone knows where to put temporary test files

## Examples

### Testing a Bug Fix
```python
# test_scripts/test_cagr_fix.py
"""Verify CAGR calculation uses calendar time, not data point count"""

import pandas as pd
import numpy as np
from backt.risk.metrics import MetricsEngine
from backt.utils.config import BacktestConfig

# Create test case: 574% return over 13.8 years should give ~14.8% CAGR
dates = pd.date_range('2012-01-02', '2025-10-24', freq='B')
equity_curve = pd.DataFrame({
    'total_equity': np.linspace(100000, 674410, len(dates))
}, index=dates)

config = BacktestConfig(
    start_date='2012-01-02',
    end_date='2025-10-24',
    initial_capital=100000.0
)

metrics_engine = MetricsEngine(config)
metrics = metrics_engine.calculate_metrics(equity_curve)

expected_cagr = 0.1482  # 14.82%
assert abs(metrics['cagr'] - expected_cagr) < 0.001, \
    f"CAGR {metrics['cagr']:.2%} does not match expected {expected_cagr:.2%}"

print("✅ CAGR calculation is correct!")
```

### Testing a Strategy
```python
# test_scripts/test_bollinger_fix.py
"""Test Bollinger Bands strategy generates trades with metadata"""

from backt import Backtester, BacktestConfig
from backt.utils.config import ExecutionConfig
from strategies.momentum import bollinger_mean_reversion

config = BacktestConfig(
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_capital=100000.0,
    execution=ExecutionConfig(slippage_pct=0.0005),
    verbose=False
)

backtester = Backtester(config)
result = backtester.run(
    strategy=bollinger_mean_reversion,
    universe=['SPY'],
    strategy_params={'bb_period': 20, 'bb_std': 2.0}
)

# Verify trades were generated
assert len(result.trades) > 0, "Expected trades to be generated"

# Verify metadata exists
assert 'meta_reason' in result.trades.columns, "Expected metadata in trades"

print(f"✅ Generated {len(result.trades)} trades with metadata")
print(result.trades[['symbol', 'side', 'meta_reason']].head())
```

## Git Ignore

The `.gitignore` file at the project root includes:
```
# Temporary test scripts
test_scripts/*.py
test_scripts/*.ipynb
```

This means Python files and notebooks in this directory are **not tracked by git** by default. This prevents accidental commits of temporary test code.

If you want to commit a test script (e.g., for sharing with team), you can force-add it:
```bash
git add -f test_scripts/my_important_test.py
```

But in general, test scripts should be temporary and deleted after use!

---

**Remember:** Keep the project root clean! Use this directory for all temporary test files.
