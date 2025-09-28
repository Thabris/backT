# Testing Requirements for BackT

## Environment Setup

⚠️ **CRITICAL: All tests must be run in the BackT environment**

### Installation Required
Before running any tests or examples, ensure BackT is properly installed:

```bash
# Install BackT in development mode with all dependencies
pip install -e .

# Or install with specific optional dependencies
pip install -e ".[dev,web]"
```

### Dependencies
The following packages are required and should be installed via the BackT setup:
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- yfinance >= 0.1.70

### Testing Environment
- **DO NOT** run tests with system Python
- **DO NOT** run tests without proper BackT installation
- **ALWAYS** ensure `pip install -e .` has been run first
- **VERIFY** all dependencies are available in the environment

## Running Examples

### Correct Way
```bash
# First install BackT
pip install -e .

# Then run examples
cd examples
python template_universe_run.py
python simple_strategy.py
```

### Incorrect Way
```bash
# ❌ This will fail - missing dependencies
python examples/template_universe_run.py

# ❌ This will fail - BackT not in path
cd examples && python template_universe_run.py
```

## Test Validation Process

1. **Environment Check**: Verify BackT installation with `pip list | grep backt`
2. **Dependency Check**: Ensure yfinance and other dependencies are available
3. **Import Test**: Run `python -c "import backt; print('BackT imported successfully')"`
4. **Example Test**: Run a simple example to validate full functionality

## Common Issues

### Import Errors
- **Problem**: `ModuleNotFoundError: No module named 'backt'`
- **Solution**: Run `pip install -e .` from project root

### Missing Dependencies
- **Problem**: `ModuleNotFoundError: No module named 'yfinance'`
- **Solution**: Run `pip install -e .` to install all dependencies

### Path Issues
- **Problem**: Cannot find modules when running examples
- **Solution**: Ensure BackT is properly installed, don't rely on sys.path modifications

## Environment Validation Script

Create and run this validation script before testing:

```python
# validate_environment.py
try:
    import backt
    from backt import Backtester, BacktestConfig
    from backt.data import YahooDataLoader
    import yfinance
    import pandas as pd
    print("✅ All dependencies available")
    print(f"BackT version: {backt.__version__}")
    print("Environment ready for testing")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install -e .")
```

This document ensures all testing is done in the proper environment with all dependencies correctly installed.