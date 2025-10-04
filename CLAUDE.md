# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BackT is a professional trading backtesting framework written in Python. It provides an event-driven simulation engine for testing trading strategies with realistic execution modeling, comprehensive risk analytics, and flexible strategy development tools.

## Installation and Setup

Install the package in development mode:
```bash
pip install -e .
```

Install with optional dependencies for development:
```bash
pip install -e ".[dev,jupyter,web]"
```

## Common Development Commands

### Testing
```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=backt --cov-report=html
```

### Code Quality
```bash
# Format code with Black
black backt/

# Lint with flake8
flake8 backt/

# Type checking with mypy
mypy backt/
```

### Running Examples
```bash
cd examples
python simple_strategy.py
python buy_and_hold.py
python advanced_strategy.py
```

### Web Interface
```bash
# Launch Streamlit web interface
streamlit run streamlit_app.py

# Or use the launcher script
python launch_streamlit.py
```

### CLI Usage
```bash
# Use the CLI tool (after installation)
backt --help
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Modules

- **`backt/engine/`** - Core backtesting engine (`Backtester` class) that orchestrates the event-driven simulation
- **`backt/data/`** - Data loading and management (Yahoo Finance, CSV, custom loaders)
- **`backt/execution/`** - Order execution simulation with configurable slippage, commissions, and market impact
- **`backt/portfolio/`** - Portfolio and position management (`PortfolioManager`, `PositionManager`)
- **`backt/signal/`** - Strategy helpers and technical indicators (`TechnicalIndicators`, `StrategyHelpers`)
- **`backt/risk/`** - Risk analytics and performance metrics (`MetricsEngine`, `RiskCalculator`)
- **`backt/reporting/`** - Results output, visualization, and trade logging
- **`backt/api/`** - CLI and Streamlit web interfaces
- **`backt/utils/`** - Configuration, types, constants, and logging

### Key Design Patterns

- **Event-Driven Simulation**: The backtester processes data timestamp-by-timestamp, calling strategies at each time step
- **Standardized Data Contracts**: All market data uses pandas DataFrames with OHLCV format and timezone support
- **Strategy Function API**: Strategies are simple functions with standardized signatures that return order dictionaries
- **Modular Execution**: Pluggable execution engines allow different market simulation models
- **Type Safety**: Full type hints throughout the codebase using dataclasses and custom types

### Strategy Development

All strategies follow this signature:
```python
def strategy_function(
    market_data: Dict[str, pd.DataFrame],  # OHLCV data for each symbol
    current_time: pd.Timestamp,            # Current simulation time
    positions: Dict[str, Position],        # Current positions
    context: Dict[str, Any],               # Persistent strategy state
    params: Dict[str, Any]                 # Strategy parameters
) -> Dict[str, Dict]:                      # Orders to execute
```

Order types supported:
- `{'action': 'target_weight', 'weight': 0.25}` - Target portfolio percentage
- `{'action': 'buy', 'size': 100}` - Fixed size orders
- `{'action': 'sell', 'size': 50}` - Fixed size orders
- `{'action': 'close'}` - Close position
- `{'action': 'buy', 'size': 100, 'order_type': 'limit', 'limit_price': 150.0}` - Limit orders

### Configuration System

The framework uses dataclass-based configuration:
- **`BacktestConfig`** - Main backtest parameters (dates, capital, trading rules)
- **`ExecutionConfig`** - Execution simulation settings (slippage, commissions, spreads)

### Data Loading

Multiple data sources supported through a common interface:
- **`YahooDataLoader`** - Yahoo Finance data (default)
- **`CSVDataLoader`** - CSV file loading
- **`CustomDataLoader`** - Custom data loading functions

### Testing Strategy

- Unit tests in `tests/` directory cover core functionality
- Integration tests verify end-to-end backtesting workflows
- Example strategies in `examples/` serve as integration tests
- Test framework supports both pytest and custom test runners

### Entry Points

- **Main Package**: Import `Backtester`, `BacktestConfig` from `backt`
- **CLI Tool**: `backt` command (defined in `backt.api.cli`)
- **Web Interface**: `streamlit_app.py` for interactive backtesting
- **Examples**: `examples/` directory contains working strategy examples

### Dependencies and Compatibility

- Python 3.8+ required
- Core dependencies: pandas, numpy, scipy, matplotlib, yfinance
- Optional: streamlit (web), jupyter (notebooks), numba (performance)
- Development tools: pytest, black, flake8, mypy, pre-commit

## Git Workflow

**IMPORTANT**: Always ask for explicit permission before running `git commit` or `git push` commands.

- Never commit or push changes without user approval
- When changes are ready, summarize what will be committed and ask: "Would you like me to commit and push these changes?"
- Only proceed with git operations after receiving explicit confirmation
- Exception: `git status`, `git diff`, and other read-only git commands can be used without asking