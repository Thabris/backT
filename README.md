# BackT - Professional Trading Backtesting Framework

**BackT** is a comprehensive, modular Python framework for backtesting trading strategies. It provides an event-driven simulation engine with realistic execution modeling, comprehensive risk analytics, and flexible strategy development tools.

## ⚡ Quick Start

### Installation

Install BackT in development mode to access all features:

```bash
# Clone or download the project
cd backtester2

# Install in editable mode with dependencies
pip install -e .

# Or install with optional features
pip install -e ".[dev,jupyter,web]"
```

### Simple Example

```python
from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers

def simple_ma_strategy(market_data, current_time, positions, context, params):
    """Simple moving average crossover strategy"""
    orders = {}

    for symbol, data in market_data.items():
        if len(data) < 50:
            continue

        short_ma = TechnicalIndicators.sma(data['close'], 20)
        long_ma = TechnicalIndicators.sma(data['close'], 50)

        if StrategyHelpers.is_crossover(short_ma, long_ma):
            orders[symbol] = {'action': 'target_weight', 'weight': 1.0}
        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            orders[symbol] = {'action': 'target_weight', 'weight': 0.0}

    return orders

# Configure and run backtest
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-01-01',
    initial_capital=100000
)

backtester = Backtester(config)
result = backtester.run(simple_ma_strategy, ['AAPL'])

# View results
print(f"Total Return: {result.performance_metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.performance_metrics['sharpe_ratio']:.3f}")
```

## 🏗️ Architecture

BackT is built with a modular architecture following professional software development practices:

```
backt/
├── data/           # Data loading and management
├── engine/         # Core backtesting engine
├── execution/      # Order execution simulation
├── portfolio/      # Portfolio and position management
├── risk/           # Risk analytics and metrics
├── signal/         # Strategy helpers and indicators
├── reporting/      # Results output and visualization
├── api/            # CLI and web interfaces
└── utils/          # Configuration and utilities
```

### Key Features

- **Event-Driven Engine**: Realistic timestamp-by-timestamp simulation
- **Standardized Data Contracts**: Pandas-based OHLCV format with timezone support
- **Flexible Strategy API**: Simple function-based strategy development
- **Realistic Execution**: Configurable slippage, commissions, and market impact
- **Comprehensive Analytics**: 25+ performance and risk metrics
- **Multiple Data Sources**: Yahoo Finance, CSV, custom loaders
- **Professional Logging**: Structured logging throughout the framework
- **Type Safety**: Full type hints and validation

## 📊 Strategy Development

### Strategy Function Signature

All strategies follow this standardized signature:

```python
def my_strategy(
    market_data: Dict[str, pd.DataFrame],  # OHLCV data for each symbol
    current_time: pd.Timestamp,            # Current simulation time
    positions: Dict[str, Position],        # Current positions
    context: Dict[str, Any],               # Persistent strategy state
    params: Dict[str, Any]                 # Strategy parameters
) -> Dict[str, Dict]:                      # Orders to execute
    pass
```

### Order Types

BackT supports multiple order types:

```python
# Target weight orders (portfolio percentage)
{'action': 'target_weight', 'weight': 0.25}

# Fixed size orders
{'action': 'buy', 'size': 100}
{'action': 'sell', 'size': 50}

# Close position
{'action': 'close'}

# Market vs Limit orders
{'action': 'buy', 'size': 100, 'order_type': 'limit', 'limit_price': 150.0}
```

### Technical Indicators

Built-in indicators for common strategies:

```python
from backt.signal import TechnicalIndicators

# Moving averages
sma = TechnicalIndicators.sma(prices, 20)
ema = TechnicalIndicators.ema(prices, 20)

# Momentum indicators
rsi = TechnicalIndicators.rsi(prices, 14)

# Volatility indicators
bb = TechnicalIndicators.bollinger_bands(prices, 20, 2)
```

## ⚙️ Configuration

### Basic Configuration

```python
from backt import BacktestConfig

config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-01-01',
    initial_capital=100000,
    data_frequency='1D',
    allow_short=False,
    max_leverage=1.0
)
```

### Advanced Execution Settings

```python
from backt.utils.config import ExecutionConfig

execution = ExecutionConfig(
    spread=0.01,                    # Bid-ask spread
    slippage_pct=0.0005,           # Slippage as % of trade size
    commission_per_share=0.001,     # Per-share commission
    commission_per_trade=1.0        # Flat commission per trade
)

config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-01-01',
    initial_capital=100000,
    execution=execution
)
```

## 📈 Results Analysis

### Performance Metrics

BackT calculates comprehensive performance metrics:

- **Return Metrics**: Total return, CAGR, daily/monthly returns
- **Risk Metrics**: Volatility, Sharpe ratio, Sortino ratio, VaR
- **Drawdown Analysis**: Maximum drawdown, recovery time, Calmar ratio
- **Trade Analytics**: Win rate, profit factor, average trade size

### Visualization

```python
from backt.reporting import PlotGenerator, ReportGenerator

# Create plots
plotter = PlotGenerator()
plotter.plot_equity_curve(result.equity_curve)
plotter.plot_drawdown(result.equity_curve)
plotter.create_performance_dashboard(result)

# Generate reports
report_gen = ReportGenerator("./results")
files = report_gen.generate_full_report(result, "my_strategy")
```

## 💾 Data Sources

### Yahoo Finance (Default)

```python
from backt.data import YahooDataLoader

loader = YahooDataLoader()
data = loader.load(['AAPL', 'MSFT'], '2020-01-01', '2023-01-01')
```

### CSV Files

```python
from backt.data import CSVDataLoader

loader = CSVDataLoader(date_column='Date')
data = loader.load(['AAPL'], '2020-01-01', '2023-01-01',
                   file_path='./data/AAPL.csv')
```

### Custom Data Sources

```python
from backt.data import CustomDataLoader

def my_data_loader(symbols, start_date, end_date, **kwargs):
    # Your custom data loading logic
    return data_dict

loader = CustomDataLoader(my_data_loader)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=backt --cov-report=html
```

## 📚 Examples

The `examples/` directory contains comprehensive strategy examples:

- **`simple_strategy.py`**: Basic moving average crossover
- **`buy_and_hold.py`**: Buy and hold baseline strategy
- **`advanced_strategy.py`**: Multi-indicator strategy with risk management

Run examples:

```bash
cd examples
python simple_strategy.py
python advanced_strategy.py
```

## 🔧 Development Setup

For development work:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black backt/
flake8 backt/

# Type checking
mypy backt/
```

## 🎯 Use Cases

BackT is ideal for:

- **Strategy Research**: Systematic testing of trading ideas
- **Academic Research**: Reproducible financial research
- **Risk Analysis**: Portfolio risk assessment and optimization
- **Education**: Learning quantitative finance and algorithmic trading
- **Production**: Foundation for live trading systems

## 📋 Requirements

- Python 3.8+
- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0 (for plotting)
- yfinance >= 0.1.70 (for Yahoo Finance data)

Optional dependencies:
- streamlit (web interface)
- jupyter (notebook support)
- numba (performance optimization)
- scikit-learn (advanced analytics)

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines for:

- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

## 📄 License

BackT is released under the MIT License. See LICENSE file for details.

## 🙋 Support

- **Documentation**: [Full documentation](https://backt.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/backt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/backt/discussions)

---

**Built with ❤️ for the quantitative finance community**
