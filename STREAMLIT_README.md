# BackT Streamlit Backtest Runner

Professional multi-sheet web interface for running backtests with the BackT framework.

## Features

### 📋 Three-Sheet Interface

1. **Configuration Sheet** - Set all backtest parameters
   - Date range and capital settings
   - Trading universe (symbols)
   - Execution costs (spreads, slippage, commissions)
   - Risk management (max leverage, position size limits)
   - Mock data toggle for testing

2. **Strategy Sheet** - Select and configure strategies
   - Auto-discovers all strategies from `strategies/` folder
   - Shows strategy documentation
   - Dynamic parameter inputs based on strategy
   - One-click backtest execution

3. **Results Sheet** - View comprehensive analysis
   - Performance metrics (returns, Sharpe, drawdown, etc.)
   - Interactive charts (equity curve, drawdown, monthly heatmap)
   - Per-symbol performance breakdown
   - Correlation analysis
   - Trade history
   - Full metrics table

## Quick Start

### Installation

```bash
# Install required dependencies
pip install streamlit yfinance

# Or install with BackT (includes all dependencies)
pip install -e ".[web]"
```

**Note**: `yfinance` is required for loading benchmark data (SPY) and real market data. If you see "yfinance not installed" errors, run:
```bash
pip install yfinance
```

### Running the App

```bash
streamlit run streamlit_backtest_runner.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Step 1: Configure Backtest

1. Go to the **Configuration** tab
2. Set your date range and initial capital
3. Enter your trading universe (comma-separated symbols)
4. Configure execution costs if needed
5. Set risk management parameters
6. Click **Save Configuration**

### Step 2: Select Strategy

1. Go to the **Strategy** tab
2. Select a strategy from the dropdown (auto-populated from `strategies/` folder)
3. View the strategy documentation in the expander
4. Configure strategy parameters (auto-generated from docstrings)
5. Click **Run Backtest**

### Step 3: View Results

1. Go to the **Results** tab (after backtest completes)
2. View performance summary with key metrics
3. Analyze charts (equity curve, drawdown, monthly returns)
4. Examine per-symbol performance
5. Review detailed metrics and trade history

## Session Status

The sidebar shows your current session status:
- ✅ Configuration Set / ⚠️ No Configuration
- ✅ Strategy Selected / ⚠️ No Strategy Selected
- ✅ Backtest Complete / ⚠️ No Results

Use the **Reset Session** button to start over.

## Strategy Integration

The app automatically discovers strategies from the `strategies/` module:

**Currently Available:**

**Trend Following:**
- `ma_crossover_long_only` - Traditional MA crossover (long only)
- `ma_crossover_long_short` - Traditional MA crossover (long-short)
- `kalman_ma_crossover_long_only` - Kalman-enhanced MA (long only)
- `kalman_ma_crossover_long_short` - Kalman-enhanced MA (long-short)
- `macd_crossover` - MACD trend following (long-only or long-short)
- `adx_trend_filter` - ADX trend strength filter with directional trading

**Mean Reversion:**
- `rsi_mean_reversion` - RSI overbought/oversold (long only)
- `bollinger_mean_reversion` - Bollinger Bands mean reversion (long only)
- `stochastic_momentum` - Stochastic oscillator momentum (long only)

**AQR Capital Management Strategies (Factor-Based):**
- `quality_minus_junk` - Long high-quality, short junk stocks (QMJ factor)
- `quality_long_only` - Long-only quality stocks
- `value_everywhere` - Multi-metric value strategy (B/P, E/P, CF/P)
- `betting_against_beta` - Long low-beta, short high-beta stocks
- `defensive_equity` - Long-only low-beta defensive stocks
- `quality_value_momentum` - Combined 3-factor strategy (AQR's "New Core")

**Adding New Strategies:**
1. Create strategy function in `strategies/` folder
2. Follow standard BackT signature
3. Add comprehensive docstring with parameters
4. Strategy will automatically appear in dropdown

### Parameter Auto-Detection

The app parses strategy docstrings to auto-generate parameter inputs:

```python
def my_strategy(...):
    """
    My Strategy Description

    Parameters:
    -----------
    fast_ma : int, default=20
        Short-term moving average period
    slow_ma : int, default=50
        Long-term moving average period
    threshold : float, default=0.02
        Signal threshold
    use_filter : bool, default=True
        Enable filtering
    """
```

This generates:
- Integer inputs for `fast_ma`, `slow_ma`
- Float input for `threshold`
- Checkbox for `use_filter`

## Example Workflow

### 1. Test with Mock Data

```
Configuration Tab:
- Start: 2020-01-01
- End: 2023-12-31
- Capital: $100,000
- Symbols: SPY, QQQ, TLT
- Use Mock Data: ✓

Strategy Tab:
- Select: ma_crossover_long_only
- fast_ma: 20
- slow_ma: 50
- Run Backtest

Results Tab:
- View performance metrics
- Analyze charts
```

### 2. Real Data Backtest

```
Configuration Tab:
- Start: 2020-01-01
- End: 2023-12-31
- Capital: $100,000
- Symbols: SPY, QQQ, TLT, GLD, IEF
- Spread: 0.05%
- Commission: $0.001/share
- Max Leverage: 2.0
- Max Position: 0.25

Strategy Tab:
- Select: kalman_ma_crossover_long_short
- Q_fast: 0.01
- Q_slow: 0.001
- R: 1.0
- Run Backtest

Results Tab:
- Compare vs benchmark
- Check per-symbol performance
- Review correlation matrix
```

## Performance Tips

1. **Start Small**: Test with few symbols and short periods first
2. **Use Mock Data**: Validate strategy logic before real data
3. **Check Logs**: Terminal shows detailed backtest progress
4. **Save Results**: Use browser screenshots or download data
5. **Iterate**: Use Reset Session to try different configurations

## Troubleshooting

### Strategy Not Appearing
- Check strategy is in `strategies/` folder
- Ensure function follows BackT signature
- Verify function name doesn't start with `_`

### Parameters Not Auto-Detected
- Add comprehensive docstring
- Use format: `param_name : type, default=value`
- Supported types: int, float, bool

### Backtest Fails
- Check date range validity
- Verify symbols are valid
- Check strategy parameters
- Review terminal for error messages

### Charts Not Showing
- Ensure backtest completed successfully
- Check browser console for errors
- Try refreshing the page

## Comparison with Notebooks

| Feature | Streamlit App | Jupyter Notebooks |
|---------|---------------|-------------------|
| Ease of Use | ⭐⭐⭐⭐⭐ GUI | ⭐⭐⭐ Code |
| Speed | ⭐⭐⭐⭐ Fast setup | ⭐⭐⭐ Slower |
| Flexibility | ⭐⭐⭐ Structured | ⭐⭐⭐⭐⭐ Full control |
| Documentation | ⭐⭐⭐⭐ Auto-generated | ⭐⭐⭐⭐ Manual |
| Sharing | ⭐⭐⭐⭐⭐ Web link | ⭐⭐⭐ File sharing |

**Use Streamlit when:**
- Quick strategy testing
- Non-technical users
- Parameter sweeps
- Demos and presentations

**Use Notebooks when:**
- Complex custom analysis
- Multiple strategy comparisons
- Research and development
- Detailed documentation

## Advanced Features

### Custom Execution Costs

Configure realistic transaction costs:
- **Spread**: Bid-ask spread in %
- **Slippage**: Price impact in %
- **Commission**: Per-share or per-trade

### Risk Management

Set portfolio-wide limits:
- **Max Leverage**: Total exposure limit (e.g., 2.0 = 200%)
- **Max Position Size**: Single position limit (e.g., 0.25 = 25%)

### Per-Symbol Analysis

Results sheet shows individual symbol performance:
- Total PnL by symbol
- Sharpe ratio per symbol
- Win rate per symbol
- Correlation matrix

## Future Enhancements

Planned features:
- Export results to CSV/Excel
- Save/load configurations
- Parameter optimization grid
- Strategy comparison mode
- Walk-forward analysis
- Live trading simulation

## Support

For issues or questions:
- Check BackT documentation
- Review strategy library README
- Open GitHub issue

## Credits

Built with:
- BackT Framework
- Streamlit
- Pandas & NumPy
- Matplotlib & Plotly
