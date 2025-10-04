# Per-Symbol Correlation Matrix Methodology

## Overview

The correlation matrix calculates **Pearson correlation coefficients** between symbols based on their **percentage PnL changes** over time. This provides a normalized measure of how symbols move together, independent of position sizes.

## Calculation Steps

### 1. Extract Per-Symbol PnL Time Series

For each symbol, we track `total_pnl` at every timestamp during the backtest:

```python
# Example for AAPL
timestamp              total_pnl
2023-01-03                  0.00
2023-01-04                150.50
2023-01-05               -230.25
2023-01-06                180.75
...
```

### 2. Calculate Percentage Returns

For each symbol, we calculate the **percentage change** in PnL from one period to the next:

```python
pct_return[t] = (total_pnl[t] - total_pnl[t-1]) / base[t-1] * 100
```

Where `base[t-1]` is calculated with safety measures:

```python
base = max(abs(total_pnl[t-1]), min_base)
min_base = $100  # Prevents division by very small numbers
```

**Example:**
```
Date        total_pnl    pnl_change    base      pct_return
2023-01-03      0.00          -          -            -
2023-01-04    150.50      150.50      100.00      150.50%
2023-01-05   -230.25     -380.75      150.50     -252.99%
2023-01-06    180.75      411.00      230.25      178.48%
```

### 3. Handle Edge Cases

**Zero/Near-Zero PnL Protection:**
- If `|total_pnl[t-1]| < $100`, we use `$100` as the base
- This prevents extreme percentage spikes when crossing zero
- Example: Going from $1 to $101 would be 10,000% without this protection

**Sign Handling:**
- We use `abs(total_pnl[t-1])` to avoid sign issues
- A change from -$100 to -$200 is calculated as: -100 / 100 = -100%
- This represents a 100% increase in losses

### 4. Align Time Series

Create a DataFrame with all symbols' percentage returns aligned by timestamp:

```python
             AAPL     MSFT    GOOGL
2023-01-04  150.50   120.30    80.50
2023-01-05 -252.99   -50.25   110.75
2023-01-06  178.48    75.60   -35.20
...
```

Any rows with NaN values (first row after `.diff()` or missing data) are dropped.

### 5. Calculate Pearson Correlation

For each pair of symbols, calculate the Pearson correlation coefficient:

```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)

Where:
- Cov(X,Y) = E[(X - μ_X)(Y - μ_Y)]
- σ_X = standard deviation of X
- σ_Y = standard deviation of Y
- μ_X, μ_Y = means of X and Y
```

**Properties:**
- Range: -1 to +1
- +1: Perfect positive correlation (move together)
- 0: No linear relationship
- -1: Perfect negative correlation (move opposite)

**Result:**
```
          AAPL   MSFT  GOOGL
AAPL      1.00   0.65  -0.12
MSFT      0.65   1.00   0.23
GOOGL    -0.12   0.23   1.00
```

## Why Percentage Returns?

### Advantages

1. **Size Normalization**: A $100 gain on a $1000 position (10%) is comparable to a $1000 gain on a $10000 position (10%)

2. **Position-Size Independent**: Correlation reflects co-movement patterns, not absolute dollar amounts

3. **Interpretability**:
   - "These symbols both tend to gain/lose 5% together"
   - vs. "These symbols both gain/lose $500 together" (depends on size)

4. **Statistical Validity**: Percentage returns are more stationary than dollar PnL, better for correlation analysis

### Example Scenario

**Portfolio:**
- AAPL: Large position ($10,000)
- MSFT: Small position ($1,000)

**Market moves 2% up:**

**Absolute Dollar PnL:**
```
AAPL: +$200
MSFT: +$20
Correlation ≈ 0.99 (both positive, but AAPL dominates)
```

**Percentage PnL:**
```
AAPL: +2%
MSFT: +2%
Correlation = 1.00 (perfect correlation)
```

The percentage-based correlation correctly identifies that these symbols move together perfectly, regardless of position size.

## Interpretation Guide

### Correlation Values

- **0.8 to 1.0**: Very strong positive correlation
  - Symbols move together very consistently
  - Limited diversification benefit
  - Example: Tech stocks in same sector

- **0.5 to 0.8**: Strong positive correlation
  - Symbols tend to move together
  - Some diversification benefit
  - Example: Large-cap growth stocks

- **0.2 to 0.5**: Moderate positive correlation
  - Symbols sometimes move together
  - Good diversification benefit
  - Example: Different sectors with market exposure

- **-0.2 to 0.2**: Low/no correlation
  - Symbols move independently
  - Excellent diversification benefit
  - Example: Unrelated markets

- **-0.5 to -0.2**: Moderate negative correlation
  - Symbols sometimes move opposite
  - Strong diversification benefit
  - Example: Gold vs. stocks

- **-1.0 to -0.5**: Strong negative correlation
  - Symbols tend to move opposite
  - Natural hedge
  - Example: Long and short positions

### Diversification Assessment

A well-diversified portfolio should have:
- Average pairwise correlation < 0.5
- Some negative correlations
- No extreme correlations (|ρ| > 0.9) unless intentional

**Example:**
```
          AAPL   TSLA   GLD
AAPL      1.00   0.65  -0.15
TSLA      0.65   1.00  -0.10
GLD      -0.15  -0.10   1.00

Average correlation: (0.65 + (-0.15) + (-0.10)) / 3 = 0.13
✓ Good diversification
```

## Implementation Notes

### Minimum Base ($100)

The `min_base = $100` parameter can be adjusted based on:
- Average position sizes
- Strategy volatility
- Desired sensitivity

**Trade-offs:**
- **Higher min_base**: More stable, less sensitive to small PnL changes
- **Lower min_base**: More sensitive, captures small position dynamics

### Missing Data Handling

- Symbols with no trades/positions appear as all-zero PnL → excluded from correlation
- Timestamps where some symbols have no data → row dropped from correlation calculation
- Only overlapping time periods are used for correlation

### Correlation vs. Causation

**Important:** High correlation does NOT imply causation!

- Correlation measures co-movement, not causal relationship
- Both symbols might be driven by a third factor (e.g., market sentiment)
- Use correlation for diversification assessment, not for predicting one symbol from another

## Code Reference

**Location:** `backt/risk/metrics.py`

**Method:** `MetricsEngine.calculate_returns_correlation_matrix()`

**Key Steps:**
```python
# 1. Calculate percentage returns
pnl_change = pnl_series - pnl_prev
pnl_base = max(abs(pnl_prev), min_base)
pct_returns = (pnl_change / pnl_base) * 100

# 2. Create aligned DataFrame
returns_df = pd.DataFrame({symbol: pct_returns for symbol in symbols})
returns_df = returns_df.dropna()

# 3. Calculate Pearson correlation
correlation_matrix = returns_df.corr(method='pearson')
```

## Example Usage

```python
# Run backtest
result = backtester.run(strategy, universe=['AAPL', 'MSFT', 'GOOGL'])

# Get correlation matrix
report = PerformanceReport(result)
corr_matrix = report.get_correlation_matrix()

# Analyze diversification
print(corr_matrix)
report.plot_correlation_heatmap()

# Get average correlation (exclude diagonal)
import numpy as np
mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
avg_corr = corr_matrix.where(mask).stack().mean()
print(f"Average pairwise correlation: {avg_corr:.2f}")
```

## Advanced Analysis

### Rolling Correlation

Track how correlations change over time:

```python
# Get per-symbol equity curves
curves = result.per_symbol_equity_curves

# Calculate rolling correlation (30-day window)
returns_df = pd.DataFrame({
    symbol: curves[symbol]['total_pnl'].pct_change()
    for symbol in curves.keys()
})

rolling_corr = returns_df['AAPL'].rolling(30).corr(returns_df['MSFT'])
rolling_corr.plot(title='AAPL-MSFT Rolling 30-Day Correlation')
```

### Conditional Correlation

Correlation during different market conditions:

```python
# Split by overall portfolio performance
overall_returns = result.equity_curve['total_equity'].pct_change()
bull_periods = overall_returns > 0
bear_periods = overall_returns < 0

# Calculate correlation in bull vs. bear markets
corr_bull = returns_df[bull_periods].corr()
corr_bear = returns_df[bear_periods].corr()
```

## References

- Pearson, K. (1895). "Notes on regression and inheritance in the case of two parents"
- Modern Portfolio Theory: Markowitz, H. (1952). "Portfolio Selection"
- Correlation in financial markets: Campbell, J. Y., et al. (2001). "Have individual stocks become more volatile?"
