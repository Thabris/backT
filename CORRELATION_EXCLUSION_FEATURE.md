# Correlation Exclusion Feature - `--exclude` Option

## Overview

Added `--exclude` option to `rank_symbols_by_strategy.py` to automatically filter out highly correlated symbols from the top performers, keeping only the best performer from each correlated pair. This helps build diversified portfolios.

## Feature Summary

**Purpose:** Prevent portfolio concentration by excluding symbols that move together (high correlation).

**Algorithm:**
1. Rank all symbols by Sortino ratio (best to worst)
2. Select top N symbols
3. Calculate pairwise return correlations
4. Iterate through symbols in rank order (best first)
5. For each symbol, check correlation with all remaining symbols
6. If correlation > threshold, exclude the worse performer
7. Final list may have fewer than N symbols

**Key Benefit:** Maximizes diversification while keeping best performers

---

## Usage

### Basic Syntax

```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --strategy macd_crossover \
  --top 10 \
  --exclude 80
```

### Threshold Formats (All Equivalent)

```bash
--exclude 80              # Integer percentage
--exclude 0.8             # Decimal (0-1)
--exclude 80%             # Percentage with %
--exclude correlation>80  # Explicit format
--exclude correlation>0.8 # Explicit decimal
```

### Combined with Other Options

```bash
# Exclude + Save as Book
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy ma_crossover_long_only \
  --top 20 --exclude 75 \
  --save "Diversified_Top20"

# Exclude + Checkpoint + Workers
python rank_symbols_by_strategy.py \
  --start 2020-01-01 --end 2024-12-31 \
  --strategy kalman_ma_crossover_long_only \
  --top 15 --exclude 85 \
  --workers 5 --checkpoint progress.json
```

---

## Example Output

### Without `--exclude`

```
TOP 10 SYMBOLS BY SORTINO RATIO
Rank  Symbol   Sortino   Correlation Issues
1     VCR      1.093
2     DBA      1.033
3     XLY      0.853     86.3% with VCR (highly correlated!)
4     VFH      0.850
5     XLU      0.849
6     XLF      0.767     90.6% with VFH (highly correlated!)
7     VPU      0.708     86.9% with XLU (highly correlated!)
8     IVV      0.661
9     NTSX     0.615     82.6% with IVV (highly correlated!)
10    FXI      0.593
```

**Problem:** Portfolio has 4 pairs of highly correlated symbols → less diversification

### With `--exclude 80`

```
CORRELATION FILTERING (Threshold: 80%)
Initial top symbols: 10
Excluded (correlated): 4
Final symbols: 6

Excluded Symbols:
  XLY      excluded (corr= 86.3% with VCR, kept better performer)
  XLF      excluded (corr= 90.6% with VFH, kept better performer)
  VPU      excluded (corr= 86.9% with XLU, kept better performer)
  NTSX     excluded (corr= 82.6% with IVV, kept better performer)

TOP 6 SYMBOLS BY SORTINO RATIO
Rank  Symbol   Sortino   Total Ret   Max Correlation
1     VCR      1.093     23.75%      48.3%
2     DBA      1.033     17.23%      6.0%
3     VFH      0.850     13.26%      28.3%
4     XLU      0.849     21.85%      15.1%
5     IVV      0.661     10.14%      48.3%
6     FXI      0.593     31.42%      16.0%

CORRELATION MATRIX
Symbol      VCR     DBA     VFH     XLU     IVV     FXI
VCR       1.000  -0.011   0.283   0.093   0.483   0.150
DBA      -0.011   1.000  -0.004   0.060   0.002   0.004
VFH       0.283  -0.004   1.000   0.151   0.254   0.019
XLU       0.093   0.060   0.151   1.000   0.086   0.008
IVV       0.483   0.002   0.254   0.086   1.000   0.160
FXI       0.150   0.004   0.019   0.008   0.160   1.000

Correlation Summary:
  Average correlation: 0.116  (was 0.583 before filtering)
  Min correlation:     -0.011
  Max correlation:     0.483  (was 0.906 before filtering)
```

**Benefit:** Portfolio now has much lower correlations → better diversification

---

## Implementation Details

### Function: `filter_correlated_symbols()`

**Location:** `rank_symbols_by_strategy.py:419-488`

**Logic:**
```python
def filter_correlated_symbols(top_symbols, correlation_threshold):
    # Start with all symbols in "keep" set
    symbols_to_keep = set(all_symbols)
    excluded_pairs = []

    # Iterate best to worst
    for i, (symbol1, metrics1) in enumerate(top_symbols):
        if symbol1 not in symbols_to_keep:
            continue  # Already excluded

        # Check correlation with remaining symbols
        for j, (symbol2, metrics2) in enumerate(top_symbols):
            if j <= i:
                continue  # Skip self and already compared

            if symbol2 not in symbols_to_keep:
                continue  # Already excluded

            correlation = get_correlation(symbol1, symbol2)

            # Exclude worse performer if correlated
            if abs(correlation) >= correlation_threshold:
                symbols_to_keep.remove(symbol2)
                excluded_pairs.append((symbol2, symbol1, correlation))

    return filtered_symbols, excluded_pairs
```

**Key Points:**
1. Uses absolute value of correlation (treats +90% and -90% the same)
2. Greedy algorithm - keeps best performer, removes all its correlated pairs
3. Order matters - processing best-first ensures we keep the strongest performers
4. Final count may be less than requested top N

### Function: `parse_correlation_threshold()`

**Location:** `rank_symbols_by_strategy.py:574-602`

**Handles:**
- Integers: `"80"` → 0.8
- Decimals: `"0.8"` → 0.8
- Percentages: `"80%"` → 0.8
- Prefixed: `"correlation>75"` → 0.75
- Validation: Ensures 0 ≤ threshold ≤ 1

---

## Use Cases

### 1. Portfolio Diversification
**Goal:** Build diversified multi-asset portfolio

```bash
python rank_symbols_by_strategy.py \
  --start 2023-01-01 --end 2024-12-31 \
  --strategy ma_crossover_long_only \
  --top 20 --exclude 70 \
  --save "Diversified_MA_Portfolio"
```

**Result:** 20 symbols filtered down to ~12-15 uncorrelated assets

### 2. Sector Exposure Management
**Goal:** Avoid overweight in specific sectors

```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy kalman_ma_crossover_long_only \
  --top 15 --exclude 75
```

**Example:** If top 10 includes XLF, VFH, and JPM (all financials, highly correlated), only the best one is kept.

### 3. Risk Management
**Goal:** Reduce portfolio volatility through diversification

```bash
python rank_symbols_by_strategy.py \
  --start 2022-01-01 --end 2024-12-31 \
  --strategy rsi_mean_reversion \
  --top 25 --exclude 80 \
  --save "Low_Correlation_RSI"
```

**Benefit:** Lower portfolio correlation → lower portfolio volatility

### 4. Multi-Strategy Rotation
**Goal:** Create symbol lists for different strategies

```bash
# Momentum strategy - diversified
python rank_symbols_by_strategy.py \
  --strategy macd_crossover --top 10 --exclude 75 \
  --save "MACD_Diversified"

# Mean reversion - diversified
python rank_symbols_by_strategy.py \
  --strategy rsi_mean_reversion --top 10 --exclude 75 \
  --save "RSI_Diversified"

# Load both in Streamlit, test independently or combined
```

---

## Best Practices

### Choosing the Threshold

| Threshold | Use Case | Expected Result |
|-----------|----------|-----------------|
| 60-70% | Maximum diversification | Keep only weakly correlated assets |
| 70-80% | Balanced approach | Remove very similar assets |
| 80-90% | Moderate filtering | Only remove nearly identical assets |
| >90% | Minimal filtering | Only remove duplicates (e.g., SPY vs IVV) |

**Recommendation:** Start with 75-80% for most portfolios

### Common Patterns

**Highly Correlated Pairs (Often Excluded Together):**
- SPY / VOO / IVV - All S&P 500 (>99% correlation)
- XLF / VFH - Both financials (>90% correlation)
- XLY / VCR - Both consumer discretionary (>85% correlation)
- GLD / IAU - Both gold (>99% correlation)
- TLT / IEF - Both treasuries (>80% correlation)

**Low Correlation Assets (Usually Kept Together):**
- Stocks vs Bonds (often negative)
- Commodities vs Equities (~0-30%)
- International vs US (~40-60%)
- Gold vs Stocks (~0-20%)

### Workflow Recommendations

**Step 1: Initial Ranking (No Filter)**
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy macd_crossover --top 20
```
→ Review correlation matrix to understand relationships

**Step 2: Apply Filter**
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy macd_crossover --top 20 --exclude 80
```
→ See how many symbols are excluded

**Step 3: Adjust Threshold**
```bash
# If too many excluded (e.g., 20 → 5), relax threshold
--exclude 85

# If too few excluded (e.g., 20 → 18), tighten threshold
--exclude 75
```

**Step 4: Save Book**
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy macd_crossover --top 20 --exclude 80 \
  --save "MACD_Optimized_2024"
```

---

## Advantages Over Manual Selection

### Before (Manual)
1. Run ranking script
2. Export results to Excel
3. Calculate correlation matrix
4. Manually identify correlated pairs
5. Manually remove worse performers
6. Create symbol list
7. Input into Streamlit

**Time:** 30-60 minutes, error-prone

### After (Automated)
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy macd_crossover --top 20 --exclude 80 \
  --save "Portfolio"
```

**Time:** 2-3 minutes, fully automated, reproducible

---

## Integration with Books

When using `--exclude` with `--save`:
- **Only filtered symbols are saved** to the book
- **Metadata includes:** Number of symbols excluded
- **Description shows:** "6 symbols (filtered from 10)"
- **Tags updated:** Shows actual count (e.g., `"top_6"` not `"top_10"`)

**Example Book:**
```json
{
  "name": "MACD_Diversified_2024",
  "symbols": ["VCR", "DBA", "VFH", "XLU", "IVV", "FXI"],
  "tags": ["ranked", "macd_crossover", "top_6"],
  "metadata": {
    "num_symbols": 6,
    "avg_sortino": 0.846,
    "correlation_filtered": true,
    "original_top_n": 10
  }
}
```

Load in Streamlit → Get diversified portfolio automatically!

---

## Performance Impact

**Computational Cost:** Minimal
- Correlation calculation: O(N²) where N = top symbols count
- Typically N = 10-20, so ~100-400 comparisons
- Adds <1 second to total runtime

**Memory:** Negligible
- Stores equity curves for top N only
- ~100KB for 20 symbols × 250 days

---

## Limitations

1. **Uses past correlations** - Not predictive of future correlations
2. **Single period** - Correlation calculated over entire backtest period
3. **Greedy algorithm** - Doesn't guarantee globally optimal diversification
4. **No sector awareness** - Purely correlation-based, doesn't consider sectors

**Mitigation:**
- Test on multiple time periods
- Combine with manual sector review
- Use walk-forward validation

---

## Testing

### Test Cases Passed

✅ Threshold parsing: All formats (80, 0.8, 80%, correlation>80)
✅ Correlation filtering: Correctly excludes correlated pairs
✅ Book saving: Saves filtered symbols (not original top N)
✅ Edge cases: Handles 0 exclusions, all exclusions
✅ Output formatting: Clear messaging about filtering

### Example Test Results

**Input:** Top 10, exclude >80%
**Output:** 6 symbols (4 excluded)
**Excluded:**
- XLY (86.3% with VCR) ✓
- XLF (90.6% with VFH) ✓
- VPU (86.9% with XLU) ✓
- NTSX (82.6% with IVV) ✓

**Final Correlation Matrix:**
- Max correlation: 48.3% (was 90.6%)
- Avg correlation: 11.6% (was 58.3%)

---

## Future Enhancements

### Potential Additions

1. **Clustering-based filtering**
   - Group correlated symbols into clusters
   - Keep N symbols from each cluster

2. **Multi-period correlation**
   - Calculate correlation across rolling windows
   - Exclude if correlated in multiple periods

3. **Sector constraints**
   - Add `--max-per-sector` option
   - Ensure sector diversification

4. **Correlation target**
   - Add `--target-correlation` option
   - Keep adding symbols until portfolio reaches target avg correlation

5. **Visualization**
   - Generate correlation heatmap
   - Show excluded pairs visually

---

## Summary

The `--exclude` option provides **automated diversification** for symbol selection:

✅ **Simple to use** - Single parameter
✅ **Flexible** - Multiple threshold formats
✅ **Effective** - Significantly reduces correlation
✅ **Integrated** - Works with books and checkpoints
✅ **Transparent** - Clear output showing what was excluded

**Bottom Line:** Build better diversified portfolios with one extra parameter!
