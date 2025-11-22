# Correlation Filtering with Backfilling Algorithm

## Overview

The `--exclude` option now uses a **backfilling algorithm** that automatically replaces excluded symbols with the next-best ranked symbols until either:
1. Target portfolio size is reached (e.g., 10 symbols), OR
2. All available candidates are exhausted

## Algorithm

### Previous Behavior (Before Backfilling)
```
1. Select top 10 symbols
2. Filter out correlated pairs
3. Result: 6 symbols (4 excluded)
→ Portfolio is smaller than requested
```

### New Behavior (With Backfilling)
```
1. Start with rank #1 symbol → Add to portfolio
2. Test rank #2 symbol → Check correlation with #1
   - If correlated: Exclude, move to #3
   - If not correlated: Add to portfolio
3. Test rank #3 symbol → Check correlation with all in portfolio
   - If correlated with ANY: Exclude, move to #4
   - If not correlated with ALL: Add to portfolio
4. Repeat until portfolio has 10 symbols OR run out of candidates
→ Portfolio reaches target size (when possible)
```

### Pseudocode
```python
portfolio = []
rank = 1

while len(portfolio) < target_size and candidates_available:
    candidate = get_symbol_at_rank(rank)

    # Check correlation with all existing portfolio members
    is_correlated = False
    for portfolio_member in portfolio:
        if correlation(candidate, portfolio_member) >= threshold:
            exclude(candidate)  # Too correlated
            is_correlated = True
            break

    if not is_correlated:
        portfolio.add(candidate)  # Passed correlation test

    rank += 1
```

---

## Example Walkthrough

### Request: Top 10 symbols, exclude >80% correlation

#### Iteration Process

| Rank | Symbol | Sortino | Action | Reason |
|------|--------|---------|--------|--------|
| 1 | VCR | 1.093 | ✅ **Add** | First symbol (no correlation test) |
| 2 | DBA | 1.033 | ✅ **Add** | -1.1% corr with VCR (< 80%) |
| 3 | XLY | 0.853 | ❌ **Exclude** | 86.3% corr with VCR (> 80%) |
| 4 | VFH | 0.850 | ✅ **Add** | 28.3% corr with VCR, -0.4% with DBA |
| 5 | XLU | 0.849 | ✅ **Add** | Below threshold with all |
| 6 | XLF | 0.767 | ❌ **Exclude** | 90.6% corr with VFH (> 80%) |
| 7 | VPU | 0.708 | ❌ **Exclude** | 86.9% corr with XLU (> 80%) |
| 8 | IVV | 0.661 | ✅ **Add** | Below threshold with all |
| 9 | NTSX | 0.615 | ❌ **Exclude** | 82.6% corr with IVV (> 80%) |
| 10 | FXI | 0.593 | ✅ **Add** | Below threshold with all |
| 11 | XLRE | 0.589 | ✅ **Add** | ← **Backfill #1** |
| 12 | EEM | 0.570 | ❌ **Exclude** | 82.0% corr with FXI |
| 13 | VOO | 0.564 | ❌ **Exclude** | 96.2% corr with IVV |
| 14 | TMV | 0.541 | ✅ **Add** | ← **Backfill #2** |
| 15 | IAU | 0.525 | ✅ **Add** | ← **Backfill #3** |
| 16 | SPY | 0.522 | ❌ **Exclude** | 95.9% corr with IVV |
| 17 | MCHI | 0.493 | ❌ **Exclude** | 96.4% corr with FXI |
| 18 | MTUM | 0.490 | ✅ **Add** | ← **Backfill #4** |

**Result:** Portfolio of 10 symbols achieved after testing 18 candidates (ranks 1-18)

---

## Output Messages

### Successful Target Reached
```
CORRELATION FILTERING WITH BACKFILL (Threshold: 80%)
Target symbols: 10
Available candidates: 77
Candidates tested: 18 (ranks 1-18)
Symbols excluded: 8
Final portfolio: 10 symbols

Excluded Symbols (by rank order):
  XLY      excluded (corr= 86.3% with VCR already in portfolio)
  XLF      excluded (corr= 90.6% with VFH already in portfolio)
  VPU      excluded (corr= 86.9% with XLU already in portfolio)
  NTSX     excluded (corr= 82.6% with IVV already in portfolio)
  EEM      excluded (corr= 82.0% with FXI already in portfolio)
  VOO      excluded (corr= 96.2% with IVV already in portfolio)
  SPY      excluded (corr= 95.9% with IVV already in portfolio)
  MCHI     excluded (corr= 96.4% with FXI already in portfolio)

ANALYSIS COMPLETE
Correlation filtering: enabled
  - Candidates evaluated: 18 (ranks 1-18)
  - Symbols excluded: 8
  - Portfolio size: 10 (target: 10)
  [OK] Target reached: 10 uncorrelated symbols
```

### Partial Fill (Exhausted Candidates)
```
CORRELATION FILTERING WITH BACKFILL (Threshold: 60%)
Target symbols: 30
Available candidates: 77
Candidates tested: 77 (ranks 1-77)
Symbols excluded: 50
Final portfolio: 27 symbols

[WARNING] Could not reach target of 30 symbols - exhausted all candidates

ANALYSIS COMPLETE
Correlation filtering: enabled
  - Candidates evaluated: 77 (ranks 1-77)
  - Symbols excluded: 50
  - Portfolio size: 27 (target: 30)
  [WARNING] Partial fill: 27/30 (exhausted candidates)
```

---

## Benefits

### 1. **Always Get Target Size (When Possible)**

**Before:**
```bash
--top 10 --exclude 80
→ Result: 6 symbols (40% shortfall)
```

**After:**
```bash
--top 10 --exclude 80
→ Result: 10 symbols (target reached by backfilling)
```

### 2. **Maximize Diversification**

- Tests symbols iteratively, not in batches
- Each new symbol must pass correlation test with **all** existing portfolio members
- Ensures no hidden correlations between portfolio members

### 3. **Preserve Quality**

- Still prioritizes highest-ranked (best Sortino) symbols
- Backfilling maintains rank order
- Portfolio #10 might be rank #18, but it's still better than rank #50

### 4. **Transparent Process**

- Shows exactly how many candidates were tested
- Lists all excluded symbols with reasons
- Clear warning if target cannot be reached

---

## Use Cases

### Case 1: Building Diversified Multi-Asset Portfolio
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy ma_crossover_long_only \
  --top 15 --exclude 75

# Without backfilling: Might get 8-10 symbols
# With backfilling: Get full 15 symbols (if available)
```

### Case 2: Tight Correlation Constraint
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy macd_crossover \
  --top 20 --exclude 60

# Very tight threshold (60%)
# Might only achieve 15-18 symbols
# Algorithm tests ALL candidates to maximize portfolio size
```

### Case 3: Large Portfolio
```bash
python rank_symbols_by_strategy.py \
  --start 2024-01-01 --end 2024-12-31 \
  --strategy kalman_ma_crossover_long_only \
  --top 30 --exclude 80

# Requesting 30 symbols
# Backfilling ensures we get as close to 30 as possible
```

---

## Performance Characteristics

### Computational Complexity

**Correlation Matrix Calculation:**
- Done once for all symbols: O(N × T)
  - N = number of valid symbols (~77)
  - T = number of time periods (~250 days)
- Takes ~1 second for 77 symbols

**Backfilling Loop:**
- Worst case: O(Target × Candidates)
  - Target = requested portfolio size (e.g., 10)
  - Candidates = number tested before reaching target
- Each iteration checks correlation with portfolio members: O(Portfolio Size)
- Total: O(Target²) in worst case
- Typical: O(Target × 2) = Very fast (milliseconds)

**Total Runtime:**
- Negligible compared to backtesting (1-2 seconds vs 60+ seconds)

### Memory Usage

- Stores equity curves for all valid symbols
- ~100KB per symbol × 77 symbols = ~8MB
- Negligible

---

## Comparison: Before vs After

### Scenario: Top 10 with 80% threshold

| Metric | Before Backfilling | After Backfilling |
|--------|-------------------|-------------------|
| Candidates tested | 10 (ranks 1-10) | 18 (ranks 1-18) |
| Symbols excluded | 4 | 8 |
| Final portfolio size | **6 symbols** | **10 symbols** |
| Target reached? | ❌ No (60% fill) | ✅ Yes (100% fill) |
| Avg correlation | 11.6% | 12.0% |
| Processing time | <1 second | <1 second |

**Verdict:** Backfilling achieves target size with minimal correlation increase

---

## Edge Cases

### 1. Insufficient Candidates
```bash
--top 50 --exclude 70
# Only 77 valid symbols available
# After filtering, might only get 30-40 symbols
→ [WARNING] Partial fill: 35/50 (exhausted candidates)
```

### 2. Very Tight Threshold
```bash
--top 20 --exclude 50
# 50% correlation is very restrictive
# Most assets have 40-60% correlation
→ [WARNING] Partial fill: 8/20 (exhausted candidates)
```

### 3. All Symbols Pass
```bash
--top 10 --exclude 95
# 95% threshold is very loose
# First 10 symbols all pass
→ Candidates tested: 10 (ranks 1-10)
→ Symbols excluded: 0
→ [OK] Target reached: 10 uncorrelated symbols
```

### 4. No Valid Symbols
```bash
--top 10 --exclude 80
# But date range has only 2 symbols with valid trades
→ ERROR: No valid results after filtering
```

---

## Implementation Details

### Function: `filter_correlated_symbols_with_backfill()`

**Location:** `rank_symbols_by_strategy.py:436-513`

**Signature:**
```python
def filter_correlated_symbols_with_backfill(
    all_sorted_symbols: List[Tuple[str, Dict]],  # Full ranked list
    target_count: int,                             # Requested portfolio size
    correlation_threshold: float                   # Threshold (0-1)
) -> Tuple[List[Tuple[str, Dict]], List[Tuple], int]:
    """
    Returns:
        - filtered_symbols: Final portfolio
        - excluded_pairs: List of (excluded, kept, correlation)
        - candidates_tested: Number of ranks evaluated
    """
```

**Key Features:**
1. **Single correlation matrix** calculated once for all symbols
2. **Iterative portfolio building** - add one symbol at a time
3. **Greedy algorithm** - first symbol passing test is added
4. **Order preservation** - maintains rank order (best first)
5. **Early termination** - stops when target reached OR candidates exhausted

---

## Real-World Example

### Building a 10-Symbol Diversified Portfolio

**Goal:** 10 symbols, max 80% correlation

**Result:**
```
Final Portfolio (10 symbols from testing 18 ranks):
1. VCR   - Consumer Discretionary ETF
2. DBA   - Agriculture Commodities
3. VFH   - Financials ETF
4. XLU   - Utilities ETF
5. IVV   - S&P 500 ETF
6. FXI   - China Equities
7. XLRE  - Real Estate ETF
8. TMV   - Inverse Treasury (3x)
9. IAU   - Gold ETF
10. MTUM - Momentum Factor ETF

Excluded (8 symbols):
- XLY (too similar to VCR)
- XLF (too similar to VFH)
- VPU (too similar to XLU)
- NTSX (too similar to IVV)
- EEM (too similar to FXI)
- VOO (too similar to IVV)
- SPY (too similar to IVV)
- MCHI (too similar to FXI)

Correlation Matrix:
Max correlation: 75.0% (IVV-MTUM)
Avg correlation: 12.0%
→ Highly diversified portfolio!
```

**Asset Class Breakdown:**
- Equities: 5 (VCR, VFH, IVV, FXI, MTUM)
- Sectors: 2 (XLU, XLRE)
- Commodities: 2 (DBA, IAU)
- Fixed Income: 1 (TMV - inverse)

**Geographic Diversification:**
- US: 7 symbols
- International: 1 symbol (FXI - China)
- Commodities: 2 (global)

---

## Summary

The **backfilling algorithm** transforms correlation filtering from a **filtering-only** operation to an **optimization** problem:

**Previous:** "Remove correlated symbols from top N"
→ Result: Smaller portfolio

**New:** "Build the largest possible portfolio of top-ranked symbols below correlation threshold"
→ Result: Target-sized portfolio (when possible)

**Key Innovation:** Instead of giving up after filtering, the algorithm keeps trying lower-ranked symbols until it succeeds or exhausts all options.

This ensures you get the **requested portfolio size** with **maximum diversification** while still **prioritizing quality** (highest-ranked symbols).
