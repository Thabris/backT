# CPCV User Guide: Finding the Best Trading Strategies

## 📖 Complete Guide to Using Combinatorial Purged Cross-Validation

This guide explains **step-by-step** how to use CPCV to develop robust, non-overfit trading strategies that work in real markets.

---

## 🎯 Quick Start: 3-Step Process

### Step 1: Configure Your Backtest
1. Go to **"⚙️ Configuration"** tab
2. Set your date range (e.g., 2015-01-01 to 2023-12-31)
3. Set initial capital (e.g., $100,000)
4. Configure universe (e.g., SPY, QQQ, TLT, GLD)
5. Click **"💾 Save Configuration"**

### Step 2: Select Strategy
1. Go to **"📈 Strategy"** tab
2. Choose a strategy from the dropdown
3. Set initial parameters (start with defaults)
4. Click **"🚀 Run Backtest"**

### Step 3: Validate with CPCV
1. Go to **"🔬 CPCV Validation"** tab
2. Keep default settings (10 folds, 2 test splits = 45 paths)
3. Click **"🚀 Run CPCV Validation"**
4. Review results

---

## 📊 Understanding the Metrics

### 1. PBO (Probability of Backtest Overfitting)

**What it measures:** The likelihood that your best in-sample result is due to luck/overfitting rather than skill.

**How to interpret:**
- **PBO < 0.3 (Green):** Excellent! Very low overfitting risk
- **PBO 0.3-0.5 (Orange):** Good, acceptable overfitting risk
- **PBO > 0.5 (Red):** High overfitting risk - strategy likely won't work out-of-sample

**Real example from our test:**
```
PBO: 0.00% ✓ Excellent
```
This means the strategy showed ZERO evidence of overfitting across all 45 paths.

### 2. DSR (Deflated Sharpe Ratio)

**What it measures:** The Sharpe ratio adjusted for multiple testing and non-normal returns.

**How to interpret:**
- **DSR > 2.0 (Green):** Highly statistically significant
- **DSR > 1.0 (Orange):** Statistically significant at 84% confidence
- **DSR < 1.0 (Red):** Not statistically significant
- **DSR < 0 (Red):** Negative adjusted performance

**Real example:**
```
DSR: -9.509 (Poor)
```
This is negative because the strategy has negative Sharpe. BUT this is still valuable - it tells us the poor performance is REAL, not due to bad luck!

### 3. Performance Degradation

**What it measures:** How much performance degrades from in-sample to out-of-sample.

**How to interpret:**
- **< 10% (Green):** Excellent consistency
- **10-20% (Orange):** Good, acceptable degradation
- **> 30% (Red):** Poor, too much degradation

**Real example:**
```
Degradation: 0.0% ✓ Excellent
```
Perfect consistency - the strategy performs the same across all paths.

### 4. Sharpe Stability

**What it measures:** How consistent the Sharpe ratio is across different validation paths.

**How to interpret:**
- **> 5.0 (Green):** Very stable/consistent
- **2.0-5.0 (Orange):** Moderately stable
- **< 2.0 (Red):** Unstable, inconsistent

**Real example:**
```
Stability: 4.71 (Good)
```
The strategy is consistent - it doesn't have wildly different results on different data splits.

---

## 🔬 Complete Workflow: From Idea to Validated Strategy

### Phase 1: Strategy Development (Weeks 1-2)

**Goal:** Develop your initial strategy idea

1. **Start with a Hypothesis**
   - Example: "Momentum works for equity ETFs"
   - Example: "Mean reversion works during high VIX"

2. **Implement Basic Version**
   - Code the strategy with reasonable defaults
   - Test on one symbol (e.g., SPY)
   - Use "📈 Strategy" tab for initial testing

3. **Initial Backtest**
   - Run basic backtest (NOT CPCV yet)
   - Check if Sharpe > 0
   - Check if drawdown is reasonable
   - Look at trade frequency

**Decision Point:** If Sharpe < 0 or max drawdown > 50%, rethink the strategy. Don't try to optimize a fundamentally broken strategy.

---

### Phase 2: Parameter Optimization (Weeks 3-4)

**Goal:** Find optimal parameters WITHOUT overfitting

#### Method 1: Manual Grid Search

1. **Define Parameter Ranges**
   ```python
   # Example for Momentum Strategy
   Parameters to test:
   - lookback: 10, 20, 30, 50, 100 days
   - threshold: 0.0%, 1.0%, 2.0%

   Total combinations: 5 × 3 = 15
   ```

2. **Test Each Combination**
   - For EACH parameter set:
     - Run regular backtest in "Strategy" tab
     - Record Sharpe, Return, Max DD
   - Create a spreadsheet with results

3. **Select Top 3-5 Candidates**
   - Choose based on:
     - Highest Sharpe ratio
     - Acceptable drawdown (< 30%)
     - Reasonable trade frequency

---

### Phase 3: CPCV Validation (Week 5)

**Goal:** Validate that top candidates aren't overfit

#### For Each Top Candidate:

1. **Run CPCV Validation**
   - Go to "🔬 CPCV Validation" tab
   - Settings:
     - Folds: 10 (or 5 for faster testing)
     - Test Splits: 2
     - Purge: 5%
     - Embargo: 2%
   - Click "Run CPCV Validation"

2. **Evaluate Results**

   **Pass Criteria (ALL must be true):**
   - ✅ PBO < 0.5
   - ✅ DSR > 1.0 (or > 0 if you accept lower confidence)
   - ✅ Degradation < 30%
   - ✅ Stability > 2.0

   **Example:**
   ```
   Parameter Set A (lookback=20, threshold=0.01):
   PBO: 0.35 ✓
   DSR: 1.8 ✓
   Degradation: 12% ✓
   Stability: 3.2 ✓
   VERDICT: PASS - Deploy this!

   Parameter Set B (lookback=100, threshold=0.0):
   PBO: 0.75 ✗
   DSR: 0.5 ✗
   Degradation: 45% ✗
   Stability: 1.2 ✗
   VERDICT: FAIL - Overfit, don't use
   ```

3. **Review Path Distributions**
   - Look at the "Sharpe Ratio Across All Validation Paths" chart
   - **Good:** All bars clustered around the mean
   - **Bad:** Wide variation, many outliers

---

### Phase 4: Final Validation (Week 6)

**Goal:** Confirm strategy works on completely unseen data

1. **Reserve Hold-Out Data**
   - Take last 1-2 years of data
   - Never use this during optimization!

2. **Run Final Test**
   - Use your CPCV-validated parameters
   - Run backtest on hold-out period only
   - Compare results to CPCV expectations

3. **Decision Matrix**

   | CPCV Mean Sharpe | Hold-Out Sharpe | Decision |
   |------------------|-----------------|----------|
   | 1.5              | 1.4             | ✅ Deploy! |
   | 1.5              | 0.8             | ⚠️ Cautious |
   | 1.5              | -0.5            | ❌ Don't deploy |
   | -0.5             | -0.4            | ✅ Strategy correctly identified as poor |

---

## 💡 Pro Tips: Finding Robust Strategies

### ✅ DO

1. **Start Simple**
   - Simple strategies are more robust
   - Example: "Buy when price > 200-day MA" often works better than complex multi-indicator systems

2. **Use Logical Parameters**
   - 20 days (≈1 month) makes sense
   - 252 days (≈1 year) makes sense
   - 73 days makes no sense - this is data mining!

3. **Test Multiple Universes**
   - If strategy works on SPY, test on QQQ
   - If it works on both, it's likely robust

4. **Look for Consistent Performance**
   - Rather have Sharpe=1.0 stable across all 45 paths
   - Than Sharpe=2.0 on 10 paths and Sharpe=-1.0 on others

5. **Increase CPCV Rigor for Deployment**
   - For research: 5 folds, 2 test splits (10 paths)
   - For live trading: 10 folds, 2 test splits (45 paths)
   - For serious capital: 15 folds, 3 test splits (455 paths)

### ❌ DON'T

1. **Don't Optimize Too Much**
   - Testing 1,000 parameter combinations = guaranteed overfitting
   - Test 10-30 combinations max

2. **Don't Ignore Negative Results**
   - If CPCV shows PBO=0.8, the strategy is overfit
   - Don't try to "fix" it with more optimization - that makes it worse!

3. **Don't Use Tiny Purge/Embargo**
   - For daily data: purge ≥ 5%, embargo ≥ 2%
   - For weekly data: purge ≥ 10%, embargo ≥ 5%
   - This prevents data leakage

4. **Don't Mix Data**
   - Optimization period: 2015-2022
   - CPCV validation: Same period (split internally)
   - Hold-out: 2023-2024
   - NEVER use hold-out data during optimization!

5. **Don't Cherry-Pick**
   - If you test 5 strategies and only 1 passes CPCV, you still need to adjust DSR for 5 trials
   - Be honest about how many things you tested

---

## 📈 Real-World Example Workflow

### Scenario: You want to develop a momentum strategy for ETFs

#### Week 1: Research & Initial Development
```
Hypothesis: "12-month momentum works for multi-asset ETFs"
Universe: SPY, QQQ, TLT, GLD
Period: 2010-2023
```

**Results from initial backtest:**
- Sharpe: 0.8
- Max DD: -25%
- Annual Return: 12%

**Decision:** Promising! Continue to optimization.

#### Week 2-3: Parameter Optimization
Test 15 combinations:
```
lookback: 3, 6, 9, 12, 18 months
threshold: 0%, 1%, 2%
```

**Top 3 Results:**
1. lookback=12, threshold=1% → Sharpe 1.2
2. lookback=9, threshold=0% → Sharpe 1.1
3. lookback=6, threshold=2% → Sharpe 1.0

#### Week 4: CPCV Validation

**Candidate #1 (lookback=12, threshold=1%):**
```
CPCV Results (45 paths):
- Mean Sharpe: 1.05 ± 0.15
- PBO: 0.25 ✓
- DSR: 1.9 ✓
- Degradation: 8% ✓
- Stability: 5.2 ✓

VERDICT: PASS ✅
```

**Candidate #2 (lookback=9, threshold=0%):**
```
CPCV Results:
- Mean Sharpe: 0.85 ± 0.40
- PBO: 0.55 ✗
- DSR: 0.7 ✗
- Degradation: 35% ✗
- Stability: 1.8 ✗

VERDICT: FAIL ❌ (Overfit!)
```

**Candidate #3 (lookback=6, threshold=2%):**
```
CPCV Results:
- Mean Sharpe: 0.95 ± 0.12
- PBO: 0.30 ✓
- DSR: 1.5 ✓
- Degradation: 5% ✓
- Stability: 6.1 ✓

VERDICT: PASS ✅
```

#### Week 5: Final Validation

Test Candidate #1 and #3 on 2023-2024 hold-out:

**Candidate #1:**
- Hold-out Sharpe: 1.1
- Close to CPCV mean of 1.05 ✓
- Deploy!

**Candidate #3:**
- Hold-out Sharpe: 0.4
- Much worse than CPCV mean of 0.95 ✗
- Don't deploy

#### Final Decision
Deploy Candidate #1 (lookback=12, threshold=1%) with confidence!

---

## 🚨 Common Pitfalls & How to Avoid Them

### Pitfall #1: "My strategy has Sharpe=3 but PBO=0.9"

**Problem:** Strategy is overfit. The great backtest is due to luck.

**Solution:**
- Simplify the strategy
- Reduce number of parameters
- Test on different universes
- Accept that most strategies won't have Sharpe > 2

### Pitfall #2: "CPCV takes too long to run"

**Problem:** 45 paths × 10 parameter sets = 450 backtests

**Solutions:**
- Start with 5 folds (10 paths) for initial screening
- Use 10 folds (45 paths) only for final candidates
- Reduce date range for parameter search
- Use faster data frequency (weekly instead of daily)

### Pitfall #3: "All my strategies fail CPCV"

**Problem:** You're either overfitting or testing bad strategies

**Solutions:**
- Start with known-good strategies (simple momentum, value)
- Test fewer parameter combinations
- Use simpler strategies
- Check if your universe makes sense
- Verify data quality

### Pitfall #4: "CPCV shows degradation=50%"

**Problem:** In-sample and out-of-sample results are very different

**Causes:**
- Data leakage (not enough purging/embargo)
- Look-ahead bias in strategy
- Overfitted parameters

**Solutions:**
- Increase purge_pct to 10%
- Increase embargo_pct to 5%
- Review strategy code for look-ahead bias
- Use fewer parameters

---

## 📚 Summary: The Complete Checklist

### Before Deployment Checklist

- [ ] Strategy has logical economic rationale
- [ ] Tested on relevant universe (not random stocks)
- [ ] Parameters make intuitive sense
- [ ] Regular backtest shows Sharpe > 0.5
- [ ] Regular backtest shows Max DD < 50%
- [ ] CPCV PBO < 0.5
- [ ] CPCV DSR > 1.0 (or > 0 for acceptable confidence)
- [ ] CPCV Degradation < 30%
- [ ] CPCV Stability > 2.0
- [ ] Hold-out test confirms CPCV results
- [ ] Transaction costs included and reasonable
- [ ] Slippage assumptions are realistic
- [ ] Strategy trades frequently enough (> 20 trades/year)
- [ ] Strategy doesn't trade too much (< 1000 trades/year for daily)

### If ALL checkboxes are ✅ → Deploy with confidence!
### If ANY checkbox is ❌ → Revise or abandon strategy

---

## 🎓 Advanced Topics

### Custom CPCV Configurations

**For different strategy types:**

1. **High-Frequency (Daily Rebalancing)**
   ```
   n_splits: 15
   n_test_splits: 3
   purge_pct: 0.10
   embargo_pct: 0.05
   ```

2. **Low-Frequency (Monthly Rebalancing)**
   ```
   n_splits: 5
   n_test_splits: 1
   purge_pct: 0.03
   embargo_pct: 0.01
   ```

3. **Options/Derivatives (Path-Dependent)**
   ```
   n_splits: 10
   n_test_splits: 2
   purge_pct: 0.15
   embargo_pct: 0.10
   ```

### Interpreting Edge Cases

**Case 1: Negative Sharpe, Low PBO**
- Strategy is consistently bad (not overfit)
- This is actually GOOD validation!
- It correctly identified a poor strategy

**Case 2: Positive Sharpe, High PBO**
- Strategy is overfit
- Don't deploy - results are luck

**Case 3: High Stability, Poor DSR**
- Strategy is consistent but not significant
- Consider: larger universe, longer period

---

## 💼 Production Deployment

Once your strategy passes CPCV:

1. **Start with Small Capital**
   - Even if backtest shows Sharpe=2, start with 10% of intended capital
   - Monitor for 3-6 months

2. **Track Live PBO**
   - Re-run CPCV every quarter
   - If PBO starts increasing → strategy is degrading

3. **Set Stop-Loss**
   - If live Sharpe < CPCV mean - 2×std for 6 months → stop strategy
   - Example: CPCV mean=1.0, std=0.2 → stop if Sharpe < 0.6

4. **Regular Revalidation**
   - Every 6 months: re-run CPCV with latest data
   - If metrics degrade, retire strategy

---

## 🔗 Additional Resources

- **Academic Paper:** Bailey & Lopez de Prado (2014) - "The Probability of Backtest Overfitting"
- **Academic Paper:** Bailey & Lopez de Prado (2014) - "The Deflated Sharpe Ratio"
- **Book:** "Advances in Financial Machine Learning" by Marcos Lopez de Prado
- **BackT Documentation:** See README.md for more examples

---

**Remember:** The goal of CPCV is not to find the "perfect" strategy (it doesn't exist), but to identify strategies that are **robust, consistent, and likely to work out-of-sample**.

Good luck finding winning strategies! 🚀
