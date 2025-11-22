# Books Feature - Quick Start Guide

## 🎯 What is a Book?

A **book** is a saved strategy configuration (strategy + parameters + symbols) that you can reuse across different time periods.

## 📖 Quick Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Configure → 2. Run Backtest → 3. Save Book          │
│ 4. Load Book → 5. Run on Different Dates → 6. Compare  │
└─────────────────────────────────────────────────────────┘
```

---

## 💾 Saving a Book (Results Tab)

### Basic Save (3 Steps)

After running a backtest, scroll to the bottom of the **Results** tab:

```
┌────────────────────────────────────────────────────────┐
│ 📚 Save as Book                                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│ Will save: ma_crossover_long_only • 3 params • 4 symbols │
│                                                        │
│ ┌────────────────────────────────┐  ┌──────────────┐ │
│ │ Momentum_Tech_LongOnly          │  │ 💾 Save Book │ │
│ └────────────────────────────────┘  └──────────────┘ │
│                                                        │
│ ✅ Book 'Momentum_Tech_LongOnly' saved successfully!   │
│ 📂 Saved to: saved_books/Momentum_Tech_LongOnly.json  │
│ 💡 Load this book in the Strategy tab                 │
└────────────────────────────────────────────────────────┘
```

**That's it!** Just enter a name and click save.

### Advanced Save (Optional)

Want to add more details? Expand **"Advanced Options"**:

```
⚙️ Advanced Options (Description & Tags)  [▼ Expanded]
┌────────────────────────────────────────────────────────┐
│ Description (optional):                                │
│ ┌────────────────────────────────────────────────────┐ │
│ │ Fast momentum strategy for tech stocks              │ │
│ │ Optimized for bull markets                          │ │
│ └────────────────────────────────────────────────────┘ │
│                                                        │
│ Tags (optional, comma-separated):                     │
│ ┌────────────────────────────────────────────────────┐ │
│ │ momentum, long-only, tech, bull-market              │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Loading a Book (Strategy Tab)

### Step 1: Select Load Mode

```
┌────────────────────────────────────────────────────────┐
│ Selection Mode                                         │
│ ○ Manual Selection    ● Load from Saved Book          │
└────────────────────────────────────────────────────────┘
```

### Step 2: Choose Your Book

```
┌────────────────────────────────────────────────────────┐
│ 📚 Select Book                                         │
│ ┌────────────────────────────────────────────────────┐ │
│ │ ▼ Momentum_Tech_LongOnly                            │ │
│ └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### Step 3: Review Book Info

```
┌───────────────────────────┬───────────────────────────┐
│ Book Information:         │                           │
├───────────────────────────┼───────────────────────────┤
│ Name: Momentum_Tech       │ Description: Fast         │
│ Strategy: MOMENTUM.ma_... │   momentum for tech       │
│ Symbols: 4 symbols        │ Tags: momentum, tech      │
│                           │ Created: 2025-10-27       │
└───────────────────────────┴───────────────────────────┘
```

### Step 4: Edit Symbols (Optional)

```
📋 Book Symbols (Editable)  [▼ Expanded]
┌────────────────────────────────────────────────────────┐
│ Edit the symbol list below. Changes will be saved.    │
│                                                        │
│ ┌────────────────────────────────────────────────────┐ │
│ │ AAPL, MSFT, GOOGL, NVDA, TSLA                       │ │
│ │                                                      │ │
│ └────────────────────────────────────────────────────┘ │
│                                                        │
│ ┌──────────────────┬──────────────────┐               │
│ │ Original:        │ New:             │               │
│ │ 4 symbols        │ 5 symbols        │               │
│ │ AAPL, MSFT,      │ AAPL, MSFT,      │               │
│ │ GOOGL, NVDA      │ GOOGL, NVDA, TSLA│               │
│ └──────────────────┴──────────────────┘               │
│                                                        │
│ [💾 Update Book Symbols]                               │
└────────────────────────────────────────────────────────┘
```

### Step 5: Review Parameters

Parameters are pre-filled but editable:

```
⚙️ Parameters (Loaded from Book - Editable)
┌────────────┬────────────┬────────────┬────────────┐
│ Fast Ma    │ Slow Ma    │ Min Periods│            │
│ 20         │ 50         │ 60         │            │
└────────────┴────────────┴────────────┴────────────┘
```

### Step 6: Run Backtest

```
┌────────────────────────────────────────────────────────┐
│                  [🚀 Run Backtest]                      │
└────────────────────────────────────────────────────────┘

✅ Loaded book 'Momentum_Tech_LongOnly' with 5 symbols!
⚠️ Note: Dates are still from your Configuration tab.
```

---

## 🔄 Common Workflows

### Workflow 1: Save and Reuse

```
1. Configure dates: 2020-01-01 to 2021-01-01
2. Select strategy: ma_crossover_long_only
3. Set parameters: fast=20, slow=50
4. Add symbols: AAPL, MSFT, GOOGL, NVDA
5. Run backtest
6. Save as book: "Momentum_Tech_2020"

Later...

7. Change dates: 2021-01-01 to 2022-01-01
8. Load book: "Momentum_Tech_2020"
9. Run backtest (same strategy, new dates)
10. Compare results!
```

### Workflow 2: Test Different Symbol Sets

```
1. Load book: "Momentum_Strategy"
2. Edit symbols: AAPL, MSFT, GOOGL
3. Run backtest → Note Sharpe ratio
4. Edit symbols: NVDA, TSLA, AMD
5. Run backtest → Compare Sharpe ratio
6. Keep the better performing symbol set
```

### Workflow 3: Parameter Sweep

```
1. Load book: "MA_Crossover_Base"
2. Params show: fast=20, slow=50
3. Run backtest → Save results
4. Edit params: fast=10, slow=30
5. Run backtest → Compare
6. Edit params: fast=30, slow=100
7. Run backtest → Compare all
8. Save best as new book: "MA_Crossover_Optimized"
```

---

## 📊 Visual Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    BOOKS FEATURE                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  RESULTS TAB                        STRATEGY TAB             │
│  ┌────────────────────┐            ┌────────────────────┐   │
│  │ After backtest:    │            │ Before backtest:   │   │
│  │                    │            │                    │   │
│  │ Enter name         │────────────│ Select book        │   │
│  │ Click save         │   REUSE    │ Review settings    │   │
│  │                    │            │ Run backtest       │   │
│  └────────────────────┘            └────────────────────┘   │
│                                                              │
│  SAVED BOOKS (JSON)                                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Strategy + Parameters + Symbols                    │   │
│  │ • Stored in saved_books/ directory                   │   │
│  │ • Editable via UI or JSON                            │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Concepts

### What's Saved
✅ **Strategy name** (e.g., ma_crossover_long_only)
✅ **All parameters** (e.g., fast_ma=20, slow_ma=50)
✅ **Symbol list** (e.g., AAPL, MSFT, GOOGL, NVDA)
✅ **Description & tags** (optional metadata)

### What's NOT Saved
❌ **Date ranges** (use current config dates)
❌ **Execution costs** (use current config costs)
❌ **Initial capital** (use current config capital)

### Why?
This lets you test the **same strategy** on **different time periods** without re-entering parameters!

---

## 💡 Tips

1. **Name clearly**: Use descriptive names like `Momentum_Tech_Fast` instead of `Strategy1`

2. **Save often**: After finding good parameters, save immediately

3. **Use tags**: Organize books by strategy type, market regime, or sector

4. **Version books**: `Momentum_v1`, `Momentum_v2_optimized`

5. **Edit symbols**: Use the editable symbol section to test variations

6. **Compare dates**: Same book, different dates = strategy robustness test

---

## 🚀 Next Steps

Now that you understand books:

1. **Run a backtest** (Configuration → Strategy → Run)
2. **Save it** (Results → Save as Book)
3. **Reload it** (Strategy → Load from Saved Book)
4. **Test different dates** (Configuration → Change dates → Run)

**Goal**: Build a library of tested, validated strategy configurations!

---

## 📁 File Locations

Books are stored here:
```
backtester2/
└── saved_books/
    ├── Momentum_Tech_LongOnly.json
    ├── MA_Crossover_Fast.json
    └── Mean_Reversion_Utils.json
```

Each book is a separate JSON file you can:
- View in a text editor
- Track in git
- Share with others
- Edit manually (if needed)

---

## ❓ Questions?

- **Can I edit a saved book?** Yes! Load it, change symbols/params, update or save as new
- **Can I delete books?** Yes, just delete the JSON file in `saved_books/`
- **Can I rename books?** Yes, rename the JSON file (keep the `.json` extension)
- **How many books can I save?** Unlimited!
- **Can I share books?** Yes! Just share the JSON file

---

**Happy backtesting!** 📈📚
