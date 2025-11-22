# Streamlit Applications

This folder contains all Streamlit web interfaces for BackT.

## Main Application

**`streamlit_backtest_runner.py`** - Master application with 5 tabs:

1. **⚙️ Configuration** - Set backtest parameters, dates, capital, universe
2. **📈 Strategy** - Select and configure trading strategies
3. **📊 Results** - View metrics, charts, and performance analysis
4. **🔬 CPCV Validation** - Combinatorial Purged Cross-Validation for overfitting detection
5. **📚 Book Manager** - Edit saved strategy configurations (symbols, parameters, metadata)

**Launch:**
```bash
# From project root:
launch_streamlit.bat

# Or manually:
cd streamlit_apps
streamlit run streamlit_backtest_runner.py
```

---

## Additional Applications

### `streamlit_book_manager.py`
Standalone book manager (now integrated into main app as Tab 5)

### `streamlit_backtest_runner_cpcv.py`
Legacy CPCV validation interface (functionality moved to main app)

### `streamlit_app.py`
Legacy simplified interface (replaced by streamlit_backtest_runner.py)

---

## Quick Start

1. **Activate environment:**
   ```bash
   conda activate backt
   ```

2. **Launch application:**
   ```bash
   # Windows:
   ..\launch_streamlit.bat

   # Or from this directory:
   streamlit run streamlit_backtest_runner.py
   ```

3. **Access in browser:**
   ```
   http://localhost:8501
   ```

---

## Features by Tab

### Tab 1: Configuration
- Date range selection
- Initial capital
- Symbol universe selection
- Execution settings (commissions, slippage)
- Market data validation

### Tab 2: Strategy
- Strategy selection from multiple modules:
  - Momentum strategies (9 strategies)
  - AQR Factor strategies (6 strategies)
  - Benchmark strategies (3 strategies)
- Parameter configuration
- Load from saved books
- Save current configuration as book

### Tab 3: Results
- Performance metrics (returns, Sharpe, Sortino, max DD)
- Equity curve with drawdown
- Trade log with detailed execution info
- Monthly returns heatmap
- Strategy state visualization
- Export results to CSV/JSON

### Tab 4: CPCV Validation
- Combinatorial Purged Cross-Validation
- Probability of Backtest Overfitting (PBO)
- Deflated Sharpe Ratio (DSR)
- In-sample vs Out-of-sample comparison
- Statistical significance testing

### Tab 5: Book Manager (NEW!)
- Load and edit saved books
- Symbol editing with visual diff
- Parameter editing with type-aware inputs
- Metadata editing (description, tags)
- Validation checks
- Automatic backups
- Delete books
- Filter by strategy/tags

---

## Workflow Example

**Step 1: Configure (Tab 1)**
- Set dates: 2020-01-01 to 2023-12-31
- Set capital: $100,000
- Select symbols: SPY, QQQ, TLT

**Step 2: Select Strategy (Tab 2)**
- Choose: MACD Crossover
- Set parameters: fast=12, slow=26, signal=9
- Or load from book: "MACD_Top5_2024"

**Step 3: View Results (Tab 3)**
- Total Return: 45.2%
- Sharpe Ratio: 1.32
- Max Drawdown: -12.3%
- Review trades and equity curve

**Step 4: Validate (Tab 4)**
- Run CPCV with 10 splits
- Check PBO < 0.5 (not overfit)
- Check DSR > 0 (statistically significant)

**Step 5: Edit Book (Tab 5)**
- Load "MACD_Top5_2024"
- Remove underperforming symbol
- Add new candidate
- Update parameters based on optimization
- Save changes (auto-backup created)

---

## Tips

- Use **Book Manager (Tab 5)** to maintain strategy configurations
- Run **CPCV (Tab 4)** before deploying any strategy
- Export results from **Tab 3** for external analysis
- Save successful configurations as books for reuse

---

## Keyboard Shortcuts

- **Ctrl+R** - Refresh page
- **F11** - Fullscreen mode
- **Esc** - Close modals/popups

---

## Troubleshooting

**Issue: Streamlit won't start**
```bash
pip install streamlit
```

**Issue: Module not found**
```bash
# From project root:
pip install -e .
```

**Issue: Data not loading**
- Check database exists: `market_data.db`
- Run update: `python update_market_data.py`

**Issue: Book Manager shows no books**
- Create books using ranking tool:
  ```bash
  python rank_symbols_by_strategy.py --save "BookName"
  ```

---

## File Organization

```
streamlit_apps/
├── streamlit_backtest_runner.py  ← Main application (use this!)
├── streamlit_book_manager.py     ← Standalone book manager
├── streamlit_backtest_runner_cpcv.py  ← Legacy CPCV
├── streamlit_app.py              ← Legacy simple interface
└── README.md                     ← This file
```

---

## Integration with CLI Tools

The Streamlit interface works seamlessly with CLI tools:

1. **Ranking Tool** → Creates books
   ```bash
   python rank_symbols_by_strategy.py --save "MyBook"
   ```

2. **Book Editor** → Edit via CLI or Streamlit
   ```bash
   python edit_book.py "MyBook" --add-symbols SPY
   # OR use Streamlit Tab 5
   ```

3. **Backtest** → Load book and run
   - Use Tab 2 → "Load from Saved Book"
   - Select book
   - Run backtest

---

## Performance

- First load: ~2-3 seconds (data loading)
- Subsequent loads: <1 second (cached)
- Backtest execution: Depends on symbols/date range
  - Single symbol, 1 year: <1 second
  - 10 symbols, 5 years: ~5 seconds
  - 50 symbols, 10 years: ~30 seconds

---

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Edge
- ✅ Firefox
- ⚠️ Safari (some features may vary)

---

## Updates

The Streamlit interface is actively maintained. New features:
- **Nov 2024**: Book Manager integrated as Tab 5
- **Nov 2024**: CPCV Validation moved to Tab 4
- **Oct 2024**: Multi-tab interface introduced

---

For more information, see the main project documentation in the parent directory.
