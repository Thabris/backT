# Books Feature Documentation

## Overview

The **Books** feature allows you to save and reuse strategy configurations across different backtests. A "book" is a modular, reusable configuration that captures:

- Strategy selection (which strategy to use)
- Strategy parameters (all parameter values)
- Symbol universe (which symbols to trade)
- Metadata (description, tags, timestamps)

Books are designed with a modular architecture to enable **portfolio-level backtesting** in the future, where you can run multiple books together on different symbol clusters.

## Key Concepts

### What is a Book?

A book is a saved configuration that contains:
- ✅ **Strategy**: The strategy module and function name
- ✅ **Parameters**: All strategy parameter values
- ✅ **Symbols**: The list of symbols to trade
- ✅ **Metadata**: Description, tags, creation/update timestamps
- ❌ **NOT included**: Date ranges (these come from your Configuration tab)

This design allows you to test the same strategy configuration across different time periods.

### Storage

Books are stored as JSON files in the `saved_books/` directory. Each book is a separate file, making them:
- Easy to version control with git
- Human-readable and editable
- Portable across systems

## How to Use Books

### 1. Creating and Saving a Book

After running a backtest:

1. Go to the **Results** tab
2. Scroll to the bottom to the **"Save as Book"** section
3. You'll see what will be saved: strategy name, parameters count, symbols count
4. Enter a **Book Name** (required) - e.g., "Momentum_Tech_LongOnly"
5. (Optional) Expand **"Advanced Options"** to add:
   - **Description** - e.g., "Fast momentum for tech stocks"
   - **Tags** - e.g., "momentum, long-only, tech"
6. Click **"💾 Save Book"**
7. The book will be saved to `saved_books/{name}.json`

**Quick Save Example:**
```
Will save: ma_crossover_long_only • 3 params • 4 symbols
Book Name: [Momentum_Tech_LongOnly]
          [💾 Save Book]
✅ Book 'Momentum_Tech_LongOnly' saved successfully!
```

**With Advanced Options:**
```
Book Name: Fast_MA_Crossover_Tech
Description: Fast momentum strategy optimized for tech stocks during bull markets
Tags: momentum, tech, long-only, bull-market
```

### 2. Loading a Book

To use a saved book:

1. Go to the **Strategy** tab
2. Select **"Load from Saved Book"** in the selection mode radio button
3. Choose a book from the dropdown
4. The book will load:
   - ✅ Strategy and parameters (pre-filled but editable)
   - ✅ Symbols (overrides Configuration tab symbols)
   - ⚠️ Dates remain from your Configuration tab
5. Review/adjust parameters if needed
6. Click **"Run Backtest"**

### 3. Editing Book Symbols

You can modify the symbol list of a saved book:

1. Load a book in the **Strategy** tab
2. Expand the **"Book Symbols (Editable)"** section
3. Edit the symbols in the text area:
   - Add new symbols (comma-separated)
   - Remove unwanted symbols
   - Symbols are automatically uppercased
4. A comparison view shows original vs. new symbol list
5. Click **"Update Book Symbols"** (only enabled when changes detected)
6. The book will be updated with the new symbol list
7. The page will refresh with the updated book

**Example:**
```
Original: AAPL, MSFT, GOOGL
Edit to: AAPL, MSFT, GOOGL, NVDA, TSLA
Result: Book updated with 5 symbols instead of 3
```

### 4. Editing Book Parameters

When you load a book, all parameters are **pre-filled** but **still editable**. This allows you to:
- Test variations of a saved strategy
- Fine-tune parameters for different market conditions
- Compare similar configurations

## Use Cases

### 1. **Regime-Based Trading**
Save different books optimized for different market regimes:
- `Momentum_Bull_Market` - Aggressive momentum for bull markets
- `Mean_Reversion_Sideways` - Range-bound strategy for sideways markets
- `Defensive_Bear_Market` - Capital preservation for bear markets

### 2. **Sector Rotation**
Create books for different sectors:
- `Tech_Growth` - Growth stocks (AAPL, MSFT, GOOGL, NVDA)
- `Financials_Value` - Value stocks (JPM, BAC, WFC, GS)
- `Commodities` - Commodity ETFs (GLD, USO, DBC)

### 3. **Multi-Timeframe Strategies**
Save the same strategy with different parameter sets:
- `MA_Fast_20_50` - Fast moving average crossover
- `MA_Medium_50_100` - Medium-term trend following
- `MA_Slow_100_200` - Long-term position trading

### 4. **Strategy Testing Across Periods**
Load the same book and test across different date ranges:
- 2008-2009 (Financial Crisis)
- 2020 (COVID Crash)
- 2021-2022 (Bull to Bear transition)

## Future: Portfolio-Level Backtesting

The modular design enables future features:

### Multi-Book Portfolios
Run multiple books simultaneously with:
- Different strategies per book
- Different symbol clusters per book
- Shared capital allocation
- Portfolio-level risk management

### Example Portfolio:
```
Portfolio "All Weather" = {
    Book 1: Momentum on Tech stocks (30% capital)
    Book 2: Mean Reversion on Utilities (20% capital)
    Book 3: Trend Following on Commodities (20% capital)
    Book 4: Buy & Hold on Bonds (30% capital)
}
```

This allows you to:
- Optimize Sharpe ratio across books
- Balance risk/return at portfolio level
- Test different book combinations
- Implement regime-switching at portfolio level

## API Reference

### Book Class

```python
@dataclass
class Book:
    name: str                       # User-friendly name
    strategy_module: str            # e.g., "MOMENTUM", "AQR"
    strategy_name: str              # e.g., "ma_crossover_long_only"
    strategy_params: Dict[str, Any] # Strategy parameters
    symbols: List[str]              # Universe of symbols
    description: str = ""           # Optional description
    created_at: str                 # ISO timestamp
    updated_at: str                 # ISO timestamp
    tags: List[str]                 # For categorization
    metadata: Dict[str, Any]        # Extensible metadata
```

### BookManager Class

```python
manager = BookManager()                    # Initialize (default: ./saved_books/)
manager = BookManager("/path/to/books")    # Custom directory

# Save/Load
manager.save_book(book, overwrite=False)   # Save book
book = manager.load_book("book_name")      # Load book
manager.delete_book("book_name")           # Delete book

# Query
books = manager.list_books()               # Get all book names (sorted)
books = manager.get_all_books()            # Load all books
books = manager.get_books_by_tag("momentum")            # Filter by tag
books = manager.get_books_by_strategy("ma_crossover")   # Filter by strategy

# Utilities
manager.book_exists("book_name")           # Check if book exists
manager.rename_book("old", "new")          # Rename book
```

### Creating Books Programmatically

```python
from backt.utils.books import create_book_from_session, BookManager

# Create a book
book = create_book_from_session(
    name="My_Strategy",
    strategy_module="MOMENTUM",
    strategy_name="ma_crossover_long_only",
    strategy_params={'fast_ma': 20, 'slow_ma': 50},
    symbols=["AAPL", "MSFT", "GOOGL"],
    description="Tech momentum strategy",
    tags=["momentum", "tech"]
)

# Save it
manager = BookManager()
manager.save_book(book)

# Load it later
loaded_book = manager.load_book("My_Strategy")
```

## File Structure

```
backtester2/
├── saved_books/                    # Books directory (auto-created)
│   ├── Fast_MA_Tech.json          # Book 1
│   ├── Slow_MA_Value.json         # Book 2
│   └── Mean_Reversion_Utils.json  # Book 3
├── backt/
│   └── utils/
│       └── books.py               # Book classes and manager
└── streamlit_backtest_runner.py   # Main app with Books UI
```

## Example Book JSON

```json
{
  "name": "Momentum_Tech_LongOnly",
  "strategy_module": "MOMENTUM",
  "strategy_name": "ma_crossover_long_only",
  "strategy_params": {
    "fast_ma": 20,
    "slow_ma": 50,
    "min_periods": 60
  },
  "symbols": [
    "AAPL",
    "MSFT",
    "GOOGL",
    "NVDA"
  ],
  "description": "Fast momentum strategy for tech stocks",
  "created_at": "2025-10-27T10:30:00",
  "updated_at": "2025-10-27T10:30:00",
  "tags": [
    "momentum",
    "long-only",
    "tech"
  ],
  "metadata": {}
}
```

## Tips and Best Practices

### Naming Conventions
Use descriptive, structured names:
- ✅ `Momentum_Tech_20_50` - Describes strategy, sector, and params
- ✅ `MR_Utilities_LongShort` - Clear and specific
- ❌ `Strategy1` - Too vague
- ❌ `test` - Not descriptive

### Tags
Use tags for organization:
- **Strategy type**: `momentum`, `mean-reversion`, `trend-following`
- **Direction**: `long-only`, `long-short`
- **Asset class**: `equities`, `commodities`, `fx`
- **Sector**: `tech`, `financials`, `utilities`
- **Market regime**: `bull-market`, `bear-market`, `sideways`

### Version Control
Since books are JSON files, you can:
- Track them in git
- Create branches for different configurations
- Share books with your team
- Review changes with git diff

### Testing Workflow
1. **Develop** - Create and tune strategy manually
2. **Save** - Save as book when satisfied
3. **Validate** - Test book across different time periods
4. **Iterate** - Load book, adjust parameters or symbols, update book
5. **Deploy** - Use validated books in production

### Symbol Management
- **Quick edits** - Use the editable symbol section to add/remove symbols on the fly
- **Version control** - Consider saving a new book when making major symbol changes
- **Bulk updates** - Edit multiple symbols at once in the text area
- **Validation** - Symbols are automatically uppercased and trimmed

## Troubleshooting

### Book Not Appearing in Dropdown
- Check that the book file exists in `saved_books/`
- Verify the JSON is valid (no syntax errors)
- Refresh the Streamlit app

### Strategy Not Found When Loading Book
- Ensure the strategy still exists in the `strategies/` folder
- Check that the strategy name matches exactly
- Verify the strategy module is correct

### Parameters Not Loading Correctly
- Check that parameter names match the strategy function
- Verify parameter types in the JSON (int, float, bool)
- Ensure parameter values are within valid ranges

### Symbols Override Not Working
- Make sure you're loading from the book (not manual selection)
- Check that the success message confirms symbol override
- Verify the symbols list in the book JSON is correct

### Update Book Symbols Button Disabled
- Button is only enabled when you've made changes to the symbol list
- Check that your edited symbols are different from the original
- Ensure at least one valid symbol is entered (no empty list)
- After updating, the page will refresh automatically

## Testing

Run the test suite to verify the books feature:

```bash
python test_books.py
```

This tests:
- Book creation and saving
- Loading and deserialization
- Querying (by tag, by strategy)
- File operations (exists, rename, delete)

## Summary

The Books feature provides:
- ✅ **Quick save** - Name + button at bottom of Results tab
- ✅ **Reusable configurations** - Save once, use many times
- ✅ **Editable** - Modify symbols and parameters when loading
- ✅ **Modular design** - Easy to organize and combine
- ✅ **Version control friendly** - JSON files in git
- ✅ **Future-ready** - Built for portfolio-level backtesting

**Simple Workflow:**
1. Run backtest → 2. Name it → 3. Save → 4. Load anytime in Strategy tab

This sets the foundation for advanced portfolio construction where multiple books can run together, enabling regime-based strategies and multi-strategy portfolios with optimized capital allocation.
