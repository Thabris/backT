# Book Editor Guide

The Book Editor provides a comprehensive system for managing saved strategy configurations (Books). You can edit symbols, parameters, metadata, and track all changes with automatic backups.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Usage](#cli-usage)
3. [Programmatic Usage](#programmatic-usage)
4. [Features](#features)
5. [Examples](#examples)

---

## Quick Start

### List All Books

```bash
python edit_book.py --list
```

Output:
```
================================================================================================
SAVED BOOKS
================================================================================================
Name                           Strategy                  Symbols    Updated
------------------------------------------------------------------------------------------------
MACD_Top5_2024                 MOMENTUM.macd_crossover   5          2024-11-09
Bollinger_Top10_Long           MOMENTUM.bollinger_mean   10         2024-11-09
================================================================================================
Total books: 2
```

### Interactive Editing

```bash
python edit_book.py "MACD_Top5_2024"
```

This opens an interactive menu where you can:
- View the book
- Edit symbols
- Edit parameters
- Update metadata
- Preview changes
- Save or discard

---

## CLI Usage

### Quick Edit Symbols

**Add symbols:**
```bash
python edit_book.py "MACD_Top5_2024" --add-symbols SPY,QQQ,TLT
```

**Remove symbols:**
```bash
python edit_book.py "MACD_Top5_2024" --remove-symbols XLU,VPU
```

**Replace all symbols:**
```bash
python edit_book.py "MACD_Top5_2024" --replace-symbols SPY,QQQ,TLT,IEF,GLD
```

### Quick Edit Parameters

```bash
python edit_book.py "MACD_Top5_2024" --params fast_period=12 slow_period=26 signal_period=9
```

**Parameter types are auto-detected:**
- Integers: `fast_period=12`
- Floats: `bb_std=2.5`
- Booleans: `allow_short=true`
- Strings: `name=MyStrategy`

### Skip Preview

```bash
python edit_book.py "MACD_Top5_2024" --add-symbols SPY --no-preview
```

---

## Programmatic Usage

### Using Quick Functions

```python
from backt.utils import quick_edit_symbols, quick_edit_params

# Add symbols
quick_edit_symbols("MACD_Top5_2024", add=["SPY", "QQQ"])

# Remove symbols
quick_edit_symbols("MACD_Top5_2024", remove=["XLU"])

# Replace all symbols
quick_edit_symbols("MACD_Top5_2024", replace=["SPY", "QQQ", "TLT"])

# Edit parameters
quick_edit_params("MACD_Top5_2024", {
    "fast_period": 12,
    "slow_period": 26,
    "signal_period": 9
})
```

### Using BookEditor Class

```python
from backt.utils import BookEditor

editor = BookEditor()

# Load a book
book = editor.load_book("MACD_Top5_2024")

# Edit symbols
book = editor.add_symbols(book, ["SPY", "QQQ"])
book = editor.remove_symbols(book, ["XLU"])

# Edit parameters
book = editor.update_parameters(book, {
    "fast_period": 12,
    "slow_period": 26
})

# Edit metadata
book = editor.set_description(book, "Top 5 MACD performers for 2024")
book = editor.add_tags(book, ["momentum", "trend-following"])

# Preview changes
print(editor.preview_changes(book))

# Validate
is_valid, warnings = editor.validate_book(book)
if not is_valid:
    for warning in warnings:
        print(f"Warning: {warning}")

# Save (creates automatic backup)
editor.save_book(book)
```

---

## Features

### 1. Symbol Management

**Add symbols:**
- Adds symbols to the existing list
- Prevents duplicates
- Auto-sorts alphabetically

**Remove symbols:**
- Removes specified symbols
- Safe if symbol doesn't exist

**Replace symbols:**
- Completely replaces the symbol list
- Removes duplicates automatically
- Sorts alphabetically

### 2. Parameter Management

**Update parameters:**
- Add new parameters
- Modify existing parameters
- Shows old → new values

**Remove parameters:**
- Delete parameters from the book
- Shows old value before deletion

### 3. Metadata Management

**Description:**
- Set or update book description

**Tags:**
- Add tags for categorization
- Remove tags
- Used for filtering books

**Custom metadata:**
- Update any metadata field
- Extensible for custom use cases

### 4. Change Tracking

**Preview:**
- View book contents before saving
- See all symbols and parameters
- Check metadata and tags

**Diff:**
- Compare original vs modified
- Shows added/removed symbols
- Shows parameter changes
- Shows metadata changes

**Example diff:**
```
================================================================================
CHANGES
================================================================================

Symbols:
  + Added (2):   QQQ, SPY
  - Removed (1): XLU

Parameters:
  fast_period: 11 → 12
  slow_period: 20 → 26

Description:
  Old: (none)
  New: Top 5 MACD performers for 2024
================================================================================
```

### 5. Validation

Checks for:
- Empty symbol lists
- Duplicate symbols
- Empty strategy names
- Missing parameters
- Excessively large symbol lists (>50)

**Example:**
```python
is_valid, warnings = editor.validate_book(book)
# warnings: ["Duplicate symbols found: SPY", "Large number of symbols (75)"]
```

### 6. Automatic Backups

**Every save creates a timestamped backup:**
```
saved_books/_backups/
  MACD_Top5_2024_20241109_153022.json
  MACD_Top5_2024_20241109_154511.json
  Bollinger_Top10_20241109_160033.json
```

**List backups:**
```python
backups = editor.list_backups("MACD_Top5_2024")
for backup in backups:
    print(backup.name)
```

**Restore from backup:**
```python
backup_path = editor.list_backups("MACD_Top5_2024")[0]  # Most recent
editor.restore_backup(backup_path)
```

---

## Examples

### Example 1: Update Strategy Parameters After Optimization

You ran parameter optimization and found better values:

```python
from backt.utils import quick_edit_params

# Update with optimized parameters
quick_edit_params("MACD_Top5_2024", {
    "fast_period": 11,
    "slow_period": 22,
    "signal_period": 8
})
```

### Example 2: Swap Out Underperforming Symbols

After reviewing performance, you want to replace underperformers:

```python
from backt.utils import BookEditor

editor = BookEditor()
book = editor.load_book("MACD_Top5_2024")

# Remove underperformers
book = editor.remove_symbols(book, ["XLU", "VPU"])

# Add new candidates
book = editor.add_symbols(book, ["SPY", "QQQ"])

# Preview and save
print(editor.preview_changes(book))
editor.save_book(book)
```

### Example 3: Bulk Update Multiple Books

Update parameters across multiple books:

```python
from backt.utils import BookEditor

editor = BookEditor()

# Get all MACD books
books = editor.manager.get_books_by_strategy("macd_crossover")

# Update all with new parameters
for book in books:
    editor.update_parameters(book, {"signal_period": 9})
    editor.save_book(book)
    print(f"Updated {book.name}")
```

### Example 4: Create a Filtered Version of a Book

Create a new book with only high-quality symbols:

```python
from backt.utils import BookEditor, create_book_from_session

editor = BookEditor()
original = editor.load_book("ETF_Universe_All")

# Filter to only equity ETFs
equity_symbols = [s for s in original.symbols
                  if s in ['SPY', 'QQQ', 'IWM', 'VGK', 'EEM']]

# Create new book
new_book = create_book_from_session(
    name="ETF_Universe_Equities",
    strategy_module=original.strategy_module,
    strategy_name=original.strategy_name,
    strategy_params=original.strategy_params,
    symbols=equity_symbols,
    description="Equity ETFs only",
    tags=["equities", "filtered"]
)

editor.save_book(new_book)
```

### Example 5: Interactive Editing with Validation

```python
from backt.utils import BookEditor

editor = BookEditor()
book = editor.load_book("MACD_Top5_2024")

# Make changes
book = editor.add_symbols(book, ["INVALID_SYMBOL", "SPY", "SPY"])  # Duplicates

# Validate before saving
is_valid, warnings = editor.validate_book(book)
if warnings:
    print("Warnings found:")
    for warning in warnings:
        print(f"  - {warning}")

    proceed = input("Save anyway? (y/n): ")
    if proceed.lower() == 'y':
        editor.save_book(book)
else:
    editor.save_book(book)
```

### Example 6: Compare Two Books

```python
from backt.utils import BookEditor

editor = BookEditor()

book1 = editor.load_book("MACD_Top5_2024")
book2 = editor.load_book("MACD_Top10_2024")

# Show differences
print(editor.show_diff(book1, book2))
```

---

## Advanced Usage

### Custom Metadata Tracking

Track performance metrics or other custom data:

```python
book = editor.update_metadata(book, {
    "avg_sortino": 1.25,
    "avg_sharpe": 0.98,
    "backtest_date": "2024-11-09",
    "notes": "Optimized on 2024 data"
})
```

### Conditional Parameter Updates

Update parameters only if they meet certain criteria:

```python
book = editor.load_book("MACD_Top5_2024")

# Only update if fast_period is below 12
if book.strategy_params.get("fast_period", 0) < 12:
    book = editor.update_parameters(book, {"fast_period": 12})
    editor.save_book(book)
```

### Batch Symbol Updates

Update symbols based on external data:

```python
import pandas as pd

# Load top performers from CSV
top_performers = pd.read_csv("top_performers.csv")
symbols = top_performers["symbol"].tolist()

# Update book
editor = BookEditor()
book = editor.load_book("Dynamic_Top20")
book = editor.replace_symbols(book, symbols)
editor.save_book(book)
```

---

## Best Practices

1. **Always preview changes** before saving (enabled by default)
2. **Use validation** before saving critical books
3. **Use descriptive tags** for easy filtering (e.g., "momentum", "long-only", "optimized")
4. **Keep backups** - they're created automatically but verify they exist
5. **Document changes** in the description field
6. **Use meaningful book names** - include strategy and date/period
7. **Test edited books** before using in production backtests

---

## Troubleshooting

**Book not found:**
```
Error: Book 'MyBook' not found
```
Solution: Use `python edit_book.py --list` to see available books.

**Duplicate symbols:**
```
Warning: Duplicate symbols found: SPY
```
Solution: The editor will auto-remove duplicates when you save.

**Invalid parameters:**
```
Error parsing parameters: invalid literal for int()
```
Solution: Check parameter format. Use `key=value` with no spaces around `=`.

**Backup restore:**
If you made a mistake, restore from backup:
```python
editor = BookEditor()
backups = editor.list_backups("MACD_Top5_2024")
editor.restore_backup(backups[0])  # Most recent backup
```

---

## API Reference

### BookEditor Methods

| Method | Description |
|--------|-------------|
| `load_book(name)` | Load a book for editing |
| `save_book(book, create_backup=True)` | Save with automatic backup |
| `add_symbols(book, symbols)` | Add symbols to book |
| `remove_symbols(book, symbols)` | Remove symbols from book |
| `replace_symbols(book, symbols)` | Replace all symbols |
| `update_parameters(book, params)` | Update strategy parameters |
| `remove_parameter(book, name)` | Remove a parameter |
| `update_metadata(book, metadata)` | Update metadata fields |
| `add_tags(book, tags)` | Add tags |
| `remove_tags(book, tags)` | Remove tags |
| `set_description(book, desc)` | Set description |
| `preview_changes(book)` | Show book preview |
| `show_diff(old, new)` | Show differences |
| `validate_book(book)` | Validate book |
| `list_backups(name)` | List backups for a book |
| `restore_backup(path)` | Restore from backup |

### Quick Functions

| Function | Description |
|----------|-------------|
| `quick_edit_symbols(name, add=[], remove=[], replace=[])` | Quick symbol editing |
| `quick_edit_params(name, params={})` | Quick parameter editing |

---

## Integration with Streamlit

The Book Editor works seamlessly with the Streamlit interface. After editing a book:

1. Edit the book using CLI or programmatically
2. Launch Streamlit: `streamlit run streamlit_app.py`
3. Select "Load from Book" in sidebar
4. Choose your edited book
5. Run backtest with updated configuration

---

## Summary

The Book Editor provides a complete solution for managing your strategy configurations:

✅ **Easy editing** - CLI and programmatic interfaces
✅ **Change tracking** - Preview and diff before saving
✅ **Safety** - Automatic backups and validation
✅ **Flexibility** - Edit symbols, parameters, and metadata
✅ **Integration** - Works with Streamlit and backtesting workflow

Start editing your books today:
```bash
python edit_book.py --list
python edit_book.py "YourBookName"
```
