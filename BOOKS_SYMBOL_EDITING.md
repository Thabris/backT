# Book Symbol Editing - Quick Reference

## Feature Overview

The editable symbol section allows you to modify the symbol list of a saved book directly in the Streamlit UI, without manually editing JSON files.

## Location

**Strategy Tab** → **"Load from Saved Book"** → Select a book → Expand **"Book Symbols (Editable)"**

## How It Works

### 1. Load a Book
- Select "Load from Saved Book" mode
- Choose a book from the dropdown
- Expand "Book Symbols (Editable)" section

### 2. Edit Symbols
The text area shows the current symbols in comma-separated format:
```
AAPL, MSFT, GOOGL
```

You can:
- **Add symbols**: `AAPL, MSFT, GOOGL, NVDA, TSLA`
- **Remove symbols**: `AAPL, MSFT` (removed GOOGL)
- **Replace symbols**: `SPY, QQQ, IWM` (completely different list)

### 3. Review Changes
When you edit the symbol list, a comparison view appears:

```
Original:                    New:
3 symbols                    5 symbols
AAPL, MSFT, GOOGL           AAPL, MSFT, GOOGL, NVDA, TSLA
```

### 4. Update Book
- Click **"Update Book Symbols"** (only enabled when changes detected)
- Book is saved with new symbol list
- Page automatically refreshes
- Success message confirms update

## Features

### Smart Parsing
- **Auto-uppercase**: `aapl` → `AAPL`
- **Auto-trim**: `AAPL , MSFT` → `AAPL, MSFT`
- **Empty removal**: Blank entries are ignored
- **Flexible separators**: Works with commas and spaces

### Button States
- **Disabled** (gray) - No changes detected
- **Enabled** (primary blue) - Changes detected, ready to save

### Change Detection
The system detects changes by comparing symbol sets:
- Order doesn't matter: `AAPL, MSFT` = `MSFT, AAPL`
- Case-insensitive: `aapl` = `AAPL`
- Duplicate removal: `AAPL, AAPL` → `AAPL`

## Use Cases

### 1. Quick Symbol Addition
You discover a new symbol that fits your book's strategy:
```
Before: AAPL, MSFT, GOOGL
Add:    NVDA
After:  AAPL, MSFT, GOOGL, NVDA
```

### 2. Sector Rotation
Switch from tech to financials:
```
Before: AAPL, MSFT, GOOGL (Tech)
After:  JPM, BAC, WFC, GS (Financials)
```

### 3. Universe Expansion
Expand from individual stocks to include ETFs:
```
Before: AAPL, MSFT, GOOGL
After:  AAPL, MSFT, GOOGL, QQQ, VGT, XLK
```

### 4. A/B Testing
Test the same strategy on different symbol sets:
1. Load book "Momentum_Strategy"
2. Edit symbols to "AAPL, MSFT, GOOGL"
3. Run backtest → Save results
4. Edit symbols to "NVDA, TSLA, AMD"
5. Run backtest → Compare results

## Best Practices

### Version Control
For major symbol changes, consider creating a new book:
```
Original book: "Momentum_Tech_Original"
Modified book: "Momentum_Tech_Expanded" (save as new)
```

### Naming Convention
Update book name to reflect symbol changes:
```
Before: "Momentum_FAANG"
After:  "Momentum_MegaCap_Tech" (if you added more symbols)
```

### Validation
After updating symbols:
1. ✅ Check success message
2. ✅ Verify symbol count
3. ✅ Review comparison view
4. ✅ Reload book to confirm changes persisted

### Testing
Test symbol changes across different time periods:
1. Update symbols
2. Test on multiple date ranges
3. Compare performance metrics
4. Keep or revert changes

## Technical Details

### How Updates Work

1. **User edits** text area
2. **Parse** comma-separated values
3. **Normalize** (uppercase, trim, deduplicate)
4. **Compare** with original symbol set
5. **Enable/disable** update button
6. **Save** book with new symbols (if button clicked)
7. **Refresh** page to show updated book

### What Gets Updated

When you update symbols:
- ✅ `symbols` field in book JSON
- ✅ `updated_at` timestamp
- ✅ Session state config symbols
- ❌ Strategy, parameters, tags unchanged
- ❌ Created date unchanged

### File Location

Books are stored in:
```
backtester2/
└── saved_books/
    └── {BookName}.json
```

After update, the JSON is overwritten with new symbol list.

## Troubleshooting

### Button Won't Enable
**Problem**: Update button stays disabled after editing

**Solutions**:
- Ensure symbols actually changed (order doesn't matter)
- Check for valid symbols (no empty list)
- Try removing/adding at least one symbol

### Symbols Not Persisting
**Problem**: After refresh, symbols revert to original

**Solutions**:
- Make sure you clicked "Update Book Symbols"
- Check for error messages
- Verify write permissions on `saved_books/` directory
- Check the JSON file directly to confirm update

### Duplicate Symbols
**Problem**: Same symbol appears multiple times

**Solution**: Duplicates are automatically removed during parsing

### Invalid Symbols
**Problem**: Entered invalid ticker symbols

**Note**: The system doesn't validate if symbols exist - it only validates format. Invalid tickers will fail during data loading when you run the backtest.

## Examples

### Example 1: Add Single Symbol
```
Load book: "MA_Cross_Tech"
Original:  AAPL, MSFT, GOOGL
Edit to:   AAPL, MSFT, GOOGL, NVDA
Update ✓
Result:    Book updated with 4 symbols
```

### Example 2: Replace Entire List
```
Load book: "Momentum_Strategy"
Original:  SPY, QQQ
Edit to:   VTI, VEA, VWO, AGG, GLD
Update ✓
Result:    Book updated with 5 symbols
```

### Example 3: Remove Symbols
```
Load book: "Large_Portfolio"
Original:  AAPL, MSFT, GOOGL, NVDA, TSLA, AMD, INTC
Edit to:   AAPL, MSFT, GOOGL
Update ✓
Result:    Book updated with 3 symbols
```

## Integration with Workflow

### Standard Workflow
```
1. Load book
2. (Optional) Edit symbols
3. (Optional) Edit parameters
4. Run backtest
5. Review results
6. (Optional) Save as new book with updated config
```

### Iteration Workflow
```
1. Load base book
2. Edit symbols → Update book
3. Run backtest → Analyze
4. Edit symbols → Update book
5. Run backtest → Compare
6. Repeat until optimal symbol set found
```

## Summary

The editable symbol feature provides:
- ✅ **Quick updates** - No JSON editing required
- ✅ **Visual feedback** - See changes before saving
- ✅ **Safe updates** - Button only enables when changes detected
- ✅ **Auto-refresh** - Page reloads with updated book
- ✅ **Validation** - Auto-uppercase, trim, deduplicate

This makes it easy to iterate on symbol universes and test different combinations without leaving the Streamlit UI!
