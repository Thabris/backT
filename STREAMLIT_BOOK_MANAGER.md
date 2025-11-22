# Streamlit Book Manager

A comprehensive web interface for managing your BackT strategy configurations (Books).

## Features

The Streamlit Book Manager provides a full-featured GUI for:

✅ **Symbol Management**
- Add/remove symbols with visual feedback
- Quick add/remove buttons
- Bulk editing via text area
- Real-time diff showing changes

✅ **Parameter Editing**
- Type-aware editing (int, float, bool, string)
- Visual comparison of old → new values
- Add new parameters
- Remove parameters

✅ **Metadata Management**
- Edit descriptions
- Manage tags for categorization
- View creation/update timestamps

✅ **Safety Features**
- Preview all changes before saving
- Automatic backups on save
- Revert changes
- Validation checks

✅ **Backup Management**
- View all backups by book
- Restore from any backup
- Timestamped backup history

✅ **Filtering & Organization**
- Filter by strategy type
- Filter by tags
- Search by name
- Summary statistics

---

## Quick Start

### Launch the Book Manager

**Option 1: Double-click the launcher**
```
launch_book_manager.bat
```

**Option 2: Command line**
```bash
streamlit run streamlit_book_manager.py
```

Your browser will open automatically at `http://localhost:8501`

---

## Interface Overview

### Sidebar (Left)

**📚 Book Manager**
- Total books count
- Filter by strategy
- Filter by tags
- Book selection dropdown
- Load book button
- Refresh and New buttons

### Main Area (Right)

When no book is loaded:
- List of all available books
- Backup manager

When a book is loaded:
- **Header**: Book name, status, quick stats
- **Save Controls**: Save, Revert, Validate, Delete
- **Three Tabs**:
  1. **📋 Symbols** - Edit symbol list
  2. **⚙️ Parameters** - Edit strategy parameters
  3. **📝 Metadata** - Edit description and tags

---

## Step-by-Step Usage

### 1. Load a Book

1. Launch the Book Manager
2. In the sidebar, select a book from the dropdown
3. Click **"📖 Load Book"**
4. The book details appear in the main area

### 2. Edit Symbols

**In the "📋 Symbols" tab:**

**Method A: Bulk Edit**
1. Edit the text area (comma-separated symbols)
2. Click **"✅ Apply Symbol Changes"**

**Method B: Quick Add**
1. Type symbol in "Quick Add" field
2. Click **"➕ Add"**

**Method C: Quick Remove**
1. Select symbol from "Quick Remove" dropdown
2. Click **"➖ Remove"**

**View Changes:**
- Expand "View Changes" to see diff
- Green = Added symbols
- Red = Removed symbols

### 3. Edit Parameters

**In the "⚙️ Parameters" tab:**

1. Each parameter shows with appropriate input:
   - Checkboxes for booleans
   - Number inputs for int/float
   - Text inputs for strings

2. Edit values as needed
3. Changed values show: `old → new`
4. Click **"💾 Apply Parameter Changes"**

**Add New Parameter:**
1. Scroll to "Add New Parameter" section
2. Enter parameter name
3. Enter value
4. Select type (str, int, float, bool)
5. Click **"💾 Apply Parameter Changes"**

### 4. Edit Metadata

**In the "📝 Metadata" tab:**

**Description:**
1. Edit the description text area
2. Click **"💾 Update Description"**

**Tags:**
1. Edit tags (comma-separated)
2. Click **"💾 Update Tags"**

### 5. Save Changes

**When you make any changes:**
- Yellow warning appears: "⚠️ Unsaved changes"
- Changes are held in memory (not saved to disk)

**To save:**
1. Click **"💾 Save Changes"** (top of page)
2. A backup is automatically created
3. Green confirmation: "✅ Book saved"
4. Status changes to "✅ No unsaved changes"

### 6. Revert Changes

**If you made a mistake:**
1. Click **"↩️ Revert Changes"**
2. All unsaved changes are discarded
3. Book returns to last saved state

### 7. Validate Book

**To check for issues:**
1. Click **"🔍 Validate"**
2. Results show:
   - ✅ No issues found, or
   - ⚠️ List of warnings (e.g., duplicate symbols, empty fields)

### 8. Delete Book

**To delete a book:**
1. Click **"🗑️ Delete Book"**
2. Confirmation prompt appears
3. Click **"✅ Confirm Delete"** or **"❌ Cancel"**

### 9. Restore from Backup

**In the main area (when no book loaded):**
1. Scroll to **"🔄 Backups"** section
2. Expand the book you want to restore
3. Find the backup timestamp
4. Click **"♻️ Restore"**
5. The book is restored from that backup

---

## Comparison: Streamlit vs CLI vs Existing Interface

| Feature | Streamlit Manager | CLI (edit_book.py) | Existing Streamlit |
|---------|-------------------|--------------------|--------------------|
| **Symbol Editing** | ✅ Visual + Quick buttons | ✅ Command-line | ✅ Text area only |
| **Parameter Editing** | ✅ Type-aware widgets | ✅ Command-line | ❌ Not available |
| **Metadata Editing** | ✅ Forms | ✅ Command-line | ❌ Not available |
| **Change Preview** | ✅ Visual diff | ✅ Text diff | ⚠️ Basic comparison |
| **Validation** | ✅ One-click | ✅ Function call | ❌ Not available |
| **Backups** | ✅ Auto + Restore UI | ✅ Auto + CLI restore | ❌ Not available |
| **Delete/Rename** | ✅ GUI | ✅ CLI | ❌ Not available |
| **Filtering** | ✅ By strategy/tag | ❌ Not available | ❌ Not available |
| **User Experience** | 🌟 Best for interactive | 🌟 Best for automation | ⚠️ Limited |

---

## Screenshots & Examples

### Example 1: Adding Symbols

**Before:**
```
Symbols: SPY, QQQ, TLT
```

**Steps:**
1. Go to Symbols tab
2. Type `IEF` in Quick Add
3. Click ➕ Add
4. Type `GLD` in Quick Add
5. Click ➕ Add
6. Click 💾 Save Changes

**After:**
```
Symbols: GLD, IEF, QQQ, SPY, TLT
```

### Example 2: Optimizing Parameters

You ran parameter optimization and found better values:

**Before:**
```
fast_period: 11
slow_period: 20
signal_period: 10
```

**Steps:**
1. Go to Parameters tab
2. Change `fast_period` to `12`
3. Change `slow_period` to `26`
4. Change `signal_period` to `9`
5. See changes: `11 → 12`, `20 → 26`, `10 → 9`
6. Click 💾 Apply Parameter Changes
7. Click 💾 Save Changes

**After:**
```
fast_period: 12
slow_period: 26
signal_period: 9
```

### Example 3: Filtering Books

**Scenario:** You have 20 books and want to find all MACD strategies

**Steps:**
1. In sidebar, select "Strategy" dropdown
2. Choose `macd_crossover`
3. Book list filters to show only MACD books
4. Select one to edit

### Example 4: Restoring from Backup

**Scenario:** You accidentally deleted all symbols and saved

**Steps:**
1. Don't load any book
2. Scroll to Backups section
3. Expand your book name
4. Find the most recent backup (before the mistake)
5. Click ♻️ Restore
6. Your book is restored!

---

## Advanced Usage

### Workflow 1: Updating Multiple Books

**Scenario:** You want to add SPY to all momentum books

1. Filter by strategy: `Momentum`
2. Load first book
3. Add SPY symbol
4. Save
5. Click "🔄 Refresh" in sidebar
6. Load next book
7. Repeat

### Workflow 2: Cleaning Up Old Books

1. Load book
2. Click Validate
3. If warnings found (duplicates, etc.), fix them
4. Save
5. Move to next book

### Workflow 3: Creating Variations

1. Load base book (e.g., "MACD_Top10")
2. Edit symbols (remove some)
3. Click Save (you'll see an error - name exists)
4. Workaround: Use CLI to create a copy first:
   ```python
   from backt.utils import BookManager
   manager = BookManager()
   book = manager.load_book("MACD_Top10")
   book.name = "MACD_Top5"
   book.symbols = book.symbols[:5]
   manager.save_book(book)
   ```
5. Then edit in Streamlit

---

## Tips & Best Practices

1. **Always save after editing** - Changes are in memory until you save
2. **Use validation** before saving important books
3. **Check backups** - They're created automatically but verify occasionally
4. **Use descriptive tags** - Makes filtering easier
5. **Filter before editing** - Narrow down books to find the right one
6. **Preview changes** - Expand diff views before saving

---

## Troubleshooting

### Issue: Book doesn't appear in list

**Solution:** Click "🔄 Refresh" in sidebar

### Issue: Changes not saved

**Cause:** Clicked Apply but not Save
**Solution:** Click **💾 Save Changes** at the top

### Issue: Can't delete book

**Cause:** Confirmation required
**Solution:** Click "🗑️ Delete Book", then "✅ Confirm Delete"

### Issue: Want to undo changes after saving

**Solution:** Use **🔄 Backups** section to restore

### Issue: Streamlit won't start

**Solution:**
```bash
# Check if Streamlit is installed
pip install streamlit

# Launch manually
streamlit run streamlit_book_manager.py
```

---

## Integration with Backtesting Workflow

### Typical Workflow:

1. **Rank Symbols** (CLI):
   ```bash
   python rank_symbols_by_strategy.py --strategy macd_crossover --save "MACD_Top10"
   ```

2. **Edit Book** (Streamlit):
   - Launch Book Manager
   - Load "MACD_Top10"
   - Remove underperformers
   - Adjust parameters based on optimization
   - Save

3. **Backtest** (Main Streamlit App):
   - Launch `streamlit run streamlit_app.py`
   - Select "Load from Saved Book"
   - Choose "MACD_Top10"
   - Run backtest with new date range

4. **Monitor Performance** (Periodic):
   - Re-rank symbols monthly
   - Compare with book symbols
   - Update book in Book Manager

---

## Keyboard Shortcuts

While Streamlit doesn't have custom shortcuts, you can use browser shortcuts:

- **Ctrl+R** - Refresh page (useful after external changes)
- **F11** - Fullscreen mode
- **Ctrl+F** - Find text on page

---

## Future Enhancements

Potential future features:
- Bulk operations (edit multiple books)
- Book comparison side-by-side
- Import/export books
- Book templates
- Performance tracking in metadata
- Visual symbol selector
- Parameter validation rules

---

## Summary

The **Streamlit Book Manager** provides a complete, user-friendly interface for managing your strategy configurations:

✅ **Comprehensive** - Edit symbols, parameters, and metadata
✅ **Safe** - Automatic backups, validation, revert
✅ **Visual** - See changes before saving
✅ **Organized** - Filter and search books
✅ **Integrated** - Works seamlessly with ranking and backtesting

**Launch it now:**
```bash
streamlit run streamlit_book_manager.py
```

Or double-click `launch_book_manager.bat`
