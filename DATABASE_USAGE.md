# Market Data Database - Quick Reference

## Overview

The local SQLite database system provides fast, reliable access to historical ETF data without Yahoo Finance rate limiting.

**Benefits:**
- ⚡ Instant data loading (no API delays)
- 🔒 No rate limiting
- 📊 100-150 MB for 15 years of 96 ETFs
- ✅ Data consistency validation
- 🔄 Easy incremental updates

---

## Initial Setup (One-Time)

### Download 15 Years of Data

```bash
python download_market_data.py --years 15
```

**What happens:**
- Downloads OHLCV data for all 96 ETF symbols
- Validates data consistency (OHLC relationships, volume checks)
- Detects and reports gaps in data
- Takes ~2-3 minutes with 1-second delays
- Creates `market_data.db` file

### Download Custom Date Range

```bash
python download_market_data.py --start 2020-01-01 --end 2025-12-31
```

### Download with Faster Processing (Less Safe)

```bash
python download_market_data.py --years 10 --delay 0.5
```

**Note:** Lower delays increase Yahoo Finance rate limit risk

---

## Daily/Weekly Updates

### Update All Symbols

```bash
python update_market_data.py
```

**What happens:**
- Checks last date for each symbol
- Fetches only new data since last update
- Takes ~1-2 minutes for 96 symbols
- No data lost if interrupted

### Custom Database Path

```bash
python update_market_data.py --db my_data.db
```

---

## Using Database in Backtests

### Option 1: Use SQLiteDataLoader (Recommended)

```python
from backt import Backtester, BacktestConfig
from backt.data.sqlite_loader import SQLiteDataLoader
from strategies.momentum import ma_crossover_long_only

# Create SQLite data loader instead of Yahoo
data_loader = SQLiteDataLoader("market_data.db")

# Create backtester
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000
)

backtester = Backtester(config, data_loader=data_loader)

# Run backtest - instant data loading!
result = backtester.run(
    strategy=ma_crossover_long_only,
    universe=['SPY', 'QQQ', 'TLT', 'GLD'],
    strategy_params={}
)

print(result.performance_metrics)
```

### Option 2: Direct Database Access

```python
from backt.data.market_data_db import MarketDataDB

db = MarketDataDB("market_data.db")

# Get data for specific symbol
spy_data = db.get_data('SPY', start_date='2020-01-01', end_date='2023-12-31')

# Get all available symbols
symbols = db.get_symbols()

# Get metadata
metadata = db.get_metadata('SPY')
print(f"Data quality: {metadata['data_quality_score']:.1f}%")

# Check for gaps
gaps = db.detect_gaps('SPY')
for gap_start, gap_end, days_missing in gaps:
    print(f"Gap: {gap_start} to {gap_end} ({days_missing} days)")

db.close()
```

---

## Using with Ranking Script

Modify `rank_symbols_by_strategy.py` to use SQLite:

```python
# At the top, change the import:
from backt.data.sqlite_loader import SQLiteDataLoader

# In run_single_symbol_backtest(), replace:
data_loader = YahooDataLoader()

# With:
data_loader = SQLiteDataLoader("market_data.db")
```

**Result:** Rank all 96 symbols in ~30 seconds instead of ~3 minutes!

---

## Database Management

### Check Database Statistics

```python
from backt.data.market_data_db import MarketDataDB

db = MarketDataDB("market_data.db")

symbols = db.get_symbols()
print(f"Total symbols: {len(symbols)}")

for symbol in symbols[:10]:  # First 10
    meta = db.get_metadata(symbol)
    print(f"{symbol:8} {meta['first_date']} to {meta['last_date']} "
          f"({meta['total_records']} days, Q:{meta['data_quality_score']:.1f}%)")

db.close()
```

### Export Symbol Data to CSV

```python
from backt.data.market_data_db import MarketDataDB

db = MarketDataDB("market_data.db")
data = db.get_data('SPY')
data.to_csv('SPY_data.csv')
db.close()
```

### Delete Corrupted Symbol

```python
from backt.data.market_data_db import MarketDataDB

db = MarketDataDB("market_data.db")
db.conn.execute("DELETE FROM market_data WHERE symbol = ?", ('BAD_SYMBOL',))
db.conn.execute("DELETE FROM metadata WHERE symbol = ?", ('BAD_SYMBOL',))
db.conn.commit()
db.close()
```

---

## Data Quality

### Quality Score Calculation

The database calculates a quality score (0-100%) for each symbol based on:
- **Data completeness**: Fewer gaps = higher score
- **Expected vs actual trading days**: ~252 days/year expected
- **OHLCV validation**: High/Low relationships, volume consistency

### Validation Checks

When inserting data, the system validates:
1. **High ≥ Low** (always)
2. **High ≥ Open, Close** (always)
3. **Low ≤ Open, Close** (always)
4. **Volume ≥ 0** (always)
5. **Zero volume detection** (warning only)

### Gap Detection

Gaps are detected when:
- More than 3 calendar days between data points
- Estimated trading days missed: `(calendar_days × 252/365) - 1`

---

## File Locations

- **Database file**: `market_data.db` (or custom path)
- **Download script**: `download_market_data.py`
- **Update script**: `update_market_data.py`
- **Data loader**: `backt/data/sqlite_loader.py`
- **Database class**: `backt/data/market_data_db.py`

---

## Troubleshooting

### "No data found for symbol"

Check if symbol is in database:
```python
from backt.data.market_data_db import MarketDataDB
db = MarketDataDB()
print(db.get_symbols())
```

If missing, download it:
```bash
python download_market_data.py --years 10
```

### Database corrupted

Delete and re-download:
```bash
del market_data.db
python download_market_data.py --years 15
```

### Slow queries

Database uses optimized indexes. If still slow, vacuum:
```python
from backt.data.market_data_db import MarketDataDB
db = MarketDataDB()
db.conn.execute("VACUUM")
db.close()
```

---

## Performance Comparison

| Operation | Yahoo Finance | SQLite Database |
|-----------|---------------|-----------------|
| Single symbol load | 1-2 seconds | <0.01 seconds |
| 96 symbols load | 2-3 minutes | <1 second |
| Ranking script | 5-10 minutes | 30 seconds |
| Rate limiting risk | High | None |

---

## Recommended Workflow

1. **Initial setup** (once):
   ```bash
   python download_market_data.py --years 15
   ```

2. **Weekly updates** (automated):
   ```bash
   python update_market_data.py
   ```

3. **Use in backtests**:
   - Replace `YahooDataLoader()` with `SQLiteDataLoader()`
   - Enjoy instant data loading!

4. **Backup** (optional):
   ```bash
   copy market_data.db market_data_backup.db
   ```
