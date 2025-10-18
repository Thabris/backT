# FinancialModelingPrep API Setup

BackT now supports FinancialModelingPrep (FMP) API for enhanced fundamental data, with automatic fallback to yfinance.

## Why Use FMP?

**Advantages over yfinance:**
- ✅ More reliable fundamental data (proper API vs scraping)
- ✅ Historical quarterly/annual statements (30+ years)
- ✅ Pre-calculated financial ratios and metrics
- ✅ Standardized data format across all stocks
- ✅ Better coverage and data completeness

**Free Tier Limits:**
- 250 API calls per day
- 500MB bandwidth per month
- ~99 symbols available (AAPL, TSLA, AMZN + 96 more major stocks)
- End-of-day data (perfect for backtesting)

## Getting Your Free API Key

1. Go to https://site.financialmodelingprep.com/developer/docs
2. Scroll down and enter your email in the signup box
3. Verify your email
4. Your free API key will be generated automatically

## Using FMP with BackT

### Method 1: Environment Variable (Recommended)

Set the `FMP_API_KEY` environment variable:

**Windows (PowerShell):**
```powershell
$env:FMP_API_KEY = "your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set FMP_API_KEY=your_api_key_here
```

**Linux/Mac:**
```bash
export FMP_API_KEY=your_api_key_here
```

**Permanent (add to `.env` file in project root):**
```
FMP_API_KEY=your_api_key_here
```

### Method 2: Pass Directly to Loader

```python
from backt.data.fundamentals import FundamentalsLoader

# With FMP API key
loader = FundamentalsLoader(fmp_api_key='your_api_key_here')

# Auto mode (tries FMP first, falls back to yfinance)
fundamentals = loader.get_fundamentals('AAPL')
```

### Method 3: Force Specific Data Source

```python
# Use FMP only (no fallback)
loader = FundamentalsLoader(
    fmp_api_key='your_api_key_here',
    data_source='fmp'
)

# Use yfinance only (no FMP)
loader = FundamentalsLoader(data_source='yfinance')

# Auto mode with fallback (default)
loader = FundamentalsLoader(
    fmp_api_key='your_api_key_here',
    data_source='auto'
)
```

## Monitoring API Usage

The loader tracks your API usage to stay within free tier limits:

```python
loader = FundamentalsLoader(fmp_api_key='your_key')

# Load some data
loader.get_fundamentals('AAPL')
loader.get_fundamentals('TSLA')
loader.get_fundamentals('GOOGL')

# Check usage
stats = loader.get_api_usage_stats()
print(f"API calls today: {stats['fmp_calls_today']}/250")
print(f"Remaining calls: {stats['fmp_calls_remaining']}")
print(f"Cache size: {stats['cache_size']} symbols")
```

## Caching

All fundamental data is cached for 24 hours by default to minimize API calls:

```python
# Default cache (24 hours)
loader = FundamentalsLoader()

# Custom cache duration (12 hours)
loader = FundamentalsLoader(cache_ttl=43200)

# Clear cache manually
loader.clear_cache()
```

## Using with AQR Strategies

AQR factor strategies automatically use the configured loader:

```python
from backt import Backtester, BacktestConfig
from backt.data import YahooDataLoader
from strategies.aqr import quality_minus_junk

# Configure backtest
config = BacktestConfig(
    start_date='2020-01-01',
    end_date='2023-12-31',
    initial_capital=100000
)

# Load market data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
data_loader = YahooDataLoader()
market_data = data_loader.load_symbols(symbols, config.start_date, config.end_date)

# Run backtest (will auto-use FMP if FMP_API_KEY is set)
backtester = Backtester(config)
result = backtester.run(
    market_data=market_data,
    strategy_func=quality_minus_junk
)
```

The strategy will automatically:
1. Try to load fundamentals from FMP (if API key available)
2. Fall back to yfinance if FMP unavailable
3. Cache all data to minimize API calls

## Streamlit Web Interface

The Streamlit app automatically detects FMP configuration:

```bash
# Set environment variable first
export FMP_API_KEY=your_api_key_here

# Launch app
streamlit run streamlit_backtest_runner.py
```

The app will show FMP status in the sidebar when running AQR strategies.

## Data Available from FMP

**Financial Ratios:**
- Profitability: ROE, ROA, profit margins, gross margins, operating margins
- Leverage: Debt-to-equity, current ratio, quick ratio
- Valuation: P/E, P/B, PEG ratio
- Efficiency: Asset turnover, inventory turnover

**Key Metrics:**
- Revenue per share, EPS, book value per share
- Free cash flow, operating cash flow
- Enterprise value, market cap
- Revenue growth, earnings growth

**Company Profile:**
- Sector, industry classification
- Beta (market risk)
- Company description

**Financial Statements (available but not yet integrated):**
- Income Statement (quarterly/annual)
- Balance Sheet (quarterly/annual)
- Cash Flow Statement (quarterly/annual)

## Troubleshooting

### "FMP API daily limit reached"
You've hit the 250 calls/day limit. Either:
- Wait until tomorrow (resets at midnight)
- Upgrade to a paid FMP plan
- Use yfinance fallback (automatic in auto mode)

### "FMP API error 401: Unauthorized"
Your API key is invalid or expired. Get a new key from FMP.

### "FMP data unavailable for [symbol]"
The symbol might not be in the free tier (limited to ~99 symbols). Loader will automatically fall back to yfinance.

### Data seems incomplete
Some fields may be None/null even with FMP. This is normal - not all companies report all metrics. The quality/value scoring functions handle missing data gracefully.

## Upgrading to Paid FMP

If you need more than 250 calls/day or access to all stocks:

- **Starter Plan**: $14/month - 750 calls/day, all US stocks
- **Developer Plan**: $29/month - 5,000 calls/day, global coverage
- **Premium Plan**: $99/month - 50,000 calls/day, real-time data

See https://site.financialmodelingprep.com/pricing-plans for details.

## Support

For FMP API issues:
- FMP Documentation: https://site.financialmodelingprep.com/developer/docs
- FMP Support: Contact through their website

For BackT integration issues:
- Open a GitHub issue
- Check BackT documentation
