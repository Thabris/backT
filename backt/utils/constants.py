"""
Constants used throughout BackT

Defines standard constants, default values, and configuration options
used across the backtesting framework.
"""

# Default execution parameters
DEFAULT_COMMISSION = 0.001  # per share
DEFAULT_SPREAD = 0.01  # bid-ask spread
DEFAULT_SLIPPAGE = 0.0005  # as percentage of trade size

# Supported data frequencies
SUPPORTED_FREQUENCIES = [
    "1min", "5min", "15min", "30min", "1H", "4H",
    "1D", "1W", "1M", "1Q", "1Y"
]

# Standard OHLCV columns
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["adj_close", "vwap", "dividends", "splits"]

# Position sides
LONG_SIDE = "long"
SHORT_SIDE = "short"

# Order actions
BUY_ACTION = "buy"
SELL_ACTION = "sell"
CLOSE_ACTION = "close"
HOLD_ACTION = "hold"
TARGET_WEIGHT_ACTION = "target_weight"

VALID_ACTIONS = {BUY_ACTION, SELL_ACTION, CLOSE_ACTION, HOLD_ACTION, TARGET_WEIGHT_ACTION}

# Order types
MARKET_ORDER = "market"
LIMIT_ORDER = "limit"
STOP_ORDER = "stop"
STOP_LIMIT_ORDER = "stop_limit"

VALID_ORDER_TYPES = {MARKET_ORDER, LIMIT_ORDER, STOP_ORDER, STOP_LIMIT_ORDER}

# Risk metrics constants
TRADING_DAYS_PER_YEAR = 252
TRADING_HOURS_PER_DAY = 24  # for crypto/forex
TRADING_MINUTES_PER_DAY = 1440

# Annualization factors for different frequencies
ANNUALIZATION_FACTORS = {
    "1min": TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY,
    "5min": TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / 5,
    "15min": TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / 15,
    "30min": TRADING_DAYS_PER_YEAR * TRADING_MINUTES_PER_DAY / 30,
    "1H": TRADING_DAYS_PER_YEAR * 24,
    "4H": TRADING_DAYS_PER_YEAR * 6,
    "1D": TRADING_DAYS_PER_YEAR,
    "1W": 52,
    "1M": 12,
    "1Q": 4,
    "1Y": 1
}

# Default configuration values
DEFAULT_INITIAL_CAPITAL = 100000.0
DEFAULT_LOOKBACK_WINDOW = 252
DEFAULT_MAX_LEVERAGE = 1.0
DEFAULT_RISK_FREE_RATE = 0.0

# File extensions for different output formats
OUTPUT_FORMATS = {
    "csv": ".csv",
    "parquet": ".parquet",
    "json": ".json",
    "pickle": ".pkl"
}

# Timezone handling
DEFAULT_TIMEZONE = "UTC"
MARKET_TIMEZONES = {
    "NYSE": "America/New_York",
    "NASDAQ": "America/New_York",
    "LSE": "Europe/London",
    "TSE": "Asia/Tokyo",
    "HKEX": "Asia/Hong_Kong",
    "ASX": "Australia/Sydney"
}

# Data validation tolerances
PRICE_TOLERANCE = 1e-8  # for floating point price comparisons
VOLUME_TOLERANCE = 1e-6  # for volume comparisons

# Logging levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"

# Performance metric thresholds
GOOD_SHARPE_RATIO = 1.0
EXCELLENT_SHARPE_RATIO = 2.0
MAX_ACCEPTABLE_DRAWDOWN = 0.2  # 20%

# Cache settings
DEFAULT_CACHE_SIZE = 1000  # number of items to cache
CACHE_EXPIRY_HOURS = 24  # hours before cache expires