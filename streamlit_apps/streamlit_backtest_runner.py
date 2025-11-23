"""
Multi-Sheet Streamlit Backtest Runner

A comprehensive web interface for running backtests with:
1. Configuration Sheet: Set backtest parameters, fees, dates, universe
2. Strategy Sheet: Select strategy and configure parameters
3. Results Sheet: View metrics, charts, and analysis

Run with: streamlit run streamlit_backtest_runner.py
"""

import sys
from pathlib import Path

# Add parent directory (project root) to path to find backt and strategies modules
# Use absolute path to ensure it works regardless of how the script is run
project_root = Path(r"C:\Users\maxim\Documents\Projects\backtester2")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import warnings
import logging

# Suppress Streamlit worker warnings
warnings.filterwarnings('ignore', message='.*missing ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*No runtime found.*')
warnings.filterwarnings('ignore', message='.*to view a Streamlit app.*')

# Suppress Streamlit logging warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from typing import Dict, Any
import importlib
import inspect
import math
from functools import lru_cache

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.reporting import PerformanceReport
from backt.utils.config import ExecutionConfig
from backt.validation import CPCVValidator, CPCVConfig, ParameterGrid
from backt.validation.overfitting import interpret_overfitting_metrics

# Import all strategies from strategies module
from strategies import momentum, aqr


# ===== ETF Universe Data Structure =====

ETF_UNIVERSE = {
    "🏦 Equities - Broad Market": {
        "SPY": "S&P 500",
        "QQQ": "Nasdaq 100",
        "IVV": "S&P 500 (iShares)",
        "VOO": "S&P 500 (Vanguard)",
        "MDY": "S&P Mid Cap 400",
        "IJH": "S&P Mid Cap 400 (iShares)",
        "IWM": "Russell 2000 Small Cap",
        "VB": "Small Cap (Vanguard)",
        "VGK": "Europe Stocks",
        "IEV": "Europe 350",
        "EEM": "Emerging Markets",
        "VWO": "Emerging Markets (Vanguard)",
        "EWJ": "Japan",
        "FXI": "China Large Cap",
        "MCHI": "China",
        "EFA": "Developed ex-US",
    },
    "📊 Equities - Factor/Style": {
        "VLUE": "Value Factor",
        "IWD": "Value Large Cap",
        "VTV": "Value (Vanguard)",
        "MTUM": "Momentum Factor",
        "QUAL": "Quality Factor",
        "SPHQ": "Quality (Invesco)",
        "USMV": "Low Volatility",
        "SPLV": "Low Volatility (Invesco)",
        "IJR": "Small Cap Value",
        "VBK": "Small Cap Growth",
        "LRGF": "Multifactor Large Cap",
        "ACWF": "Multifactor Global",
    },
    "💰 Fixed Income": {
        "AGG": "US Aggregate Bond",
        "BND": "Total Bond Market",
        "SHY": "1-3 Year Treasury",
        "VGSH": "Short-Term Treasury",
        "TLT": "20+ Year Treasury",
        "IEF": "7-10 Year Treasury",
        "TIP": "TIPS Inflation-Protected",
        "SCHP": "TIPS (Schwab)",
        "LQD": "Investment Grade Corporate",
        "VCIT": "Intermediate Corporate",
        "HYG": "High Yield Corporate",
        "JNK": "High Yield (SPDR)",
        "EMB": "Emerging Market Bond",
        "VWOB": "Emerging Market Bond (Vanguard)",
    },
    "🪙 Commodities": {
        "DBC": "Commodity Index",
        "COMT": "Commodity Optimum Yield",
        "GSG": "Commodity Broad",
        "GLD": "Gold",
        "IAU": "Gold (iShares)",
        "SLV": "Silver",
        "USO": "Crude Oil WTI",
        "BNO": "Brent Crude",
        "DBA": "Agriculture",
        "DBB": "Industrial Metals",
    },
    "🌍 Currencies": {
        "UUP": "US Dollar Bull",
        "USDU": "US Dollar (WisdomTree)",
        "FXE": "Euro",
        "FXY": "Japanese Yen",
        "FXB": "British Pound",
        "CEW": "Emerging Market Currency",
    },
    "💹 Volatility": {
        "VXX": "VIX Short-Term Futures",
        "UVXY": "VIX 2x Leveraged",
        "VIXM": "VIX Mid-Term Futures",
        "TAIL": "Tail Risk Hedge",
        "SVXY": "Short VIX",
    },
    "🧮 Alternative": {
        "RPAR": "Risk Parity ETF",
        "NTSX": "90/60 Stocks/Bonds",
        "AOR": "Moderate Allocation",
        "DBMF": "Managed Futures",
        "KMLM": "Managed Futures (KFA)",
        "FIG": "Global Macro",
        "GVAL": "Global Value",
        "COM": "Commodity Trend",
    },
    "🧭 Sector": {
        "XLK": "Technology",
        "VGT": "Technology (Vanguard)",
        "XLE": "Energy",
        "VDE": "Energy (Vanguard)",
        "XLF": "Financials",
        "VFH": "Financials (Vanguard)",
        "XLV": "Healthcare",
        "VHT": "Healthcare (Vanguard)",
        "XLI": "Industrials",
        "VIS": "Industrials (Vanguard)",
        "XLY": "Consumer Discretionary",
        "VCR": "Consumer Discretionary (Vanguard)",
        "XLU": "Utilities",
        "VPU": "Utilities (Vanguard)",
        "XLRE": "Real Estate",
        "VNQ": "REIT (Vanguard)",
    },
    "⚙️ Leveraged/Inverse": {
        "SPXL": "S&P 500 3x Bull",
        "SPXS": "S&P 500 3x Bear",
        "TQQQ": "Nasdaq 100 3x Bull",
        "SQQQ": "Nasdaq 100 3x Bear",
        "TMF": "Treasury 3x Bull",
        "TMV": "Treasury 3x Bear",
        "UGL": "Gold 2x Bull",
        "UCO": "Crude Oil 2x Bull",
        "SCO": "Crude Oil 2x Bear",
    }
}

# Quick preset portfolios
ETF_PRESETS = {
    "Classic 60/40": ["SPY", "AGG"],
    "All Weather": ["SPY", "TLT", "IEF", "GLD", "DBC"],
    "Risk Parity": ["SPY", "TLT", "GLD", "DBC", "VNQ"],
    "Global Diversified": ["SPY", "EFA", "EEM", "AGG", "VNQ", "DBC"],
    "Momentum Rotation": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "Defensive": ["USMV", "SPLV", "XLU", "VPU", "AGG", "TLT"],
    "Growth Focused": ["QQQ", "XLK", "VGT", "MTUM", "VBK"],
    "Trend Following": ["SPY", "TLT", "GLD", "DBC", "VNQ", "EFA"],
    "Factor Multi-Strategy": ["VLUE", "MTUM", "QUAL", "USMV"],
    "Commodity Macro": ["GLD", "SLV", "DBC", "USO", "UUP"],
}


# Page configuration
st.set_page_config(
    page_title="BackT Backtest Runner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)


# Minimal CSS - Only hide sidebar
st.markdown("""
<style>
    /* HIDE SIDEBAR COMPLETELY */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    /* Hide sidebar collapse button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


def load_spy_benchmark_data_from_backtest(backtest_result, initial_capital: float):
    """
    Load SPY benchmark data using the same date range as the backtest
    Returns tuple: (benchmark_df, error_message)
    """
    # Extract date range from backtest equity curve
    start_date = backtest_result.equity_curve.index[0].strftime('%Y-%m-%d')
    end_date = backtest_result.equity_curve.index[-1].strftime('%Y-%m-%d')

    # Load from Yahoo Finance (with caching)
    return _load_spy_from_yahoo(start_date, end_date, initial_capital)


@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def _load_spy_from_yahoo(start_date: str, end_date: str, initial_capital: float):
    """
    Load SPY data from Yahoo Finance with caching
    Internal function called by load_spy_benchmark_data_from_backtest
    """
    try:
        from backt.data.loaders import YahooDataLoader
        loader = YahooDataLoader()

        # Load SPY - when loading a single symbol, loader returns DataFrame directly
        spy_data = loader.load(
            ['SPY'],
            start_date,
            end_date
        )

        # Check if we got data (could be DataFrame for single symbol or dict)
        if spy_data is not None:
            # If it's a dict (multiple symbols), extract SPY
            if isinstance(spy_data, dict):
                if 'SPY' in spy_data:
                    spy_prices = spy_data['SPY']['close']
                else:
                    return (None, "SPY not in returned data")
            # If it's a DataFrame (single symbol), use directly
            elif isinstance(spy_data, pd.DataFrame):
                if 'close' in spy_data.columns:
                    spy_prices = spy_data['close']
                else:
                    return (None, "Close prices not found in SPY data")
            else:
                return (None, f"Unexpected data type: {type(spy_data)}")

            initial_shares = initial_capital / spy_prices.iloc[0]
            benchmark_equity = spy_prices * initial_shares

            benchmark_df = pd.DataFrame({
                'total_equity': benchmark_equity,
                'total_pnl': benchmark_equity - initial_capital
            }, index=spy_prices.index)

            return (benchmark_df, None)
        else:
            return (None, "SPY data not available from Yahoo Finance. Check your internet connection.")
    except ImportError as e:
        if 'yfinance' in str(e):
            return (None, "yfinance not installed. Install with: pip install yfinance")
        return (None, f"Import error: {str(e)}")
    except Exception as e:
        return (None, f"Error loading SPY data: {str(e)}")

    return (None, "Unknown error")


@st.cache_data(max_entries=50, show_spinner=False)
def calculate_monthly_metric_series(equity_curve_hash: str, equity_curve_dict: dict, metric: str):
    """
    Calculate monthly metric series with caching
    Uses hash to invalidate cache when equity curve changes
    """
    # Reconstruct equity_curve from dict
    equity_curve = pd.DataFrame({
        'total_equity': equity_curve_dict['total_equity'],
        'total_pnl': equity_curve_dict['total_pnl']
    })
    equity_curve.index = pd.to_datetime(equity_curve_dict['index'])

    # Calculate the metric
    if metric == 'returns':
        series = equity_curve['total_equity'].pct_change()
        agg_method = 'sum'
    elif metric == 'pnl':
        series = equity_curve['total_pnl']
        agg_method = 'sum'
    elif metric == 'drawdown':
        equity = equity_curve['total_equity']
        running_max = equity.expanding().max()
        series = (equity - running_max) / running_max
        agg_method = 'mean'
    elif metric == 'volatility':
        series = equity_curve['total_equity'].pct_change().rolling(20).std()
        agg_method = 'mean'
    elif metric == 'sharpe_ratio':
        returns = equity_curve['total_equity'].pct_change()
        series = (returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
        agg_method = 'mean'
    else:
        series = equity_curve['total_equity'].pct_change()
        agg_method = 'sum'

    # Create monthly pivot table
    df = pd.DataFrame({
        'value': series,
        'year': series.index.year,
        'month': series.index.month
    })

    # Aggregate by month
    if agg_method == 'sum':
        monthly = df.groupby(['year', 'month'])['value'].sum()
    else:
        monthly = df.groupby(['year', 'month'])['value'].mean()

    return monthly


def get_available_strategies():
    """
    Discover all available strategies from the strategies module
    Returns dict of {strategy_name: (module, function, docstring)}

    Note: Cannot be cached because it returns function objects which aren't hashable.
    However, this operation is fast enough (~100-200ms) that caching isn't critical.
    """
    strategies = {}

    # Import benchmark module
    from strategies import benchmark

    # Get all strategy modules
    strategy_modules = {
        'benchmark': benchmark,
        'momentum': momentum,
        'aqr': aqr,
        # Add more modules here as they're created
    }

    for module_name, module in strategy_modules.items():
        # Get all functions in the module
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith('_'):  # Skip private functions
                # Get docstring for description
                doc = inspect.getdoc(obj)
                desc = doc.split('\n')[0] if doc else name

                strategies[name] = {
                    'module': module_name,
                    'function': obj,
                    'description': desc,
                    'docstring': doc
                }

    return strategies


@st.cache_data(show_spinner=False)
def extract_strategy_params(_strategy_func):
    """
    Extract parameter names and defaults from strategy docstring
    Returns dict of {param_name: {'type': type, 'default': value, 'description': str}}

    Cached to avoid repeated docstring parsing when strategy is selected.
    Note: _strategy_func has underscore prefix to tell Streamlit not to hash it (functions aren't hashable).
    """
    doc = inspect.getdoc(_strategy_func)
    if not doc:
        return {}

    params = {}
    in_params_section = False
    current_param = None

    for line in doc.split('\n'):
        line = line.strip()

        if 'Parameters:' in line or 'Parameters' in line:
            in_params_section = True
            continue

        if in_params_section:
            if line.startswith('-') or line == '':
                continue

            # End of parameters section
            if 'Returns:' in line or 'Example:' in line:
                break

            # Parse parameter line like "fast_ma : int, default=20"
            if ':' in line:
                parts = line.split(':', 1)
                param_name = parts[0].strip()
                param_info = parts[1].strip()

                # Extract type and default
                param_type = 'str'
                default_value = None
                description = param_info

                if 'int' in param_info.lower():
                    param_type = 'int'
                elif 'float' in param_info.lower():
                    param_type = 'float'
                elif 'bool' in param_info.lower():
                    param_type = 'bool'

                # Extract default value
                if 'default=' in param_info or 'default:' in param_info:
                    default_str = param_info.split('default')[-1].strip(' =:')
                    try:
                        if param_type == 'int':
                            default_value = int(default_str.split(',')[0].split()[0])
                        elif param_type == 'float':
                            default_value = float(default_str.split(',')[0].split()[0])
                        elif param_type == 'bool':
                            default_value = 'true' in default_str.lower()
                    except:
                        pass

                params[param_name] = {
                    'type': param_type,
                    'default': default_value,
                    'description': description
                }
                current_param = param_name
            elif current_param and line:
                # Continuation of description
                params[current_param]['description'] += ' ' + line

    return params


# ===== Book Management Caching Functions =====

@st.cache_resource(show_spinner=False)
def get_book_manager(books_dir: str):
    """
    Get cached BookManager instance

    Uses @st.cache_resource to maintain singleton BookManager.
    This avoids repeatedly instantiating BookManager on every access.

    Parameters:
    -----------
    books_dir : str
        Path to books directory

    Returns:
    --------
    BookManager
        Cached BookManager instance
    """
    from backt.utils.books import BookManager
    return BookManager(books_dir=books_dir)


@st.cache_data(ttl=60, show_spinner=False)
def list_available_books(books_dir: str):
    """
    List available books with 60-second TTL

    Caches book listing for 60 seconds to reduce filesystem scans.
    TTL ensures fresh data after new books are created.

    Parameters:
    -----------
    books_dir : str
        Path to books directory

    Returns:
    --------
    list
        List of available book names
    """
    manager = get_book_manager(books_dir)
    return manager.list_books()


@st.cache_data(ttl=300, show_spinner=False)
def load_book_cached(books_dir: str, book_name: str):
    """
    Load book with 5-minute TTL

    Caches loaded books for 5 minutes to avoid repeated JSON reads.
    TTL ensures fresh data after book edits.

    Parameters:
    -----------
    books_dir : str
        Path to books directory
    book_name : str
        Name of book to load

    Returns:
    --------
    Book
        Loaded book object
    """
    manager = get_book_manager(books_dir)
    return manager.load_book(book_name)


def render_smart_etf_selector():
    """
    Smart ETF selector with presets, categories, and manual input
    Returns: List of selected symbols
    """
    st.caption("🌍 **Trading Universe**")

    # Initialize session state for selected symbols
    if 'selected_symbols' not in st.session_state:
        st.session_state.selected_symbols = ["SPY", "QQQ", "TLT", "GLD"]

    # Selection mode tabs
    selection_mode = st.radio(
        "Selection Mode",
        ["Quick Presets", "Browse Categories", "Manual Input"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.write("")

    if selection_mode == "Quick Presets":
        # Preset selection
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_preset = st.selectbox(
                "Choose a preset portfolio",
                options=list(ETF_PRESETS.keys()),
                label_visibility="collapsed"
            )
        with col2:
            if st.button("Apply Preset", type="primary", width="stretch"):
                st.session_state.selected_symbols = ETF_PRESETS[selected_preset].copy()
                st.rerun()

        # Show preset details
        if selected_preset:
            preset_symbols = ETF_PRESETS[selected_preset]
            st.caption(f"**{selected_preset}:** {', '.join(preset_symbols)}")

    elif selection_mode == "Browse Categories":
        # Category-based selection
        st.caption("Select ETFs by category (click to expand)")

        for category_name, etfs in ETF_UNIVERSE.items():
            with st.expander(f"{category_name} ({len(etfs)} ETFs)", expanded=False):
                # Create checkbox grid (3 columns)
                cols = st.columns(3)
                for idx, (symbol, description) in enumerate(etfs.items()):
                    with cols[idx % 3]:
                        is_selected = symbol in st.session_state.selected_symbols
                        if st.checkbox(
                            f"**{symbol}** - {description}",
                            value=is_selected,
                            key=f"etf_{symbol}"
                        ):
                            if symbol not in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.append(symbol)
                        else:
                            if symbol in st.session_state.selected_symbols:
                                st.session_state.selected_symbols.remove(symbol)

    else:  # Manual Input
        # Manual text input
        current_symbols_str = ", ".join(st.session_state.selected_symbols)
        manual_input = st.text_area(
            "Enter symbols (comma-separated)",
            value=current_symbols_str,
            height=100,
            placeholder="SPY, QQQ, TLT, GLD, BTC-USD, AAPL...",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns([2, 1])
        with col2:
            if st.button("Update Symbols", type="primary", width="stretch"):
                # Parse manual input
                symbols = [s.strip().upper() for s in manual_input.split(',') if s.strip()]
                st.session_state.selected_symbols = symbols
                st.rerun()

        st.caption("You can enter any symbol including stocks, crypto (BTC-USD), or custom tickers")

    # Display current selection
    st.write("")
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.session_state.selected_symbols:
            symbols_display = ", ".join(st.session_state.selected_symbols[:10])
            if len(st.session_state.selected_symbols) > 10:
                symbols_display += f", ... (+{len(st.session_state.selected_symbols) - 10} more)"
            st.caption(f"**Selected ({len(st.session_state.selected_symbols)}):** {symbols_display}")
        else:
            st.caption("**No symbols selected**")

    with col2:
        if st.button("Clear All", width="stretch"):
            st.session_state.selected_symbols = []
            st.rerun()

    with col3:
        if st.button("Select All ETFs", width="stretch"):
            # Collect all ETFs from all categories
            all_etfs = []
            for category_etfs in ETF_UNIVERSE.values():
                all_etfs.extend(category_etfs.keys())
            st.session_state.selected_symbols = all_etfs
            st.rerun()

    return st.session_state.selected_symbols


def render_configuration_sheet():
    """Sheet 1: Backtest Configuration"""
    st.subheader("⚙️ Backtest Configuration")

    # Smart ETF Selector (outside form for interactivity)
    symbols = render_smart_etf_selector()

    st.write("")  # Small spacer

    with st.form("config_form"):
        # Date Range - compact with breathing room
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])
        with col1:
            st.caption("📅 **Dates**")
            start_date = st.date_input("Start", value=date(2012, 1, 2), max_value=date.today())  # First Monday of Jan 2012
        with col2:
            st.markdown("<p style='font-size: 0.8rem; color: transparent; margin-bottom: 0.3rem; margin-top: 0.4rem; line-height: 1.2;'>**.**</p>", unsafe_allow_html=True)
            end_date = st.date_input("End", value=date(2025, 10, 24), max_value=date.today())
        with col3:
            st.caption("💰 **Capital**")
            initial_capital = st.number_input("Initial ($)", value=100000, min_value=1000, step=10000, format="%d")

        st.write("")  # Small spacer

        # Execution Costs - compact
        st.caption("💸 **Execution Costs**")
        col1, col2, col3 = st.columns(3)
        with col1:
            spread = st.number_input("Spread %", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        with col2:
            slippage_pct = st.number_input("Slippage %", value=0.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        with col3:
            commission_per_share = st.number_input("Commission", value=0.0, min_value=0.0, step=0.001, format="%.3f")

        st.write("")  # Small spacer

        # Testing options
        use_mock_data = st.checkbox("Use Mock Data", value=False, help="Use mock data for testing")

        # Compact save button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submitted = st.form_submit_button("Save Configuration", type="primary", width="stretch")

        if submitted:
            # Store in session state
            st.session_state.config = {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'symbols': symbols,
                'spread': spread / 100,  # Convert to decimal
                'slippage_pct': slippage_pct / 100,
                'commission_per_share': commission_per_share,
                'use_mock_data': use_mock_data
            }
            st.success("✅ Configuration saved! Go to 'Strategy' tab to select and configure strategy.")


def render_strategy_sheet():
    """Sheet 2: Strategy Selection and Parameters"""
    st.subheader("📈 Strategy Selection")

    if 'config' not in st.session_state:
        st.warning("⚠️ Please configure backtest parameters first in the 'Configuration' tab.")
        return

    # Get available strategies (cache in session state to avoid repeated module inspection)
    if 'available_strategies' not in st.session_state:
        st.session_state.available_strategies = get_available_strategies()
    strategies = st.session_state.available_strategies

    if not strategies:
        st.error("No strategies found! Make sure strategies are properly defined in the strategies/ folder.")
        return

    # Strategy Selection Mode - NEW
    st.caption("**Selection Mode**")
    selection_mode = st.radio(
        "Selection Mode",
        ["Manual Selection", "Load from Saved Book"],
        horizontal=True,
        label_visibility="collapsed",
        help="Choose to configure strategy manually or load from a saved book"
    )

    st.write("")  # Spacer

    # Initialize variables
    selected_strategy_name = None
    selected_strategy = None
    selected_module = None
    strategy_params = {}
    loaded_from_book = False

    # Load from Book Mode - NEW
    if selection_mode == "Load from Saved Book":
        # Use absolute path to saved_books in project root
        books_dir = str(project_root / "saved_books")
        available_books = list_available_books(books_dir)

        if not available_books:
            st.info("📚 No saved books found. Run a backtest and save it as a book in the Results tab.")
            st.caption("💡 Books allow you to save strategy configurations for reuse and portfolio construction.")
            return

        # Book selection
        st.caption("**📚 Select Book**")
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_book_name = st.selectbox(
                "Select Book",
                available_books,
                label_visibility="collapsed",
                help="Choose a saved book to load"
            )

        if selected_book_name:
            try:
                book = load_book_cached(books_dir, selected_book_name)

                # Display book info
                st.caption("**Book Information:**")
                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.write(f"**Name:** {book.name}")
                    st.write(f"**Strategy:** {book.strategy_module}.{book.strategy_name}")
                    st.write(f"**Symbols:** {len(book.symbols)} symbols")

                with info_col2:
                    if book.description:
                        st.write(f"**Description:** {book.description}")
                    if book.tags:
                        st.write(f"**Tags:** {', '.join(book.tags)}")
                    st.write(f"**Created:** {book.created_at[:10]}")

                # Show and edit symbols
                with st.expander("📋 Book Symbols (Editable)", expanded=False):
                    st.caption("Edit the symbol list below. Changes will be saved to the book.")

                    # Create editable symbol list
                    symbols_text = st.text_area(
                        "Symbols (comma-separated)",
                        value=", ".join(book.symbols),
                        height=100,
                        help="Add or remove symbols. Separate with commas.",
                        key=f"book_symbols_{selected_book_name}"
                    )

                    # Parse edited symbols
                    edited_symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]

                    # Check if symbols changed
                    symbols_changed = set(edited_symbols) != set(book.symbols)

                    # Show comparison if changed
                    if symbols_changed:
                        st.write("")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption("**Original:**")
                            st.write(f"{len(book.symbols)} symbols")
                            st.code(", ".join(book.symbols), language=None)
                        with col2:
                            st.caption("**New:**")
                            st.write(f"{len(edited_symbols)} symbols")
                            st.code(", ".join(edited_symbols), language=None)

                    # Update button (only enabled if changed)
                    st.write("")
                    update_col1, update_col2 = st.columns([1, 2])
                    with update_col1:
                        update_button = st.button(
                            "💾 Update Book Symbols",
                            disabled=not symbols_changed,
                            type="primary" if symbols_changed else "secondary",
                            width="stretch",
                            key=f"update_book_symbols_{selected_book_name}"
                        )

                    if update_button and symbols_changed:
                        try:
                            # Update book with new symbols
                            book.symbols = edited_symbols
                            manager = get_book_manager(books_dir)
                            manager.save_book(book, overwrite=True)
                            st.success(f"✅ Book '{book.name}' updated with {len(edited_symbols)} symbols!")
                            # Clear cache to reflect updated book
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to update book: {str(e)}")

                # Load strategy from book
                selected_strategy_name = book.strategy_name
                selected_module = book.strategy_module.lower()

                if selected_strategy_name in strategies:
                    selected_strategy = strategies[selected_strategy_name]
                    strategy_params = book.strategy_params.copy()
                    loaded_from_book = True

                    # Override config symbols with book symbols (use edited symbols if available)
                    final_symbols = edited_symbols if 'symbols_text' in locals() else book.symbols
                    st.session_state.config['symbols'] = final_symbols

                    st.success(f"✅ Loaded book '{book.name}' with {len(final_symbols)} symbols!")
                    st.caption("⚠️ **Note:** Dates are still from your Configuration tab. Symbols are now from the book.")
                else:
                    st.error(f"❌ Strategy '{selected_strategy_name}' not found. Make sure it exists in the strategies folder.")
                    return

            except Exception as e:
                st.error(f"❌ Failed to load book: {str(e)}")
                st.exception(e)
                return

    # Manual Selection Mode
    else:
        # Strategy selection - two-tier system to avoid dropdown overflow
        # Group strategies by module
        strategies_by_module = {}
        for name, info in strategies.items():
            module = info['module']
            if module not in strategies_by_module:
                strategies_by_module[module] = []
            strategies_by_module[module].append(name)

        # Sort modules and put benchmark first
        module_order = ['benchmark', 'momentum', 'aqr']
        available_modules = [m for m in module_order if m in strategies_by_module]

        # Step 1: Category selection using radio buttons
        st.caption("**Category**")
        selected_module = st.radio(
            "Category",
            available_modules,
            format_func=lambda x: x.upper(),
            horizontal=True,
            label_visibility="collapsed"
        )

        # Step 2: Strategy selection from chosen category
        st.caption("**Strategy**")
        strategies_in_module = sorted(strategies_by_module[selected_module])

        # Use narrower column to constrain selectbox and dropdown
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_strategy_name = st.selectbox(
                "Strategy",
                strategies_in_module,
                label_visibility="collapsed",
                help=f"Select a {selected_module} strategy"
            )

        selected_strategy = strategies[selected_strategy_name]

    # Show strategy description inline
    st.caption(f"ℹ️ {selected_strategy['description']}")

    # Show strategy documentation - compact
    with st.expander("📖 Full Documentation", expanded=False):
        st.caption(f"**Module:** `strategies.{selected_strategy['module']}` | **Function:** `{selected_strategy_name}`")
        if selected_strategy['docstring']:
            st.code(selected_strategy['docstring'], language='text')

    # Extract and render parameters - compact
    st.write("")  # Small spacer
    if loaded_from_book:
        st.caption("**⚙️ Parameters (Loaded from Book - Editable)**")
    else:
        st.caption("**⚙️ Parameters**")

    strategy_func = selected_strategy['function']
    params_spec = extract_strategy_params(strategy_func)

    if not params_spec:
        st.info("No configurable parameters")
        if not loaded_from_book:
            strategy_params = {}
    else:
        # If not loaded from book, initialize empty dict
        if not loaded_from_book:
            strategy_params = {}

        # Group parameters by type
        int_params = {k: v for k, v in params_spec.items() if v['type'] == 'int'}
        float_params = {k: v for k, v in params_spec.items() if v['type'] == 'float'}
        bool_params = {k: v for k, v in params_spec.items() if v['type'] == 'bool'}

        # Render all parameters in compact grid
        all_params = list(int_params.items()) + list(float_params.items()) + list(bool_params.items())

        if all_params:
            cols = st.columns(4)  # 4-column grid
            for idx, (param_name, param_info) in enumerate(all_params):
                with cols[idx % 4]:
                    # Use book value if loaded, otherwise use default from param_info
                    if loaded_from_book and param_name in strategy_params:
                        default_value = strategy_params[param_name]
                    else:
                        default_value = param_info['default'] if param_info['default'] is not None else (20 if param_info['type'] == 'int' else 0.1 if param_info['type'] == 'float' else False)

                    if param_info['type'] == 'int':
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=int(default_value),
                            min_value=1, step=1,
                            key=f"param_{param_name}_{'book' if loaded_from_book else 'manual'}"
                        )
                    elif param_info['type'] == 'float':
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=float(default_value),
                            min_value=0.0, step=0.0001, format="%.4f",
                            key=f"param_{param_name}_{'book' if loaded_from_book else 'manual'}"
                        )
                    elif param_info['type'] == 'bool':
                        strategy_params[param_name] = st.checkbox(
                            param_name.replace('_', ' ').title(),
                            value=bool(default_value),
                            key=f"param_{param_name}_{'book' if loaded_from_book else 'manual'}"
                        )

    # Run Backtest Button - compact
    st.write("")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        run_button = st.button("🚀 Run Backtest", type="primary", width="stretch")

    if run_button:
        # Clear previous results to prevent memory buildup
        if 'backtest_result' in st.session_state:
            del st.session_state.backtest_result
        if 'backtest_config' in st.session_state:
            del st.session_state.backtest_config
        if 'backtest_symbols' in st.session_state:
            del st.session_state.backtest_symbols

        # Store strategy selection
        st.session_state.selected_strategy_name = selected_strategy_name
        st.session_state.selected_strategy_func = strategy_func
        st.session_state.selected_strategy_module = selected_module.upper() if selected_module else 'UNKNOWN'
        st.session_state.strategy_params = strategy_params

        # Run the backtest
        with st.spinner("Running backtest..."):
            try:
                config_data = st.session_state.config

                # Create execution config
                execution_config = ExecutionConfig(
                    spread=config_data['spread'],
                    slippage_pct=config_data['slippage_pct'],
                    commission_per_share=config_data['commission_per_share']
                )

                # Create backtest config
                # Note: allow_short is always True at config level; strategies control shorting via their own parameters
                config = BacktestConfig(
                    start_date=config_data['start_date'].strftime('%Y-%m-%d'),
                    end_date=config_data['end_date'].strftime('%Y-%m-%d'),
                    initial_capital=config_data['initial_capital'],
                    allow_short=True,  # Always True - strategies control their own shorting behavior
                    execution=execution_config,
                    use_mock_data=config_data['use_mock_data'],
                    verbose=False
                )

                # Create and run backtester
                backtester = Backtester(config)
                result = backtester.run(
                    strategy=strategy_func,
                    universe=config_data['symbols'],
                    strategy_params=strategy_params
                )

                # Store results
                st.session_state.backtest_result = result
                st.session_state.backtest_config = config
                st.session_state.backtest_symbols = config_data['symbols']  # Store symbols for benchmark lookup

                st.success("✅ Backtest completed! Go to 'Results' tab to view analysis.")

            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                st.exception(e)


def create_monthly_heatmap(equity_curve, metric='returns', title='Monthly Heatmap', use_cache=True):
    """
    Create a monthly heatmap for a given metric

    Parameters:
    -----------
    equity_curve : pd.DataFrame
        Equity curve with datetime index
    metric : str
        Metric to display: 'returns', 'pnl', 'drawdown', 'volatility', 'sharpe_ratio'
    title : str
        Chart title
    use_cache : bool
        Whether to use cached calculations (default: True)

    Returns:
    --------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import calendar

    # Use cached calculation if enabled
    if use_cache:
        # Create a hash of the equity curve for cache invalidation
        equity_hash = str(hash(tuple(equity_curve['total_equity'].head(10)))) + str(hash(tuple(equity_curve['total_equity'].tail(10))))

        # Convert equity_curve to dict for caching (DataFrames aren't hashable)
        equity_dict = {
            'total_equity': equity_curve['total_equity'].tolist(),
            'total_pnl': equity_curve['total_pnl'].tolist(),
            'index': equity_curve.index.astype(str).tolist()
        }

        monthly = calculate_monthly_metric_series(equity_hash, equity_dict, metric)
    else:
        # Original calculation without caching
        if metric == 'returns':
            series = equity_curve['total_equity'].pct_change()
            agg_method = 'sum'
        elif metric == 'pnl':
            series = equity_curve['total_pnl']
            agg_method = 'sum'
        elif metric == 'drawdown':
            equity = equity_curve['total_equity']
            running_max = equity.expanding().max()
            series = (equity - running_max) / running_max
            agg_method = 'mean'
        elif metric == 'volatility':
            series = equity_curve['total_equity'].pct_change().rolling(20).std()
            agg_method = 'mean'
        elif metric == 'sharpe_ratio':
            returns = equity_curve['total_equity'].pct_change()
            series = (returns.rolling(20).mean() / returns.rolling(20).std()) * np.sqrt(252)
            agg_method = 'mean'
        else:
            series = equity_curve['total_equity'].pct_change()
            agg_method = 'sum'

        df = pd.DataFrame({
            'value': series,
            'year': series.index.year,
            'month': series.index.month
        })

        if agg_method == 'sum':
            monthly = df.groupby(['year', 'month'])['value'].sum()
        else:
            monthly = df.groupby(['year', 'month'])['value'].mean()

    # Pivot to heatmap format
    monthly_df = monthly.reset_index()
    pivot_table = monthly_df.pivot(index='year', columns='month', values='value')

    # Determine color map and formatting based on metric
    if metric == 'returns':
        fmt = '.1%'
        cmap = 'RdYlGn'
    elif metric == 'pnl':
        fmt = '.0f'
        cmap = 'RdYlGn'
    elif metric == 'drawdown':
        fmt = '.1%'
        cmap = 'RdYlGn_r'  # Reverse: red for large drawdowns
    elif metric == 'volatility':
        fmt = '.1%'
        cmap = 'YlOrRd'
    elif metric == 'sharpe_ratio':
        fmt = '.2f'
        cmap = 'RdYlGn'
    else:
        fmt = '.1%'
        cmap = 'RdYlGn'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create heatmap
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=0 if metric in ['returns', 'pnl', 'drawdown', 'sharpe_ratio'] else None,
        cbar_kws={'label': metric.replace('_', ' ').title()},
        ax=ax
    )

    # Set labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    # Set month labels only for columns that exist in the pivot table
    existing_months = pivot_table.columns.tolist()
    month_labels = [calendar.month_abbr[int(m)] for m in existing_months]
    ax.set_xticklabels(month_labels, rotation=0)

    plt.tight_layout()
    return fig


def create_signal_analysis_charts(result, symbols, start_date, end_date):
    """
    Create interactive signal analysis charts with price, signals, and positions

    Parameters:
    -----------
    result : BacktestResult
        Backtest result object
    symbols : list
        List of symbols traded
    start_date : datetime
        Start date for the window
    end_date : datetime
        End date for the window

    Returns:
    --------
    tuple : (price_chart, position_chart, trades_df_filtered)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Filter trades to date window
    trades_df = result.trades.copy()

    # Ensure timezone compatibility
    if trades_df.index.tz is not None:
        # Make start_date and end_date timezone-aware to match trades_df
        if start_date.tz is None:
            start_date = start_date.tz_localize('UTC')
        if end_date.tz is None:
            end_date = end_date.tz_localize('UTC')
    else:
        # Make start_date and end_date timezone-naive to match trades_df
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)

    trades_df = trades_df[(trades_df.index >= start_date) & (trades_df.index <= end_date)]

    # Filter equity curve to date window
    equity_window = result.equity_curve.copy()

    # Ensure timezone compatibility for equity curve
    if equity_window.index.tz is not None:
        # Already handled above for start_date/end_date
        pass
    else:
        # Make start_date and end_date timezone-naive if needed
        start_date_naive = start_date.tz_localize(None) if start_date.tz is not None else start_date
        end_date_naive = end_date.tz_localize(None) if end_date.tz is not None else end_date
        start_date = start_date_naive
        end_date = end_date_naive

    equity_window = equity_window[(equity_window.index >= start_date) & (equity_window.index <= end_date)]

    if trades_df.empty:
        return None, None, None

    # Get unique symbols from trades
    traded_symbols = trades_df['symbol'].unique()

    # Load price data for the window
    try:
        from backt.data.loaders import YahooDataLoader
        loader = YahooDataLoader()

        price_data = {}
        for symbol in traded_symbols:
            data = loader.load(
                [symbol],
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if data is not None:
                if isinstance(data, dict) and symbol in data:
                    price_data[symbol] = data[symbol]
                elif isinstance(data, pd.DataFrame):
                    price_data[symbol] = data
    except Exception as e:
        st.warning(f"Could not load price data: {str(e)}")
        return None, None, None

    # Create subplots: Price charts + Portfolio charts (Total Value + Cumulative PnL)
    n_symbols = len(traded_symbols)
    n_portfolio_rows = 2  # Total Value + Cumulative PnL
    total_rows = n_symbols + n_portfolio_rows

    # Calculate row heights: price charts get 60%, portfolio charts get 40%
    price_height = 0.6 / n_symbols
    portfolio_height = 0.2  # Each portfolio chart gets 20%

    fig = make_subplots(
        rows=total_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            [f"{sym} - Price & Signals" for sym in traded_symbols] +
            ["Portfolio Total Value", "Cumulative PnL by Symbol"]
        ),
        row_heights=[price_height]*n_symbols + [portfolio_height, portfolio_height]
    )

    # Color scheme for buy/sell
    buy_color = '#2ca02c'
    sell_color = '#d62728'

    # Plot each symbol
    for idx, symbol in enumerate(traded_symbols, 1):
        if symbol not in price_data:
            continue

        prices = price_data[symbol]

        # Add price line
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices['close'],
                mode='lines',
                name=f'{symbol} Price',
                line=dict(color='#1f77b4', width=1.5),
                showlegend=(idx == 1)
            ),
            row=idx, col=1
        )

        # Get trades for this symbol
        symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()

        # Add buy signals
        buys = symbol_trades[symbol_trades['side'] == 'buy']
        if not buys.empty:
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys['price'],
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color=buy_color,
                        line=dict(width=1, color='white')
                    ),
                    showlegend=(idx == 1)
                ),
                row=idx, col=1
            )

        # Add sell signals
        sells = symbol_trades[symbol_trades['side'] == 'sell']
        if not sells.empty:
            fig.add_trace(
                go.Scatter(
                    x=sells.index,
                    y=sells['price'],
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color=sell_color,
                        line=dict(width=1, color='white')
                    ),
                    showlegend=(idx == 1)
                ),
                row=idx, col=1
            )

        # Update y-axis label
        fig.update_yaxes(title_text="Price ($)", row=idx, col=1)

    # Add portfolio equity curve (first portfolio chart)
    portfolio_value_row = n_symbols + 1
    fig.add_trace(
        go.Scatter(
            x=equity_window.index,
            y=equity_window['total_equity'],
            mode='lines',
            name='Total Portfolio Value',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)'
        ),
        row=portfolio_value_row, col=1
    )

    # Add cumulative PnL by symbol (second portfolio chart)
    cumulative_pnl_row = n_symbols + 2
    if result.per_symbol_equity_curves:
        # Generate distinct colors for each symbol
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for idx, (symbol, symbol_equity) in enumerate(result.per_symbol_equity_curves.items()):
            if symbol not in traded_symbols:
                continue

            # Calculate cumulative PnL for this symbol
            # Symbol equity curve contains the equity attributed to this symbol
            symbol_equity_window = symbol_equity[
                (symbol_equity.index >= pd.Timestamp(start_date)) &
                (symbol_equity.index <= pd.Timestamp(end_date))
            ]

            if not symbol_equity_window.empty and 'total_equity' in symbol_equity_window.columns:
                # Cumulative PnL = current equity - initial capital allocated to this symbol
                initial_value = symbol_equity_window['total_equity'].iloc[0]
                cumulative_pnl = symbol_equity_window['total_equity'] - initial_value

                color = colors[idx % len(colors)]
                fig.add_trace(
                    go.Scatter(
                        x=symbol_equity_window.index,
                        y=cumulative_pnl,
                        mode='lines',
                        name=f'{symbol} PnL',
                        line=dict(color=color, width=1.5),
                        stackgroup='one',  # Stack the PnLs
                        fillcolor=color
                    ),
                    row=cumulative_pnl_row, col=1
                )
    else:
        # Fallback: Calculate cumulative PnL from trades
        for idx, symbol in enumerate(traded_symbols):
            symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
            if symbol_trades.empty:
                continue

            # Calculate cumulative PnL from trade history
            symbol_trades['pnl'] = 0.0
            for i in range(len(symbol_trades)):
                trade = symbol_trades.iloc[i]
                if trade['side'] == 'sell':
                    # Simplified PnL calculation: value of sell - value of corresponding buy
                    # This is approximate - actual PnL tracking would need position tracking
                    symbol_trades.iloc[i, symbol_trades.columns.get_loc('pnl')] = trade['value']

            symbol_trades['cumulative_pnl'] = symbol_trades['pnl'].cumsum()

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=symbol_trades.index,
                    y=symbol_trades['cumulative_pnl'],
                    mode='lines',
                    name=f'{symbol} PnL',
                    line=dict(color=color, width=1.5)
                ),
                row=cumulative_pnl_row, col=1
            )

    # Update layout
    fig.update_layout(
        height=200 * (n_symbols + n_portfolio_rows),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=20, t=80, b=40)
    )

    # Update x-axis for bottom subplot only
    fig.update_xaxes(title_text="Date", row=total_rows, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Portfolio Value ($)", row=portfolio_value_row, col=1)
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=cumulative_pnl_row, col=1)

    return fig, trades_df


def render_signal_analysis_section():
    """Render the interactive signal analysis section"""
    if 'backtest_result' not in st.session_state:
        return

    result = st.session_state.backtest_result
    config = st.session_state.backtest_config
    symbols = st.session_state.get('backtest_symbols', ['SPY'])

    st.write("")
    st.caption("**📡 Signal Analysis & Visualization**")

    # Check if we have results
    if result.equity_curve.empty:
        st.warning("No backtest data available for signal analysis.")
        return

    if result.trades.empty:
        st.info("No trades executed during backtest - nothing to analyze.")
        return

    # Get full date range from backtest
    full_start = result.equity_curve.index[0]
    full_end = result.equity_curve.index[-1]

    # Initialize session state for date window
    # Always validate that dates are within the current data range
    if 'signal_window_start' not in st.session_state:
        st.session_state.signal_window_start = full_start.date()
    else:
        # Clamp to valid range if outside
        if st.session_state.signal_window_start < full_start.date():
            st.session_state.signal_window_start = full_start.date()
        elif st.session_state.signal_window_start > full_end.date():
            st.session_state.signal_window_start = full_end.date()

    if 'signal_window_end' not in st.session_state:
        st.session_state.signal_window_end = full_end.date()
    else:
        # Clamp to valid range if outside
        if st.session_state.signal_window_end < full_start.date():
            st.session_state.signal_window_end = full_start.date()
        elif st.session_state.signal_window_end > full_end.date():
            st.session_state.signal_window_end = full_end.date()

    # Quick window presets - BEFORE date pickers so they set values first
    st.caption("**Quick Windows**")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Last Month", width="stretch", key="btn_last_month"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=30)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col2:
        if st.button("Last 3 Months", width="stretch", key="btn_last_3m"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=90)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col3:
        if st.button("Last 6 Months", width="stretch", key="btn_last_6m"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=180)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col4:
        if st.button("Last Year", width="stretch", key="btn_last_year"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=365)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col5:
        if st.button("Full Range", width="stretch", key="btn_full_range"):
            st.session_state.signal_window_start = full_start.date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    st.write("")

    # Date window selection - AFTER quick buttons
    st.caption("**Or Select Custom Time Window**")

    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        window_start = st.date_input(
            "Window Start",
            value=st.session_state.signal_window_start,
            min_value=full_start.date(),
            max_value=full_end.date(),
            key="signal_start_picker"
        )

    with col2:
        window_end = st.date_input(
            "Window End",
            value=st.session_state.signal_window_end,
            min_value=full_start.date(),
            max_value=full_end.date(),
            key="signal_end_picker"
        )

    with col3:
        if st.button("Reset to Full Range", width="stretch", key="btn_reset"):
            st.session_state.signal_window_start = full_start.date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    # Only update session state if date pickers changed (not from button click)
    # This prevents overwriting button-set values
    if window_start != st.session_state.signal_window_start or window_end != st.session_state.signal_window_end:
        st.session_state.signal_window_start = window_start
        st.session_state.signal_window_end = window_end

    # Use session state as source of truth
    start_ts = pd.Timestamp(st.session_state.signal_window_start)
    end_ts = pd.Timestamp(st.session_state.signal_window_end)

    # Create charts
    st.write("")
    with st.spinner("Loading signal analysis..."):
        fig, trades_window = create_signal_analysis_charts(
            result, symbols, start_ts, end_ts
        )

    if fig is not None:
        # Display the chart
        st.plotly_chart(fig, width="stretch")

        # Trade statistics for the window
        if trades_window is not None and not trades_window.empty:
            st.write("")
            st.caption("**Trade Statistics for Selected Window**")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Trades", len(trades_window))

            with col2:
                buys = len(trades_window[trades_window['side'] == 'buy'])
                st.metric("Buy Signals", buys)

            with col3:
                sells = len(trades_window[trades_window['side'] == 'sell'])
                st.metric("Sell Signals", sells)

            with col4:
                symbols_traded = trades_window['symbol'].nunique()
                st.metric("Symbols Traded", symbols_traded)

            with col5:
                avg_trade_value = trades_window['value'].mean()
                st.metric("Avg Trade Size", f"${avg_trade_value:,.0f}")

            # Detailed trades table for window
            with st.expander("📋 Trades in Selected Window", expanded=False):
                display_trades = trades_window.reset_index()
                display_trades['timestamp'] = display_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                # Select columns to display, including reason if available
                base_cols = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'commission']
                if 'meta_reason' in display_trades.columns:
                    display_cols = base_cols + ['meta_reason']
                else:
                    display_cols = base_cols

                st.dataframe(
                    display_trades[display_cols],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        'meta_reason': st.column_config.TextColumn(
                            'Reason',
                            width='large',
                            help='Why this trade was triggered'
                        )
                    }
                )
    else:
        st.info("No trades found in the selected window. Try a different date range.")


def render_results_sheet():
    """Sheet 3: Results and Analysis"""
    st.subheader("📊 Backtest Results")

    if 'backtest_result' not in st.session_state:
        st.info("No backtest results yet. Please run a backtest first in the 'Strategy' tab.")
        return

    result = st.session_state.backtest_result
    config = st.session_state.backtest_config

    # Check if backtest produced results
    if result.equity_curve.empty:
        st.error("❌ Backtest failed to produce results. The equity curve is empty.")
        st.info("**Possible causes:**")
        st.write("- Strategy produced no orders")
        st.write("- All orders were rejected by risk management")
        st.write("- Data loading failed")
        st.write("- Strategy parameters are invalid")

        if hasattr(result, 'trades') and not result.trades.empty:
            st.write(f"- Found {len(result.trades)} trades but no equity curve")

        return

    # Create performance report
    report = PerformanceReport(result, initial_capital=config.initial_capital)

    # Summary Section - compact
    st.write("")  # Small spacer
    st.caption("**📈 Performance Summary**")

    metrics_df = report.get_metrics_dataframe(transpose=True)
    strategy_metrics = metrics_df['Strategy']

    # Key Metrics Row 1
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = strategy_metrics.get('total_return', 0)
        st.metric(
            "Total Return",
            f"{total_return:.2%}",
            delta=f"{total_return:.2%}"
        )

    with col2:
        sharpe = strategy_metrics.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")

    with col3:
        max_dd = strategy_metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd:.2%}")

    with col4:
        st.metric("Total Trades", len(result.trades))

    # Key Metrics Row 2
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cagr = strategy_metrics.get('cagr', 0)
        st.metric("CAGR", f"{cagr:.2%}")

    with col2:
        volatility = strategy_metrics.get('annualized_volatility', 0)
        st.metric("Volatility", f"{volatility:.2%}")

    with col3:
        sortino = strategy_metrics.get('sortino_ratio', 0)
        st.metric("Sortino Ratio", f"{sortino:.3f}")

    with col4:
        win_rate = strategy_metrics.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}")

    # Charts Section - compact
    st.write("")  # Small spacer
    st.caption("**📈 Performance Charts**")

    # Generate all charts from report
    try:
        charts = report.generate_charts(return_fig=True, show_plots=False)

        if charts and isinstance(charts, list):
            for fig in charts:
                if fig is not None:
                    st.pyplot(fig)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
        elif charts:
            st.pyplot(charts)
            import matplotlib.pyplot as plt
            plt.close(charts)
    except Exception as e:
        st.warning(f"Could not generate some charts: {str(e)}")

    # Signal Analysis Section - NEW INTERACTIVE MODULE
    st.write("")  # Small spacer
    st.markdown("---")  # Divider
    render_signal_analysis_section()

    # Correlation Matrix Section - show if multiple symbols
    if result.returns_correlation_matrix is not None and not result.returns_correlation_matrix.empty:
        st.write("")  # Small spacer
        st.markdown("---")  # Divider
        st.caption("**🔗 Returns Correlation Matrix**")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Compact, elegant sizing based on number of symbols
            n_symbols = len(result.returns_correlation_matrix)
            size = min(6, max(4, n_symbols * 0.8))  # Scale with symbols, cap at 6

            fig, ax = plt.subplots(figsize=(size, size * 0.85))
            sns.heatmap(
                result.returns_correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation', 'shrink': 0.8},
                annot_kws={'size': 9},
                ax=ax
            )
            ax.set_title('Returns Correlation Matrix', fontsize=11, fontweight='bold', pad=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not generate correlation heatmap: {str(e)}")

    # Monthly Heatmap Comparison Section - compact
    st.write("")  # Small spacer
    st.markdown("---")  # Divider
    st.caption("**📅 Monthly Heatmap Comparison**")

    # Metric selector - compact
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_metric = st.selectbox(
            "Metric",
            options=['returns', 'sharpe_ratio', 'pnl', 'drawdown', 'volatility'],
            format_func=lambda x: {
                'returns': 'Returns (%)',
                'sharpe_ratio': 'Sharpe Ratio',
                'pnl': 'Profit/Loss ($)',
                'drawdown': 'Drawdown (%)',
                'volatility': 'Volatility (%)'
            }[x],
            key='heatmap_metric_selector',
        label_visibility="collapsed"
    )

    # Get benchmark data if available
    try:
        metrics_df = report.get_metrics_dataframe(transpose=True)
        has_benchmark = 'SPY (Buy & Hold)' in metrics_df.columns or len(metrics_df.columns) > 1
    except:
        has_benchmark = False

    # Create heatmaps - compact
    col1, col2 = st.columns(2)

    with col1:
        st.caption("**Strategy**")
        try:
            strategy_heatmap = create_monthly_heatmap(
                result.equity_curve,
                metric=selected_metric,
                title=f'Strategy - Monthly {selected_metric.capitalize()}',
                use_cache=True
            )
            st.pyplot(strategy_heatmap)
            import matplotlib.pyplot as plt
            plt.close(strategy_heatmap)
        except Exception as e:
            st.error(f"Could not generate strategy heatmap: {str(e)}")

    with col2:
        st.caption("**Benchmark (SPY)**")
        try:
            # Load SPY benchmark data from Yahoo Finance (cached)
            benchmark_df, error_msg = load_spy_benchmark_data_from_backtest(
                result,
                config.initial_capital
            )

            if benchmark_df is not None:
                # Generate benchmark heatmap with caching
                benchmark_heatmap = create_monthly_heatmap(
                    benchmark_df,
                    metric=selected_metric,
                    title=f'SPY Benchmark - Monthly {selected_metric.capitalize()}',
                    use_cache=True
                )
                st.pyplot(benchmark_heatmap)
                import matplotlib.pyplot as plt
                plt.close(benchmark_heatmap)
            else:
                st.info(f"SPY benchmark data not available. {error_msg if error_msg else 'Please check your internet connection.'}")

        except Exception as e:
            st.warning(f"Could not generate benchmark heatmap: {str(e)}")

    # Per-Symbol Analysis - compact
    st.write("")  # Small spacer
    st.caption("**🎯 Per-Symbol Performance**")

    per_symbol_df = report.get_per_symbol_metrics_dataframe(transpose=True)
    if per_symbol_df is not None and not per_symbol_df.empty:
        # Display key per-symbol metrics
        display_cols = ['total_pnl', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        available_cols = [col for col in display_cols if col in per_symbol_df.columns]

        if available_cols:
            st.dataframe(
                per_symbol_df[available_cols].style.format({
                    'total_pnl': '${:,.2f}',
                    'sharpe_ratio': '{:.3f}',
                    'max_drawdown': '{:.2%}',
                    'win_rate': '{:.1%}'
                }),
                width="stretch"
            )

        # Per-symbol PnL chart
        try:
            import matplotlib.pyplot as plt
            fig_per_symbol = plt.figure()
            report.plot_per_symbol_pnl(show=False)
            st.pyplot(fig_per_symbol)
            plt.close(fig_per_symbol)
        except Exception as e:
            st.info(f"Per-symbol PnL chart not available: {str(e)}")

        # Correlation heatmap
        corr_matrix = report.get_correlation_matrix()
        if corr_matrix is not None and not corr_matrix.empty:
            try:
                import matplotlib.pyplot as plt
                fig_corr = plt.figure()
                report.plot_correlation_heatmap(show=False)
                st.pyplot(fig_corr)
                plt.close(fig_corr)
            except Exception as e:
                st.info(f"Correlation heatmap not available: {str(e)}")

    # Full Metrics Table - compact
    with st.expander("📊 Detailed Metrics", expanded=False):
        st.dataframe(metrics_df, width="stretch")

    # Trade History (Detailed) - compact
    with st.expander("💼 Trade History", expanded=False):
        if not result.trades.empty:
            trades_df = result.trades.copy()

            # Add helpful info - compact
            st.caption(f"**Total Fills:** {len(trades_df)} (each buy/sell = 1 fill)")

            # Trade statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                buys = len(trades_df[trades_df['side'] == 'buy'])
                st.metric("Buy Fills", buys)
            with col2:
                sells = len(trades_df[trades_df['side'] == 'sell'])
                st.metric("Sell Fills", sells)
            with col3:
                symbols = trades_df['symbol'].nunique()
                st.metric("Symbols Traded", symbols)
            with col4:
                avg_size = trades_df['value'].mean()
                st.metric("Avg Fill Size", f"${avg_size:,.0f}")

            # Per-symbol breakdown - compact
            st.caption("**Fills by Symbol:**")
            symbol_trades = trades_df.groupby('symbol').agg({
                'quantity': 'sum',
                'value': 'sum',
                'commission': 'sum'
            }).round(2)
            symbol_trades['num_fills'] = trades_df.groupby('symbol').size()
            symbol_trades = symbol_trades.sort_values('num_fills', ascending=False)
            st.dataframe(symbol_trades, width="stretch")

            # Full trade log - compact
            st.caption("**Full Trade Log:**")

            # Prepare display dataframe
            display_df = trades_df.reset_index()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            # Reorder columns for better readability
            base_columns = ['timestamp', 'symbol', 'side', 'quantity', 'price', 'value', 'commission']
            meta_columns = [col for col in display_df.columns if col.startswith('meta_')]
            other_columns = [col for col in display_df.columns if col not in base_columns + meta_columns]

            ordered_columns = base_columns + meta_columns + other_columns
            ordered_columns = [col for col in ordered_columns if col in display_df.columns]

            display_df = display_df[ordered_columns]

            # Show trades with pagination
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                rows_per_page = st.selectbox("Rows per page", [50, 100, 250, 500], index=1, key="trades_rows")
            with col2:
                total_pages = (len(display_df) - 1) // rows_per_page + 1
                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, key="trades_page")

            start_idx = (page - 1) * rows_per_page
            end_idx = start_idx + rows_per_page

            # Configure column display with better formatting for metadata
            column_config = {}
            if 'meta_reason' in display_df.columns:
                column_config['meta_reason'] = st.column_config.TextColumn(
                    'Trade Reason',
                    width='large',
                    help='Why this trade was triggered'
                )
            if 'meta_signal' in display_df.columns:
                column_config['meta_signal'] = st.column_config.TextColumn(
                    'Signal',
                    width='medium',
                    help='Strategy signal type'
                )

            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                width="stretch",
                hide_index=True,
                column_config=column_config
            )
            st.caption(f"Showing fills {start_idx + 1}-{min(end_idx, len(display_df))} of {len(display_df)} total fills")

            # Download button
            csv = trades_df.to_csv()
            st.download_button(
                label="📥 Download All Trades as CSV",
                data=csv,
                file_name="backtest_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("No trades executed")

    # Book Saving Section - Quick Save
    st.write("")  # Small spacer
    st.markdown("---")  # Divider
    st.subheader("📚 Save as Book")

    # Check if we have all required session state
    if 'selected_strategy_name' not in st.session_state or 'strategy_params' not in st.session_state:
        st.warning("Strategy information not available. Please run a backtest first.")
    else:
        # Quick save section - always visible
        st.caption("Save this configuration for later reuse. You can load it in the Strategy tab.")

        col1, col2 = st.columns([3, 1])

        with col1:
            # What will be saved info
            st.caption(f"**Will save:** {st.session_state.selected_strategy_name} • {len(st.session_state.strategy_params)} params • {len(st.session_state.config['symbols'])} symbols")

            # Name input
            quick_book_name = st.text_input(
                "Book Name",
                placeholder="e.g., Momentum_Tech_LongOnly",
                help="Enter a unique name for this book",
                key="quick_book_name_input",
                label_visibility="collapsed"
            )

        with col2:
            st.write("")  # Spacing to align with input
            st.write("")  # More spacing
            save_button = st.button(
                "💾 Save Book",
                type="primary",
                width="stretch",
                key="quick_save_book_button"
            )

        # Advanced options (optional)
        with st.expander("⚙️ Advanced Options (Description & Tags)", expanded=False):
            adv_col1, adv_col2 = st.columns([2, 1])

            with adv_col1:
                book_description = st.text_area(
                    "Description (optional)",
                    placeholder="e.g., Fast momentum strategy for tech stocks during bull markets",
                    help="Optional description of this book's purpose and characteristics",
                    height=80,
                    key="book_description_input"
                )

                book_tags = st.text_input(
                    "Tags (optional, comma-separated)",
                    placeholder="e.g., momentum, long-only, tech",
                    help="Optional tags for categorization",
                    key="book_tags_input"
                )

            with adv_col2:
                st.caption("**What's included:**")
                st.write(f"✓ Strategy: {st.session_state.selected_strategy_name}")
                st.write(f"✓ Parameters: {len(st.session_state.strategy_params)}")
                st.write(f"✓ Symbols: {len(st.session_state.config['symbols'])}")
                st.write("")
                st.caption("❌ **Not included:**")
                st.write("• Date ranges")

        # Handle save button click
        if save_button:
            if not quick_book_name:
                st.error("⚠️ Please enter a book name.")
            else:
                try:
                    from backt.utils.books import BookManager, create_book_from_session

                    # Parse tags (from advanced options if expanded)
                    tags_input = st.session_state.get('book_tags_input', '')
                    tags_list = [tag.strip() for tag in tags_input.split(',')] if tags_input else []

                    # Get description (from advanced options if expanded)
                    description_input = st.session_state.get('book_description_input', '')

                    # Get strategy module from strategy name
                    strategy_module = st.session_state.get('selected_strategy_module', 'UNKNOWN')

                    # Create book
                    book = create_book_from_session(
                        name=quick_book_name,
                        strategy_module=strategy_module,
                        strategy_name=st.session_state.selected_strategy_name,
                        strategy_params=st.session_state.strategy_params,
                        symbols=st.session_state.config['symbols'],
                        description=description_input,
                        tags=tags_list
                    )

                    # Save book
                    books_dir = str(project_root / "saved_books")
                    manager = get_book_manager(books_dir)

                    # Check if book already exists
                    if manager.book_exists(quick_book_name):
                        st.error(f"❌ Book '{quick_book_name}' already exists!")
                        st.info("💡 Choose a different name or delete the existing book first.")
                        st.caption(f"Existing book location: {manager.books_dir / f'{quick_book_name}.json'}")
                    else:
                        manager.save_book(book)
                        # Clear cache to reflect new book
                        st.cache_data.clear()
                        st.success(f"✅ Book '{quick_book_name}' saved successfully!")
                        st.info(f"📂 Saved to: {manager.books_dir / f'{quick_book_name}.json'}")
                        st.caption("💡 Load this book in the Strategy tab under 'Load from Saved Book'")
                        st.balloons()

                except Exception as e:
                    st.error(f"❌ Failed to save book: {str(e)}")
                    st.exception(e)




# ==== CPCV Validation Functions ====

def render_cpcv_validation_sheet():
    """Sheet 4: CPCV Validation - Professional Strategy Validation"""
    st.subheader("🔬 CPCV Validation")
    st.caption("**Combinatorial Purged Cross-Validation** - Detect overfitting and validate strategy robustness")

    if 'config' not in st.session_state or 'selected_strategy_name' not in st.session_state:
        st.warning("Please configure backtest parameters and select a strategy first.")
        st.info("👈 Go to 'Configuration' and 'Strategy' tabs to set up your backtest.")
        return

    # CPCV Mode Selection
    st.write("")
    mode = st.radio(
        "Validation Mode",
        ["Single Strategy Validation", "Parameter Grid Optimization", "Strategy Comparison"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.write("")  # Spacer

    if mode == "Single Strategy Validation":
        render_single_strategy_cpcv()
    elif mode == "Parameter Grid Optimization":
        render_parameter_optimization_cpcv()
    else:
        render_strategy_comparison_cpcv()


def render_single_strategy_cpcv():
    """Validate a single strategy with CPCV"""
    st.caption("**Single Strategy Validation**")
    st.write("Validate one strategy configuration across multiple train/test paths to detect overfitting.")

    # CPCV Configuration - Use form to batch inputs and prevent page rerun on every change
    st.write("")
    st.caption("**CPCV Settings**")

    # Use form to batch all inputs together - only reruns when submitted
    with st.form(key="cpcv_settings_form"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_splits = st.number_input("Number of Folds", value=10, min_value=3, max_value=20, step=1)
        with col2:
            n_test_splits = st.number_input("Test Folds per Path", value=2, min_value=1, max_value=5, step=1)
        with col3:
            purge_pct = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0, step=1.0) / 100
        with col4:
            embargo_pct = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0, step=1.0) / 100

        # Calculate number of paths (only when form is submitted)
        import math
        n_paths = math.comb(n_splits, n_test_splits)
        st.caption(f"This will generate **{n_paths} validation paths** (C({n_splits},{n_test_splits}))")

        # Submit button for form
        st.write("")
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            form_submitted = st.form_submit_button("⚙️ Update Settings", width="stretch")

    # Store settings in session state when form is submitted
    if form_submitted or 'cpcv_n_splits' not in st.session_state:
        st.session_state.cpcv_n_splits = n_splits
        st.session_state.cpcv_n_test_splits = n_test_splits
        st.session_state.cpcv_purge_pct = purge_pct
        st.session_state.cpcv_embargo_pct = embargo_pct
        st.session_state.cpcv_n_paths = n_paths

    # Display current settings
    if 'cpcv_n_splits' in st.session_state:
        st.info(f"**Current Settings:** {st.session_state.cpcv_n_splits} folds, {st.session_state.cpcv_n_test_splits} test folds/path, {st.session_state.cpcv_purge_pct*100:.1f}% purge, {st.session_state.cpcv_embargo_pct*100:.1f}% embargo → **{st.session_state.cpcv_n_paths} paths**")

    # Run Button (separate from form to avoid conflicts)
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_validation = st.button("🚀 Run CPCV Validation", type="primary", width="stretch")

    # Use session state values for validation
    if 'cpcv_n_splits' in st.session_state:
        n_splits = st.session_state.cpcv_n_splits
        n_test_splits = st.session_state.cpcv_n_test_splits
        purge_pct = st.session_state.cpcv_purge_pct
        embargo_pct = st.session_state.cpcv_embargo_pct
        n_paths = st.session_state.cpcv_n_paths

    if run_validation:
        with st.spinner(f"Running CPCV validation across {n_paths} paths... This may take a few minutes."):
            # Get configuration from session
            config_dict = st.session_state.config
            strategy_name = st.session_state.selected_strategy_name
            strategy_params = st.session_state.get('strategy_params', {})

            # Convert config dict to BacktestConfig object
            from backt.utils.config import ExecutionConfig

            execution_config = ExecutionConfig(
                spread=config_dict.get('spread', 0.0),
                slippage_pct=config_dict.get('slippage_pct', 0.0),
                commission_per_share=config_dict.get('commission_per_share', 0.0)
            )

            config = BacktestConfig(
                start_date=config_dict['start_date'].strftime('%Y-%m-%d'),
                end_date=config_dict['end_date'].strftime('%Y-%m-%d'),
                initial_capital=config_dict['initial_capital'],
                allow_short=True,  # Always True - strategies control their own shorting behavior
                use_mock_data=config_dict.get('use_mock_data', False),
                execution=execution_config,
                verbose=False
            )

            symbols = config_dict.get('symbols', ['SPY'])

            # Get strategy function (cache in session state to avoid repeated calls)
            if 'available_strategies' not in st.session_state:
                st.session_state.available_strategies = get_available_strategies()
            strategies = st.session_state.available_strategies
            strategy_func = strategies[strategy_name]['function']

            # Create CPCV config (with parallel processing and numba JIT enabled)
            cpcv_config = CPCVConfig(
                n_splits=n_splits,
                n_test_splits=n_test_splits,
                purge_pct=purge_pct,
                embargo_pct=embargo_pct,
                n_jobs=-1,  # Use all available CPU cores for parallel processing
                use_numba=True  # Enable numba JIT compilation for 2-5x additional speedup
            )

            # Run CPCV validation
            try:
                validator = CPCVValidator(config, cpcv_config)
                result = validator.validate(
                    strategy=strategy_func,
                    symbols=symbols,
                    strategy_params=strategy_params
                )

                # Store result in session
                st.session_state.cpcv_result = result

                st.success(f"Validation complete! Ran {result.n_paths} paths in {result.total_runtime_seconds:.1f}s")

            except Exception as e:
                st.error(f"CPCV validation failed: {str(e)}")
                st.exception(e)
                return

    # Display results if available
    if 'cpcv_result' in st.session_state:
        display_cpcv_results(st.session_state.cpcv_result)


def display_cpcv_results(result):
    """Display comprehensive CPCV validation results"""
    st.write("")
    st.caption("**📊 Validation Results**")

    # Summary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Paths Completed", f"{result.n_paths}")
    with col2:
        st.metric("Mean Sharpe", f"{result.mean_sharpe:.3f}")
    with col3:
        st.metric("Std Sharpe", f"{result.std_sharpe:.3f}")
    with col4:
        st.metric("Mean Return", f"{result.mean_return:.1%}")
    with col5:
        st.metric("Mean Max DD", f"{result.mean_max_drawdown:.1%}")

    st.write("")

    # Overfitting Metrics - Highlight Box
    st.caption("**🎯 Overfitting Detection Metrics**")

    metrics = result.overfitting_metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pbo_color = "green" if metrics.pbo < 0.3 else ("orange" if metrics.pbo < 0.5 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{pbo_color}' == 'green' else rgba(255,165,0,0.1) if '{pbo_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {pbo_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>PBO</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.pbo:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        dsr_color = "green" if metrics.dsr > 2.0 else ("orange" if metrics.dsr > 1.0 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{dsr_color}' == 'green' else rgba(255,165,0,0.1) if '{dsr_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {dsr_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>DSR</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.dsr:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        deg_color = "green" if abs(metrics.degradation_pct) < 10 else ("orange" if abs(metrics.degradation_pct) < 20 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{deg_color}' == 'green' else rgba(255,165,0,0.1) if '{deg_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {deg_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>Degradation</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.degradation_pct:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        stab_color = "green" if metrics.sharpe_stability > 5.0 else ("orange" if metrics.sharpe_stability > 2.0 else "red")
        st.markdown(f"""
        <div style='background-color: rgba(0,255,0,0.1) if '{stab_color}' == 'green' else rgba(255,165,0,0.1) if '{stab_color}' == 'orange' else rgba(255,0,0,0.1);
                    padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid {stab_color};'>
            <p style='margin: 0; font-size: 0.75rem; color: #666;'>Stability</p>
            <p style='margin: 0; font-size: 1.5rem; font-weight: bold;'>{metrics.sharpe_stability:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # Interpretations
    st.write("")
    with st.expander("📖 Metric Interpretations", expanded=True):
        for metric, interpretation in result.overfitting_interpretations.items():
            st.write(f"• **{metric.upper()}:** {interpretation}")

    # Validation Status
    st.write("")
    if result.passes_validation():
        st.success("✅ **Strategy PASSES validation criteria**")
    else:
        st.error("❌ **Strategy FAILS validation criteria**")
        if result.validation_warnings:
            for warning in result.validation_warnings:
                st.warning(f"⚠️ {warning}")

    # Visualization Section
    st.write("")
    st.caption("**📈 Validation Path Analysis**")

    # Create path distribution chart
    fig = create_path_distribution_chart(result)
    st.plotly_chart(fig, width="stretch")

    # Performance distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = create_sharpe_distribution_chart(result)
        st.plotly_chart(fig_dist, width="stretch")

    with col2:
        fig_scatter = create_path_scatter_chart(result)
        st.plotly_chart(fig_scatter, width="stretch")

    # Detailed path results table
    st.write("")
    with st.expander("🔍 Detailed Path Results", expanded=False):
        path_data = []
        for path in result.path_results:
            path_data.append({
                'Path ID': path.path_id,
                'Test Folds': str(path.test_fold_indices),
                'Sharpe Ratio': f"{path.sharpe_ratio:.3f}",
                'Return': f"{path.total_return:.2%}",
                'Max Drawdown': f"{path.max_drawdown:.2%}"
            })

        path_df = pd.DataFrame(path_data)
        st.dataframe(path_df, width="stretch", hide_index=True)


def create_path_distribution_chart(result):
    """Create chart showing Sharpe ratio across all validation paths"""
    sharpe_values = [p.sharpe_ratio for p in result.path_results]
    path_ids = [p.path_id for p in result.path_results]

    fig = go.Figure()

    # Bar chart of Sharpe ratios
    fig.add_trace(go.Bar(
        x=path_ids,
        y=sharpe_values,
        name='Sharpe Ratio',
        marker=dict(
            color=sharpe_values,
            colorscale='RdYlGn',
            cmin=-2,
            cmax=2,
            colorbar=dict(title="Sharpe")
        )
    ))

    # Add mean line
    fig.add_hline(
        y=result.mean_sharpe,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Mean: {result.mean_sharpe:.3f}"
    )

    fig.update_layout(
        title="Sharpe Ratio Across All Validation Paths",
        xaxis_title="Path ID",
        yaxis_title="Sharpe Ratio",
        height=350,
        hovermode='x',
        showlegend=False
    )

    return fig


def create_sharpe_distribution_chart(result):
    """Create histogram of Sharpe ratio distribution"""
    sharpe_values = [p.sharpe_ratio for p in result.path_results]

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=sharpe_values,
        nbinsx=20,
        name='Distribution',
        marker=dict(color='steelblue')
    ))

    # Add mean line
    fig.add_vline(
        x=result.mean_sharpe,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {result.mean_sharpe:.3f}"
    )

    fig.update_layout(
        title="Sharpe Ratio Distribution",
        xaxis_title="Sharpe Ratio",
        yaxis_title="Frequency",
        height=300,
        showlegend=False
    )

    return fig


def create_path_scatter_chart(result):
    """Create scatter plot of Return vs Drawdown"""
    returns = [p.total_return * 100 for p in result.path_results]
    drawdowns = [abs(p.max_drawdown * 100) for p in result.path_results]
    sharpe = [p.sharpe_ratio for p in result.path_results]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdowns,
        y=returns,
        mode='markers',
        marker=dict(
            size=10,
            color=sharpe,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Sharpe"),
            cmin=-2,
            cmax=2
        ),
        text=[f"Path {p.path_id}" for p in result.path_results],
        hovertemplate='<b>%{text}</b><br>Return: %{y:.1f}%<br>Max DD: %{x:.1f}%'
    ))

    fig.update_layout(
        title="Return vs Max Drawdown (colored by Sharpe)",
        xaxis_title="Max Drawdown (%)",
        yaxis_title="Total Return (%)",
        height=300
    )

    return fig


def render_parameter_optimization_cpcv():
    """Parameter grid optimization with CPCV validation"""
    st.caption("**Parameter Grid Optimization with CPCV**")
    st.write("Optimize strategy parameters using parallel search, then validate top candidates with CPCV.")

    # Get configuration from session
    if 'config' not in st.session_state or 'selected_strategy_name' not in st.session_state:
        st.warning("Please configure backtest parameters and select a strategy first.")
        return

    config = st.session_state.config
    strategy_name = st.session_state.selected_strategy_name

    # Get strategy function (cache in session state to avoid repeated calls)
    if 'available_strategies' not in st.session_state:
        st.session_state.available_strategies = get_available_strategies()
    strategies = st.session_state.available_strategies
    strategy_func = strategies[strategy_name]['function']

    st.write("")
    st.caption("**Step 1: Choose Optimization Method**")

    col1, col2 = st.columns(2)
    with col1:
        optimization_method = st.selectbox(
            "Optimization Algorithm",
            ["Grid Search (Exhaustive)", "FLAML (Intelligent)"],
            help="Grid Search tests all combinations. FLAML intelligently explores the space (10x faster)."
        )

    with col2:
        optimization_metric = st.selectbox(
            "Optimization Metric",
            ["sharpe_ratio", "sortino_ratio", "total_return", "calmar_ratio"],
            help="Metric to maximize during parameter search"
        )

    st.write("")
    st.caption("**Step 2: Define Parameter Space**")

    # Extract parameters from strategy
    strategy_params = extract_strategy_params(strategy_func)

    if strategy_params:
        st.write("**Strategy Parameters Detected:**")

        # Show available parameters as buttons
        param_buttons = []
        cols = st.columns(min(len(strategy_params), 4))
        for idx, (param_name, param_info) in enumerate(strategy_params.items()):
            with cols[idx % len(cols)]:
                default_val = param_info.get('default', 'N/A')
                if st.button(f"➕ {param_name}", key=f"add_param_{param_name}",
                            help=f"Type: {param_info.get('type', 'unknown')}, Default: {default_val}"):
                    if 'param_definitions' not in st.session_state:
                        st.session_state.param_definitions = []
                    if param_name not in [p['name'] for p in st.session_state.param_definitions]:
                        # Suggest ranges based on parameter type and default
                        suggested_min = 5
                        suggested_max = 50
                        suggested_step = 5

                        if default_val and default_val != 'N/A':
                            try:
                                default_num = float(default_val)
                                if param_info.get('type') == 'int':
                                    suggested_min = max(1, int(default_num * 0.5))
                                    suggested_max = int(default_num * 2)
                                    suggested_step = max(1, int(default_num * 0.1))
                                elif param_info.get('type') == 'float':
                                    suggested_min = round(default_num * 0.5, 4)
                                    suggested_max = round(default_num * 2, 4)
                                    suggested_step = round(default_num * 0.1, 4)
                            except:
                                pass

                        st.session_state.param_definitions.append({
                            'name': param_name,
                            'type': param_info.get('type', 'int'),
                            'default': default_val,
                            'min': suggested_min,
                            'max': suggested_max,
                            'step': suggested_step
                        })
                        st.rerun()

    st.write("")
    st.write("**Or add custom parameter:**")

    if 'param_definitions' not in st.session_state:
        st.session_state.param_definitions = []

    col1, col2 = st.columns([3, 1])
    with col1:
        new_param_name = st.text_input("Parameter Name", key="new_param", placeholder="e.g., custom_param")
    with col2:
        if st.button("➕ Add Custom"):
            if new_param_name and new_param_name not in [p['name'] for p in st.session_state.param_definitions]:
                st.session_state.param_definitions.append({
                    'name': new_param_name,
                    'type': 'int',
                    'min': 5,
                    'max': 50,
                    'step': 5
                })
                st.rerun()

    # Display parameter configurations - Use form to batch inputs and prevent page rerun
    param_grid = {}
    if st.session_state.param_definitions:
        st.write("")
        st.write("**Configure Parameter Ranges:**")

        # Wrap parameter range inputs in form to prevent reruns
        with st.form(key="param_ranges_form"):
            param_ranges_data = []

            for idx, param_def in enumerate(st.session_state.param_definitions):
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])

                with col1:
                    param_type = param_def.get('type', 'int')
                    default_val = param_def.get('default', 'N/A')
                    st.write(f"**{param_def['name']}**")
                    if default_val != 'N/A':
                        st.caption(f"Default: {default_val}")

                # Determine step format based on type
                is_float = param_type == 'float'

                if is_float:
                    # Float parameters
                    min_default = float(param_def.get('min', 0.01))
                    max_default = float(param_def.get('max', 1.0))
                    step_default = float(param_def.get('step', 0.01))
                    step_format = 0.0001
                    step_min_value = 0.00001
                else:
                    # Integer parameters
                    min_default = int(param_def.get('min', 5))
                    max_default = int(param_def.get('max', 50))
                    step_default = int(param_def.get('step', 5))
                    step_format = 1
                    step_min_value = 1

                with col2:
                    min_val = st.number_input(
                        f"Min",
                        key=f"min_{idx}",
                        value=min_default,
                        step=step_format,
                        format="%.5f" if is_float else "%d"
                    )

                with col3:
                    max_val = st.number_input(
                        f"Max",
                        key=f"max_{idx}",
                        value=max_default,
                        step=step_format,
                        format="%.5f" if is_float else "%d"
                    )

                with col4:
                    step_val = st.number_input(
                        f"Step",
                        key=f"step_{idx}",
                        value=step_default,
                        step=step_format,
                        min_value=step_min_value,
                        format="%.5f" if is_float else "%d"
                    )

                with col5:
                    # Show number of values
                    if step_val > 0:
                        n_values = int((max_val - min_val) / step_val) + 1
                        st.caption(f"({n_values} values)")
                    else:
                        st.caption("⚠️")

                # Store data for later processing
                param_ranges_data.append({
                    'name': param_def['name'],
                    'type': param_type,
                    'min': min_val,
                    'max': max_val,
                    'step': step_val,
                    'is_float': is_float
                })

            # CPCV settings also in the same form
            st.write("")
            st.caption("**CPCV Validation Settings**")

            col1, col2 = st.columns(2)
            with col1:
                top_k = st.number_input("Top K to Validate", value=3, min_value=1, max_value=10,
                                       help="Number of best parameter sets to validate with CPCV")
            with col2:
                n_splits = st.number_input("CPCV Folds", value=10, min_value=3, max_value=20)

            col1, col2, col3 = st.columns(3)
            with col1:
                n_test_splits = st.number_input("Test Folds", value=2, min_value=1, max_value=5)
            with col2:
                purge_pct_input = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0)
            with col3:
                embargo_pct_input = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0)

            # Submit button for form
            st.write("")
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                param_form_submitted = st.form_submit_button("⚙️ Update Parameters", width="stretch")

        # Process form data after submission
        if param_form_submitted or 'param_grid_cache' not in st.session_state:
            # Build parameter grid from form data
            for param_data in param_ranges_data:
                if param_data['is_float']:
                    import numpy as np
                    param_grid[param_data['name']] = list(np.arange(
                        param_data['min'],
                        param_data['max'] + param_data['step'],
                        param_data['step']
                    ))
                else:
                    param_grid[param_data['name']] = list(range(
                        int(param_data['min']),
                        int(param_data['max']) + 1,
                        int(param_data['step'])
                    ))

            # Store in session state
            st.session_state.param_grid_cache = param_grid
            st.session_state.opt_top_k = top_k
            st.session_state.opt_n_splits = n_splits
            st.session_state.opt_n_test_splits = n_test_splits
            st.session_state.opt_purge_pct = purge_pct_input / 100
            st.session_state.opt_embargo_pct = embargo_pct_input / 100
        else:
            # Use cached values
            param_grid = st.session_state.param_grid_cache
            top_k = st.session_state.opt_top_k
            n_splits = st.session_state.opt_n_splits
            n_test_splits = st.session_state.opt_n_test_splits
            purge_pct = st.session_state.opt_purge_pct
            embargo_pct = st.session_state.opt_embargo_pct

        # Delete buttons (outside form to allow immediate rerun)
        st.write("")
        st.write("**Manage Parameters:**")
        delete_cols = st.columns(len(st.session_state.param_definitions))
        for idx, param_def in enumerate(st.session_state.param_definitions):
            with delete_cols[idx]:
                if st.button(f"🗑️ {param_def['name']}", key=f"del_{idx}"):
                    st.session_state.param_definitions.pop(idx)
                    if 'param_grid_cache' in st.session_state:
                        del st.session_state.param_grid_cache
                    st.rerun()

        # Show parameter space size
        if param_grid:
            total_combinations = 1
            for values in param_grid.values():
                total_combinations *= len(values)

            st.write("")
            if optimization_method == "Grid Search (Exhaustive)":
                st.info(f"📊 **{total_combinations} total combinations** will be tested (exhaustive)")
            else:
                est_evals = min(total_combinations, 100)
                st.info(f"🧠 **~{est_evals} combinations** will be intelligently sampled from {total_combinations} possible (FLAML)")

    # Run Button (separate from form)
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_optimization = st.button("🚀 Run Optimization", type="primary", width="stretch")

    if run_optimization and param_grid:
        from backt.optimization.optimizer import StrategyOptimizer
        from backt.optimization.flaml_optimizer import FLAMLOptimizer
        from backt.validation.cpcv_validator import CPCVConfig
        from backt import BacktestConfig

        try:
            # Convert config dict to BacktestConfig object if needed
            if isinstance(config, dict):
                from backt.utils.config import ExecutionConfig

                # Extract symbols separately (not a BacktestConfig parameter)
                symbols = config.get('symbols', ['SPY'])

                # Extract execution-related parameters
                execution_config = ExecutionConfig(
                    spread=config.get('spread', 0.01),
                    slippage_pct=config.get('slippage_pct', 0.0005),
                    commission_per_share=config.get('commission_per_share', 0.001)
                )

                # Create BacktestConfig with only valid parameters
                backtest_config = BacktestConfig(
                    start_date=config['start_date'],
                    end_date=config['end_date'],
                    initial_capital=config.get('initial_capital', 100000.0),
                    allow_short=True,  # Always True - strategies control their own shorting behavior
                    execution=execution_config,
                    verbose=False  # Suppress backtest logs during optimization
                )
            else:
                backtest_config = config
                symbols = backtest_config.universe if hasattr(backtest_config, 'universe') else ['SPY']

            # Show optimization method
            with st.spinner(f"Running {optimization_method}..."):
                # Create CPCV config
                cpcv_config = CPCVConfig(
                    n_splits=n_splits,
                    n_test_splits=n_test_splits,
                    purge_pct=purge_pct,
                    embargo_pct=embargo_pct,
                    n_jobs=-1,  # Use all CPU cores
                    use_numba=True
                )

                # Select optimizer based on user choice
                if optimization_method == "FLAML (Intelligent)":
                    # Use FLAML for intelligent parameter search
                    from flaml import tune

                    # Convert param_grid to FLAML param_space format
                    param_space = {}
                    for param_name, values in param_grid.items():
                        if isinstance(values, range):
                            # Convert range to randint
                            param_space[param_name] = {
                                'domain': tune.randint(values.start, values.stop)
                            }
                        elif isinstance(values, (list, tuple)):
                            # Convert list to choice
                            param_space[param_name] = {
                                'domain': tune.choice(list(values))
                            }
                        else:
                            # Single value - use uniform around it
                            param_space[param_name] = {
                                'domain': tune.choice([values])
                            }

                    optimizer = FLAMLOptimizer(
                        strategy_function=strategy_func,
                        config=backtest_config,
                        symbols=symbols
                    )

                    # Run FLAML optimization with CPCV
                    result = optimizer.optimize_with_cpcv(
                        param_space=param_space,  # Use param_space not param_grid
                        optimization_metric=optimization_metric,
                        top_k=top_k,
                        cpcv_config=cpcv_config,
                        time_budget_s=300,  # 5 minute budget for FLAML search
                        num_samples=100,  # Max 100 evaluations
                        verbose=0  # Suppress FLAML verbose output
                    )
                else:
                    # Use Grid Search for exhaustive search
                    optimizer = StrategyOptimizer(
                        strategy_function=strategy_func,
                        config=backtest_config,
                        symbols=symbols
                    )

                    # Run optimization with CPCV
                    result = optimizer.optimize_with_cpcv(
                        param_grid=param_grid,
                        optimization_metric=optimization_metric,
                        n_jobs=-1,  # Parallel processing
                        top_k=top_k,
                        cpcv_config=cpcv_config,
                        verbose=False
                    )

                # Store results in session
                st.session_state.optimization_result = result

            st.success(f"✅ Optimization complete! Tested {result.total_combinations} combinations in {result.execution_time:.1f}s")

            # Display results
            st.write("")
            st.caption("**Optimization Results**")

            # Top parameters
            st.write("**Best Parameters:**")
            st.json(result.best_params)

            st.write(f"**Best {optimization_metric}:** {result.best_metric_value:.4f}")

            # Top K results
            st.write("")
            st.caption(f"**Top {top_k} Parameter Sets**")

            # Handle both Grid Search (DataFrame) and FLAML (list) result formats
            if hasattr(result.all_results, 'head'):
                # Grid Search returns DataFrame
                top_df = result.all_results.head(top_k).copy()
            else:
                # FLAML returns list of ParameterSetResult - convert to DataFrame
                top_df = result.to_dataframe().head(top_k).copy()
            param_cols = [col for col in top_df.columns if col.startswith('param_')]
            metric_cols = [optimization_metric, 'total_return', 'max_drawdown', 'sharpe_ratio']
            metric_cols = [col for col in metric_cols if col in top_df.columns]
            display_cols = param_cols + metric_cols

            st.dataframe(
                top_df[display_cols].style.format({
                    col: "{:.4f}" for col in metric_cols
                }),
                width="stretch"
            )

            # CPCV Validation Results
            if hasattr(result, 'cpcv_results') and result.cpcv_results:
                st.write("")
                st.caption("**CPCV Validation Results**")

                cpcv_data = []
                for idx, cpcv_item in enumerate(result.cpcv_results, 1):
                    cpcv_result = cpcv_item['cpcv_result']
                    cpcv_data.append({
                        'Rank': idx,
                        'Parameters': str(cpcv_item['params']),
                        'PBO': cpcv_result.overfitting_metrics.pbo,
                        'DSR': cpcv_result.overfitting_metrics.dsr,
                        'Degradation %': cpcv_result.overfitting_metrics.degradation_pct,
                        'Validation': '✅ Pass' if cpcv_result.passes_validation() else '❌ Fail'
                    })

                cpcv_df = pd.DataFrame(cpcv_data)
                st.dataframe(
                    cpcv_df.style.format({
                        'PBO': '{:.3f}',
                        'DSR': '{:.3f}',
                        'Degradation %': '{:.1f}%'
                    }).applymap(
                        lambda x: 'background-color: #d4edda' if x == '✅ Pass' else 'background-color: #f8d7da',
                        subset=['Validation']
                    ),
                    width="stretch"
                )

                # Recommendation
                st.write("")
                st.caption("**Recommendation**")

                best_validated = None
                for cpcv_item in result.cpcv_results:
                    if cpcv_item['cpcv_result'].passes_validation():
                        best_validated = cpcv_item
                        break

                if best_validated:
                    st.success(f"✅ **Recommended parameters:** {best_validated['params']}")
                    st.write(f"   • PBO: {best_validated['cpcv_result'].overfitting_metrics.pbo:.3f} (< 0.50)")
                    st.write(f"   • DSR: {best_validated['cpcv_result'].overfitting_metrics.dsr:.3f} (> 1.0)")
                    st.write(f"   • Degradation: {best_validated['cpcv_result'].overfitting_metrics.degradation_pct:.1f}% (< 30%)")
                else:
                    st.warning("⚠️ None of the top parameters passed CPCV validation. Strategy may be overfit.")

        except Exception as e:
            st.error(f"Optimization failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def render_strategy_comparison_cpcv():
    """Compare multiple strategies with CPCV"""
    st.caption("**Strategy Comparison**")
    st.write("Compare different strategies using CPCV validation.")

    st.info("🚧 Feature coming soon! This will allow you to compare multiple strategies.")
    st.write("**Planned features:**")
    st.write("• Select multiple strategies to compare")
    st.write("• Run CPCV on each strategy")
    st.write("• Side-by-side comparison of PBO, DSR, Sharpe")
    st.write("• Identify which strategy is most robust")


# ==== Book Manager Functions ====

def render_book_manager_sheet():
    """Sheet 5: Book Manager - Edit saved strategy configurations"""
    from backt.utils import BookEditor, Book
    import pandas as pd

    st.subheader("📚 Book Manager")
    st.caption("Manage your saved strategy configurations")

    # Initialize session state for book manager
    if 'bm_selected_book' not in st.session_state:
        st.session_state.bm_selected_book = None
    if 'bm_modified' not in st.session_state:
        st.session_state.bm_modified = False
    if 'bm_original' not in st.session_state:
        st.session_state.bm_original = None

    # Use absolute path to saved_books in project root
    books_dir = project_root / "saved_books"
    editor = BookEditor(books_dir=str(books_dir))
    books = editor.manager.get_all_books()

    if not books:
        st.info("📚 No saved books found. Create books by saving backtest results or using the ranking tool.")
        st.caption("💡 Books allow you to save strategy configurations for reuse")
        return

    # Book selection
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        book_names = [b.name for b in books]
        selected_name = st.selectbox(
            "Select Book",
            [""] + book_names,
            key="bm_book_selector"
        )

    with col2:
        # Filter options
        all_strategies = sorted(list(set([b.strategy_name for b in books])))
        filter_strategy = st.selectbox("Filter by Strategy", ["All"] + all_strategies, key="bm_filter_strategy")

    with col3:
        if st.button("🔄 Refresh", width="stretch"):
            st.rerun()

    if not selected_name:
        # Show overview
        st.divider()
        st.caption(f"**{len(books)} books available**")

        # Apply filter
        display_books = books
        if filter_strategy != "All":
            display_books = [b for b in books if b.strategy_name == filter_strategy]

        book_data = [{
            "Name": b.name,
            "Strategy": b.strategy_name,
            "Symbols": len(b.symbols),
            "Parameters": len(b.strategy_params),
            "Tags": ", ".join(b.tags[:3]) if b.tags else "-",
            "Updated": b.updated_at[:10]
        } for b in display_books]
        st.dataframe(pd.DataFrame(book_data), width="stretch", hide_index=True)
        return

    # Load book
    if st.session_state.bm_selected_book is None or st.session_state.bm_selected_book.name != selected_name:
        book = editor.load_book(selected_name)
        st.session_state.bm_selected_book = book
        st.session_state.bm_original = Book.from_dict(book.to_dict())
        st.session_state.bm_modified = False

    book = st.session_state.bm_selected_book
    original = st.session_state.bm_original

    # Status and controls
    st.divider()

    if st.session_state.bm_modified:
        st.warning("⚠️ Unsaved changes")
    else:
        st.success("✅ No unsaved changes")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Save", disabled=not st.session_state.bm_modified, type="primary", width="stretch"):
            editor.save_book(book, create_backup=True)
            st.session_state.bm_modified = False
            st.session_state.bm_original = Book.from_dict(book.to_dict())
            st.success(f"Saved '{book.name}' (backup created)")
            st.rerun()

    with col2:
        if st.button("↩️ Revert", disabled=not st.session_state.bm_modified, width="stretch"):
            st.session_state.bm_selected_book = Book.from_dict(original.to_dict())
            st.session_state.bm_modified = False
            st.rerun()

    with col3:
        if st.button("🔍 Validate", width="stretch"):
            is_valid, warnings = editor.validate_book(book)
            if is_valid:
                st.success("✅ Valid")
            else:
                for w in warnings:
                    st.warning(f"⚠️ {w}")

    with col4:
        if st.button("🗑️ Delete", width="stretch", type="secondary"):
            st.session_state.show_delete = True

    if st.session_state.get('show_delete', False):
        st.error(f"⚠️ Delete '{book.name}'?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Confirm", type="primary", width="stretch"):
                editor.manager.delete_book(book.name)
                st.session_state.bm_selected_book = None
                st.session_state.show_delete = False
                st.rerun()
        with c2:
            if st.button("❌ Cancel", width="stretch"):
                st.session_state.show_delete = False
                st.rerun()

    # Tabs for editing
    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Symbols", "⚙️ Parameters", "📝 Metadata"])

    with tab1:
        st.caption("**Edit Symbols**")
        symbols_text = st.text_area(
            "Symbol List (comma-separated)",
            value=", ".join(book.symbols),
            height=150,
            key="bm_symbols"
        )
        new_symbols = sorted(list(set([s.strip().upper() for s in symbols_text.split(',') if s.strip()])))

        if set(new_symbols) != set(book.symbols):
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"**Current:** {len(book.symbols)} symbols")
            with col2:
                st.caption(f"**New:** {len(new_symbols)} symbols")

            added = set(new_symbols) - set(book.symbols)
            removed = set(book.symbols) - set(new_symbols)
            if added:
                st.success(f"➕ {', '.join(sorted(added))}")
            if removed:
                st.error(f"➖ {', '.join(sorted(removed))}")

            if st.button("✅ Apply Symbol Changes", type="primary"):
                book.symbols = new_symbols
                st.session_state.bm_modified = True
                st.rerun()

    with tab2:
        st.caption("**Edit Parameters**")
        with st.form("bm_params"):
            updated_params = {}
            for key, value in sorted(book.strategy_params.items()):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text(key)
                with col2:
                    if isinstance(value, bool):
                        updated_params[key] = st.checkbox("", value=value, key=f"bm_p_{key}", label_visibility="collapsed")
                    elif isinstance(value, int):
                        updated_params[key] = st.number_input("", value=value, step=1, key=f"bm_p_{key}", label_visibility="collapsed")
                    elif isinstance(value, float):
                        updated_params[key] = st.number_input("", value=value, step=0.01, format="%.4f", key=f"bm_p_{key}", label_visibility="collapsed")
                    else:
                        updated_params[key] = st.text_input("", value=str(value), key=f"bm_p_{key}", label_visibility="collapsed")

            if st.form_submit_button("💾 Apply Parameter Changes", type="primary"):
                book.strategy_params = updated_params
                st.session_state.bm_modified = True
                st.rerun()

    with tab3:
        st.caption("**Edit Metadata**")
        new_desc = st.text_area("Description", value=book.description, height=100, key="bm_desc")
        if new_desc != book.description:
            if st.button("💾 Update Description"):
                book.description = new_desc
                st.session_state.bm_modified = True
                st.rerun()

        st.caption("**Tags (comma-separated)**")
        tags_text = st.text_input("", value=", ".join(book.tags), key="bm_tags", label_visibility="collapsed")
        new_tags = [t.strip() for t in tags_text.split(',') if t.strip()]
        if set(new_tags) != set(book.tags):
            if st.button("💾 Update Tags"):
                book.tags = new_tags
                st.session_state.bm_modified = True
                st.rerun()

        st.divider()
        st.caption(f"**Created:** {book.created_at[:19]}")
        st.caption(f"**Updated:** {book.updated_at[:19]}")


def main():
    """Main application"""

    # Display banner at the top
    banner_path = Path(__file__).parent.parent / "visuals" / "banner.png"
    if banner_path.exists():
        st.image(str(banner_path), use_column_width=True)

    # Define pages using Streamlit's native navigation (top position)
    pages = [
        st.Page(render_configuration_sheet, title="⚙️ Configuration"),
        st.Page(render_strategy_sheet, title="📈 Strategy"),
        st.Page(render_results_sheet, title="📊 Results"),
        st.Page(render_cpcv_validation_sheet, title="🔬 CPCV Validation"),
        st.Page(render_book_manager_sheet, title="📚 Book Manager"),
    ]

    # Create navigation at top
    pg = st.navigation(pages, position="top")
    pg.run()


if __name__ == "__main__":
    main()
