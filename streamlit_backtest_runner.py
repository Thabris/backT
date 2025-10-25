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
sys.path.insert(0, str(Path(__file__).parent))

import warnings
import logging

# Suppress Streamlit deprecation and worker warnings
warnings.filterwarnings('ignore', message='.*use_container_width.*')
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
        "VQT": "Volatility Hedge",
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
        "TCTL": "Tactical Allocation",
        "COM": "Commodity Trend",
        "WDTI": "WisdomTree Trend",
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
        "DGLD": "Gold 2x Bear",
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
    initial_sidebar_state="expanded"
)


# Custom CSS - Professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --bg-light: #f8f9fa;
        --bg-dark: #2c3e50;
        --text-muted: #6c757d;
    }

    /* Reduce overall padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1.2rem !important;
    }

    /* Header styling */
    h1 {
        font-size: 2.1rem !important;
        font-weight: 600 !important;
        color: #1f77b4 !important;
        margin-bottom: 0.6rem !important;
    }

    h2 {
        font-size: 1.4rem !important;
        font-weight: 500 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.9rem !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 500 !important;
        margin-top: 0.9rem !important;
        margin-bottom: 0.6rem !important;
    }

    /* Compact tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 0.6rem;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        padding: 0 18px !important;
        font-size: 0.92rem !important;
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4 !important;
        color: white !important;
    }

    /* Compact buttons */
    .stButton > button {
        font-size: 0.88rem !important;
        padding: 0.45rem 1.1rem !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f77b4 0%, #1557a0 100%) !important;
        box-shadow: 0 2px 8px rgba(31,119,180,0.3);
    }

    /* Metric cards - more professional */
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        color: #6c757d !important;
        font-weight: 500 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Compact forms */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input {
        font-size: 0.88rem !important;
        padding: 0.45rem 0.65rem !important;
    }

    /* Selectbox - fix text overflow */
    .stSelectbox > div > div {
        font-size: 0.88rem !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        min-width: 100% !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        min-height: 2.4rem !important;
    }

    /* Dropdown menu items - ensure text wraps and is readable */
    [data-baseweb="popover"] {
        max-width: 500px !important;
        width: auto !important;
        max-height: 400px !important;
    }

    [role="option"] {
        white-space: normal !important;
        word-wrap: break-word !important;
        min-height: 2.4rem !important;
        padding: 0.5rem 1rem !important;
        line-height: 1.4 !important;
    }

    [role="listbox"] {
        max-width: 500px !important;
        max-height: 400px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Ensure scrollbar is always visible on dropdown menus */
    [role="listbox"]::-webkit-scrollbar {
        width: 10px !important;
        display: block !important;
    }

    [role="listbox"]::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 5px !important;
    }

    [role="listbox"]::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 5px !important;
    }

    [role="listbox"]::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }

    /* Labels - smaller and muted */
    label {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #495057 !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.15rem !important;
        color: #2c3e50 !important;
    }

    /* Caption text - smaller */
    .stCaption {
        font-size: 0.8rem !important;
        color: #6c757d !important;
    }

    /* Compact expanders */
    .streamlit-expanderHeader {
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.1rem !important;
    }

    /* Success/Warning/Error messages - compact */
    .stSuccess, .stWarning, .stError, .stInfo {
        padding: 0.6rem 1.1rem !important;
        font-size: 0.88rem !important;
    }

    /* Divider - subtle */
    hr {
        margin: 1.1rem 0 !important;
        border-color: #e9ecef !important;
    }

    /* Dataframe styling */
    .dataframe {
        font-size: 0.8rem !important;
    }

    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.4rem !important;
    }

    /* Compact columns */
    [data-testid="column"] {
        padding: 0 0.4rem !important;
    }

    /* Compact forms */
    .stForm {
        border: none !important;
        padding: 0.6rem !important;
    }

    /* Compact captions */
    .stCaption {
        margin-bottom: 0.3rem !important;
        margin-top: 0.4rem !important;
    }

    /* Tighter input fields */
    .stTextInput input, .stNumberInput input, .stDateInput input {
        height: 2.4rem !important;
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


@st.cache_data(ttl=3600, show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


def extract_strategy_params(strategy_func):
    """
    Extract parameter names and defaults from strategy docstring
    Returns dict of {param_name: {'type': type, 'default': value, 'description': str}}
    """
    doc = inspect.getdoc(strategy_func)
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
            if st.button("Apply Preset", type="primary", use_container_width=True):
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
            if st.button("Update Symbols", type="primary", use_container_width=True):
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
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_symbols = []
            st.rerun()

    with col3:
        if st.button("Select All (Category)", use_container_width=True,
                    disabled=(selection_mode != "Browse Categories")):
            # Add all symbols from currently expanded category
            st.info("Expand a category and select ETFs individually")

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
        with col4:
            st.markdown("<p style='font-size: 0.8rem; color: transparent; margin-bottom: 0.3rem; margin-top: 0.4rem; line-height: 1.2;'>**.**</p>", unsafe_allow_html=True)
            allow_short = st.checkbox("Short", value=True, help="Allow short selling")

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
            submitted = st.form_submit_button("Save Configuration", type="primary", use_container_width=True)

        if submitted:
            # Store in session state
            st.session_state.config = {
                'start_date': start_date,
                'end_date': end_date,
                'initial_capital': initial_capital,
                'allow_short': allow_short,
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

    # Get available strategies
    strategies = get_available_strategies()

    if not strategies:
        st.error("No strategies found! Make sure strategies are properly defined in the strategies/ folder.")
        return

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
    col1, col2 = st.columns([1, 3])

    with col1:
        st.caption("**Category**")
        selected_module = st.radio(
            "Category",
            available_modules,
            format_func=lambda x: x.upper(),
            label_visibility="collapsed"
        )

    with col2:
        st.caption("**Strategy**")
        # Step 2: Strategy selection from chosen category
        strategies_in_module = sorted(strategies_by_module[selected_module])

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
    st.caption("**⚙️ Parameters**")

    strategy_func = selected_strategy['function']
    params_spec = extract_strategy_params(strategy_func)

    if not params_spec:
        st.info("No configurable parameters")
        strategy_params = {}
    else:
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
                    if param_info['type'] == 'int':
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=param_info['default'] if param_info['default'] is not None else 20,
                            min_value=1, step=1
                        )
                    elif param_info['type'] == 'float':
                        strategy_params[param_name] = st.number_input(
                            param_name.replace('_', ' ').title(),
                            value=param_info['default'] if param_info['default'] is not None else 0.1,
                            min_value=0.0, step=0.01, format="%.3f"
                        )
                    elif param_info['type'] == 'bool':
                        strategy_params[param_name] = st.checkbox(
                            param_name.replace('_', ' ').title(),
                            value=param_info['default'] if param_info['default'] is not None else False
                        )

    # Run Backtest Button - compact
    st.write("")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        run_button = st.button("🚀 Run Backtest", type="primary", use_container_width=True)

    if run_button:
        # Store strategy selection
        st.session_state.selected_strategy_name = selected_strategy_name
        st.session_state.selected_strategy_func = strategy_func
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
                config = BacktestConfig(
                    start_date=config_data['start_date'].strftime('%Y-%m-%d'),
                    end_date=config_data['end_date'].strftime('%Y-%m-%d'),
                    initial_capital=config_data['initial_capital'],
                    allow_short=config_data['allow_short'],
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

    # Create subplots: Price charts + Position chart
    n_symbols = len(traded_symbols)
    fig = make_subplots(
        rows=n_symbols + 1, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[f"{sym} - Price & Signals" for sym in traded_symbols] + ["Portfolio Position"],
        row_heights=[0.8/n_symbols]*n_symbols + [0.2]
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

    # Add portfolio equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_window.index,
            y=equity_window['total_equity'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#ff7f0e', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)'
        ),
        row=n_symbols + 1, col=1
    )

    # Update layout
    fig.update_layout(
        height=250 * (n_symbols + 1),
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
    fig.update_xaxes(title_text="Date", row=n_symbols + 1, col=1)

    # Update y-axis for portfolio
    fig.update_yaxes(title_text="Portfolio Value ($)", row=n_symbols + 1, col=1)

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
    if 'signal_window_start' not in st.session_state:
        st.session_state.signal_window_start = full_start.date()
    if 'signal_window_end' not in st.session_state:
        st.session_state.signal_window_end = full_end.date()

    # Quick window presets - BEFORE date pickers so they set values first
    st.caption("**Quick Windows**")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("Last Month", use_container_width=True, key="btn_last_month"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=30)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col2:
        if st.button("Last 3 Months", use_container_width=True, key="btn_last_3m"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=90)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col3:
        if st.button("Last 6 Months", use_container_width=True, key="btn_last_6m"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=180)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col4:
        if st.button("Last Year", use_container_width=True, key="btn_last_year"):
            st.session_state.signal_window_start = (full_end - pd.Timedelta(days=365)).date()
            st.session_state.signal_window_end = full_end.date()
            st.rerun()

    with col5:
        if st.button("Full Range", use_container_width=True, key="btn_full_range"):
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
        if st.button("Reset to Full Range", use_container_width=True, key="btn_reset"):
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
        st.plotly_chart(fig, use_container_width=True)

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
                    use_container_width=True,
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

    # Correlation Matrix Section - show if multiple symbols
    if result.returns_correlation_matrix is not None and not result.returns_correlation_matrix.empty:
        st.write("")  # Small spacer
        st.markdown("---")  # Divider
        st.caption("**🔗 Returns Correlation Matrix**")

        with st.expander("View Correlation Matrix", expanded=True):
            # Display as heatmap
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, ax = plt.subplots(figsize=(10, 8))
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
                    cbar_kws={'label': 'Correlation'},
                    ax=ax
                )
                ax.set_title('Returns Correlation Matrix', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"Could not generate correlation heatmap: {str(e)}")

            # Also display as table for exact values
            st.write("**Correlation Values:**")
            st.dataframe(
                result.returns_correlation_matrix.style.background_gradient(
                    cmap='coolwarm',
                    vmin=-1,
                    vmax=1
                ).format("{:.3f}"),
                use_container_width=True
            )

    # Signal Analysis Section - NEW INTERACTIVE MODULE
    st.write("")  # Small spacer
    st.markdown("---")  # Divider
    render_signal_analysis_section()

    # Monthly Heatmap Comparison Section - compact
    st.write("")  # Small spacer
    st.markdown("---")  # Divider
    st.caption("**📅 Monthly Heatmap Comparison**")

    # Metric selector - compact
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
                use_container_width=True
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
        st.dataframe(metrics_df, use_container_width=True)

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
            st.dataframe(symbol_trades, use_container_width=True)

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
            rows_per_page = st.selectbox("Rows per page", [50, 100, 250, 500], index=1, key="trades_rows")
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
                use_container_width=True,
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

    # CPCV Configuration
    st.write("")
    st.caption("**CPCV Settings**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_splits = st.number_input("Number of Folds", value=10, min_value=3, max_value=20, step=1)
    with col2:
        n_test_splits = st.number_input("Test Folds per Path", value=2, min_value=1, max_value=5, step=1)
    with col3:
        purge_pct = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0, step=1.0) / 100
    with col4:
        embargo_pct = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0, step=1.0) / 100

    # Calculate number of paths
    import math
    n_paths = math.comb(n_splits, n_test_splits)
    st.caption(f"This will generate **{n_paths} validation paths** (C({n_splits},{n_test_splits}))")

    # Run Button
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_validation = st.button("🚀 Run CPCV Validation", type="primary", use_container_width=True)

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
                allow_short=config_dict.get('allow_short', False),
                use_mock_data=config_dict.get('use_mock_data', False),
                execution=execution_config,
                verbose=False
            )

            symbols = config_dict.get('symbols', ['SPY'])

            # Get strategy function
            strategies = get_available_strategies()
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
    st.plotly_chart(fig, use_container_width=True)

    # Performance distribution
    col1, col2 = st.columns(2)
    with col1:
        fig_dist = create_sharpe_distribution_chart(result)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_scatter = create_path_scatter_chart(result)
        st.plotly_chart(fig_scatter, use_container_width=True)

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
        st.dataframe(path_df, use_container_width=True, hide_index=True)


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

    # Get strategy function
    strategies = get_available_strategies()
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

    # Display parameter configurations
    param_grid = {}
    if st.session_state.param_definitions:
        st.write("")
        st.write("**Configure Parameter Ranges:**")

        for idx, param_def in enumerate(st.session_state.param_definitions):
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1, 1, 1, 0.5])

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
                step_format = 0.001
            else:
                # Integer parameters
                min_default = int(param_def.get('min', 5))
                max_default = int(param_def.get('max', 50))
                step_default = int(param_def.get('step', 5))
                step_format = 1

            with col2:
                min_val = st.number_input(
                    f"Min",
                    key=f"min_{idx}",
                    value=min_default,
                    step=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col3:
                max_val = st.number_input(
                    f"Max",
                    key=f"max_{idx}",
                    value=max_default,
                    step=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col4:
                step_val = st.number_input(
                    f"Step",
                    key=f"step_{idx}",
                    value=step_default,
                    step=step_format,
                    min_value=step_format,
                    format="%.4f" if is_float else "%d"
                )

            with col5:
                # Show number of values
                if step_val > 0:
                    n_values = int((max_val - min_val) / step_val) + 1
                    st.caption(f"({n_values} values)")
                else:
                    st.caption("⚠️")

            with col6:
                if st.button("🗑️", key=f"del_{idx}"):
                    st.session_state.param_definitions.pop(idx)
                    st.rerun()

            # Build parameter grid
            if is_float:
                # For float parameters, create list with proper precision
                import numpy as np
                param_grid[param_def['name']] = list(np.arange(min_val, max_val + step_val, step_val))
            else:
                # For int parameters, use range
                param_grid[param_def['name']] = list(range(int(min_val), int(max_val) + 1, int(step_val)))

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

    st.write("")
    st.caption("**Step 3: CPCV Validation Settings**")

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
        purge_pct = st.number_input("Purge %", value=5.0, min_value=0.0, max_value=20.0) / 100
    with col3:
        embargo_pct = st.number_input("Embargo %", value=2.0, min_value=0.0, max_value=10.0) / 100

    # Run Button
    st.write("")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_optimization = st.button("🚀 Run Optimization", type="primary", use_container_width=True)

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
                    allow_short=config.get('allow_short', False),
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
                use_container_width=True
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
                    use_container_width=True
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


def main():
    """Main application"""

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Configuration",
        "📈 Strategy",
        "📊 Results",
        "🔬 CPCV Validation"
    ])

    with tab1:
        render_configuration_sheet()

    with tab2:
        render_strategy_sheet()

    with tab3:
        render_results_sheet()

    with tab4:
        render_cpcv_validation_sheet()

    # Sidebar - Status - compact
    with st.sidebar:
        # Logo/Header - compact
        st.markdown(
            '<div style="text-align: center; padding: 0.5rem 0 0.3rem 0;">'
            '<h2 style="margin: 0; color: #1f77b4; font-size: 1.2rem;">BackT</h2>'
            '<p style="margin: 0; font-size: 0.65rem; color: #6c757d;">Professional Backtesting</p>'
            '</div>',
            unsafe_allow_html=True
        )

        st.caption("**Session Status**")

        # Configuration status
        if 'config' in st.session_state:
            config = st.session_state.config
            st.markdown(
                '<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); '
                'padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                '<div style="font-size: 0.75rem; font-weight: 600; color: #155724;">✓ Configuration</div>'
                f'<div style="font-size: 0.7rem; color: #155724; margin-top: 0.2rem;">{config["start_date"]} → {config["end_date"]}</div>'
                f'<div style="font-size: 0.7rem; color: #155724;">${config["initial_capital"]:,.0f} | {len(config["symbols"])} symbols</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background: #fff3cd; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                '<div style="font-size: 0.75rem; font-weight: 600; color: #856404;">⚠ No Configuration</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # Strategy status
        if 'selected_strategy_name' in st.session_state:
            st.markdown(
                '<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); '
                'padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                '<div style="font-size: 0.75rem; font-weight: 600; color: #155724;">✓ Strategy</div>'
                f'<div style="font-size: 0.7rem; color: #155724; margin-top: 0.2rem;">{st.session_state.selected_strategy_name}</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background: #fff3cd; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                '<div style="font-size: 0.75rem; font-weight: 600; color: #856404;">⚠ No Strategy</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # Results status
        if 'backtest_result' in st.session_state:
            result = st.session_state.backtest_result

            # Check if equity curve has data
            if not result.equity_curve.empty and 'total_equity' in result.equity_curve.columns:
                final_value = result.equity_curve['total_equity'].iloc[-1]
                initial_value = result.equity_curve['total_equity'].iloc[0]
                total_return = (final_value / initial_value - 1) * 100
            else:
                # Backtest failed or no data
                total_return = 0
                final_value = 0
                initial_value = 0
            return_color = '#155724' if total_return >= 0 else '#721c24'
            bg_color = 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)' if total_return >= 0 else 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)'

            st.markdown(
                f'<div style="background: {bg_color}; '
                'padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                f'<div style="font-size: 0.75rem; font-weight: 600; color: {return_color};">✓ Results</div>'
                f'<div style="font-size: 0.7rem; color: {return_color}; margin-top: 0.2rem;">Return: {total_return:+.2f}%</div>'
                f'<div style="font-size: 0.7rem; color: {return_color};">{len(result.trades)} fills executed</div>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background: #fff3cd; padding: 0.6rem; border-radius: 6px; margin-bottom: 0.5rem;">'
                '<div style="font-size: 0.75rem; font-weight: 600; color: #856404;">⚠ No Results</div>'
                '</div>',
                unsafe_allow_html=True
            )

        # Reset button - compact
        st.write("")
        if st.button("🔄 Reset Session", use_container_width=True, help="Clear all session data"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
