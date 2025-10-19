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
from strategies import momentum


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
    .stSelectbox > div > div,
    .stDateInput > div > div > input {
        font-size: 0.88rem !important;
        padding: 0.45rem 0.65rem !important;
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

    # Get all strategy modules
    strategy_modules = {
        'momentum': momentum,
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


def render_configuration_sheet():
    """Sheet 1: Backtest Configuration"""
    st.subheader("⚙️ Backtest Configuration")

    with st.form("config_form"):
        # Date Range - compact with breathing room
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])
        with col1:
            st.caption("📅 **Dates**")
            start_date = st.date_input("Start", value=date(2020, 1, 1), max_value=date.today())
        with col2:
            st.markdown("<p style='font-size: 0.8rem; color: transparent; margin-bottom: 0.3rem; margin-top: 0.4rem; line-height: 1.2;'>**.**</p>", unsafe_allow_html=True)
            end_date = st.date_input("End", value=date(2023, 12, 31), max_value=date.today())
        with col3:
            st.caption("💰 **Capital**")
            initial_capital = st.number_input("Initial ($)", value=100000, min_value=1000, step=10000, format="%d")
        with col4:
            st.markdown("<p style='font-size: 0.8rem; color: transparent; margin-bottom: 0.3rem; margin-top: 0.4rem; line-height: 1.2;'>**.**</p>", unsafe_allow_html=True)
            allow_short = st.checkbox("Short", value=True, help="Allow short selling")

        st.write("")  # Small spacer

        # Trading Universe - compact
        st.caption("🌍 **Trading Universe**")
        universe_input = st.text_input("Symbols", value="SPY, QQQ, TLT, GLD", placeholder="SPY, QQQ, TLT, GLD", label_visibility="collapsed")
        symbols = [s.strip().upper() for s in universe_input.split(',') if s.strip()]

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

        # Risk Management - compact
        st.caption("🛡️ **Risk Management**")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            max_leverage = st.number_input("Max Leverage", value=2.0, min_value=1.0, max_value=10.0, step=0.5)
        with col2:
            max_position_size = st.number_input("Max Position", value=0.25, min_value=0.01, max_value=1.0, step=0.05, format="%.2f")
        with col3:
            use_mock_data = st.checkbox("Mock Data", value=False, help="Use mock data for testing")

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
                'max_leverage': max_leverage,
                'max_position_size': max_position_size,
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

    # Strategy selection - compact
    strategy_names = list(strategies.keys())
    selected_strategy_name = st.selectbox(
        "Strategy",
        strategy_names,
        format_func=lambda x: f"{x} - {strategies[x]['description'][:60]}...",
        label_visibility="collapsed"
    )

    selected_strategy = strategies[selected_strategy_name]

    # Show strategy documentation - compact
    with st.expander("📖 Docs", expanded=False):
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
                    max_leverage=config_data['max_leverage'],
                    max_position_size=config_data['max_position_size'],
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


def render_results_sheet():
    """Sheet 3: Results and Analysis"""
    st.subheader("📊 Backtest Results")

    if 'backtest_result' not in st.session_state:
        st.info("No backtest results yet. Please run a backtest first in the 'Strategy' tab.")
        return

    result = st.session_state.backtest_result
    config = st.session_state.backtest_config

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

    # Monthly Heatmap Comparison Section - compact
    st.write("")  # Small spacer
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

            st.dataframe(
                display_df.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
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
                max_leverage=config_dict.get('max_leverage', 1.0),
                max_position_size=config_dict.get('max_position_size', None),
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
    st.caption("**Parameter Grid Optimization**")
    st.write("Test multiple parameter combinations and validate each with CPCV.")

    st.info("🚧 Feature coming soon! This will allow you to optimize parameters using CPCV validation.")
    st.write("**Planned features:**")
    st.write("• Define parameter grids (e.g., lookback: 10-50, threshold: 0-0.05)")
    st.write("• Run CPCV on each combination")
    st.write("• Compare PBO/DSR across parameters")
    st.write("• Find robust parameter sets")


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
            final_value = result.equity_curve['total_equity'].iloc[-1]
            initial_value = result.equity_curve['total_equity'].iloc[0]
            total_return = (final_value / initial_value - 1) * 100
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
