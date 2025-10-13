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

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.reporting import PerformanceReport
from backt.utils.config import ExecutionConfig

# Import all strategies from strategies module
from strategies import momentum


# Page configuration
st.set_page_config(
    page_title="BackT Backtest Runner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
    }
</style>
""", unsafe_allow_html=True)


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
    st.header("⚙️ Backtest Configuration")

    with st.form("config_form"):
        # Date Range
        st.subheader("📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=date(2020, 1, 1),
                max_value=date.today()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=date(2023, 12, 31),
                max_value=date.today()
            )

        # Capital Settings
        st.subheader("💰 Capital Settings")
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                value=100000,
                min_value=1000,
                step=1000,
                format="%d"
            )
        with col2:
            allow_short = st.checkbox("Allow Short Selling", value=True)

        # Trading Universe
        st.subheader("🌍 Trading Universe")
        universe_input = st.text_area(
            "Symbols (comma-separated)",
            value="SPY, QQQ, TLT, GLD",
            help="Enter stock symbols separated by commas"
        )
        symbols = [s.strip().upper() for s in universe_input.split(',') if s.strip()]
        st.caption(f"Selected {len(symbols)} symbols: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")

        # Execution Costs
        st.subheader("💸 Execution Costs")
        col1, col2, col3 = st.columns(3)
        with col1:
            spread = st.number_input(
                "Bid-Ask Spread (%)",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f"
            )
        with col2:
            slippage_pct = st.number_input(
                "Slippage (%)",
                value=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.2f"
            )
        with col3:
            commission_per_share = st.number_input(
                "Commission per share ($)",
                value=0.0,
                min_value=0.0,
                step=0.001,
                format="%.3f"
            )

        # Risk Management
        st.subheader("🛡️ Risk Management")
        col1, col2 = st.columns(2)
        with col1:
            max_leverage = st.number_input(
                "Max Leverage (1.0 = 100%)",
                value=2.0,
                min_value=1.0,
                max_value=10.0,
                step=0.5,
                help="Maximum portfolio leverage allowed"
            )
        with col2:
            max_position_size = st.number_input(
                "Max Position Size (0.25 = 25%)",
                value=0.25,
                min_value=0.01,
                max_value=1.0,
                step=0.05,
                help="Maximum single position as fraction of portfolio"
            )

        # Data Settings
        st.subheader("📊 Data Settings")
        use_mock_data = st.checkbox("Use Mock Data (for testing)", value=False)

        submitted = st.form_submit_button("✅ Save Configuration", type="primary", use_container_width=True)

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
    st.header("📈 Strategy Selection")

    if 'config' not in st.session_state:
        st.warning("⚠️ Please configure backtest parameters first in the 'Configuration' tab.")
        return

    # Get available strategies
    strategies = get_available_strategies()

    if not strategies:
        st.error("No strategies found! Make sure strategies are properly defined in the strategies/ folder.")
        return

    # Strategy selection
    strategy_names = list(strategies.keys())
    strategy_descriptions = [strategies[name]['description'] for name in strategy_names]

    selected_strategy_name = st.selectbox(
        "Select Strategy",
        strategy_names,
        format_func=lambda x: f"{x} - {strategies[x]['description'][:80]}..."
    )

    selected_strategy = strategies[selected_strategy_name]

    # Show strategy documentation
    with st.expander("📖 Strategy Documentation"):
        st.markdown(f"**Module:** `strategies.{selected_strategy['module']}`")
        st.markdown(f"**Function:** `{selected_strategy_name}`")
        if selected_strategy['docstring']:
            st.code(selected_strategy['docstring'], language='text')

    # Extract and render parameters
    st.subheader("⚙️ Strategy Parameters")

    strategy_func = selected_strategy['function']
    params_spec = extract_strategy_params(strategy_func)

    if not params_spec:
        st.info("This strategy has no configurable parameters or parameters could not be auto-detected.")
        strategy_params = {}
    else:
        strategy_params = {}

        # Group parameters by type
        int_params = {k: v for k, v in params_spec.items() if v['type'] == 'int'}
        float_params = {k: v for k, v in params_spec.items() if v['type'] == 'float'}
        bool_params = {k: v for k, v in params_spec.items() if v['type'] == 'bool'}

        # Render integer parameters
        if int_params:
            st.markdown("**Integer Parameters**")
            cols = st.columns(min(3, len(int_params)))
            for idx, (param_name, param_info) in enumerate(int_params.items()):
                with cols[idx % 3]:
                    strategy_params[param_name] = st.number_input(
                        param_name.replace('_', ' ').title(),
                        value=param_info['default'] if param_info['default'] is not None else 20,
                        min_value=1,
                        step=1,
                        help=param_info['description']
                    )

        # Render float parameters
        if float_params:
            st.markdown("**Float Parameters**")
            cols = st.columns(min(3, len(float_params)))
            for idx, (param_name, param_info) in enumerate(float_params.items()):
                with cols[idx % 3]:
                    strategy_params[param_name] = st.number_input(
                        param_name.replace('_', ' ').title(),
                        value=param_info['default'] if param_info['default'] is not None else 0.1,
                        min_value=0.0,
                        step=0.001,
                        format="%.4f",
                        help=param_info['description']
                    )

        # Render boolean parameters
        if bool_params:
            st.markdown("**Boolean Parameters**")
            cols = st.columns(min(3, len(bool_params)))
            for idx, (param_name, param_info) in enumerate(bool_params.items()):
                with cols[idx % 3]:
                    strategy_params[param_name] = st.checkbox(
                        param_name.replace('_', ' ').title(),
                        value=param_info['default'] if param_info['default'] is not None else False,
                        help=param_info['description']
                    )

    # Run Backtest Button
    st.divider()

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        run_button = st.button(
            "🚀 Run Backtest",
            type="primary",
            use_container_width=True
        )

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

                st.success("✅ Backtest completed! Go to 'Results' tab to view analysis.")

            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                st.exception(e)


def render_results_sheet():
    """Sheet 3: Results and Analysis"""
    st.header("📊 Backtest Results")

    if 'backtest_result' not in st.session_state:
        st.info("No backtest results yet. Please run a backtest first in the 'Strategy' tab.")
        return

    result = st.session_state.backtest_result
    config = st.session_state.backtest_config

    # Create performance report
    report = PerformanceReport(result, initial_capital=config.initial_capital)

    # Summary Section
    st.subheader("📈 Performance Summary")

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

    st.divider()

    # Charts Section
    st.subheader("📈 Performance Charts")

    # Generate charts from report
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    # Equity Curve
    fig_equity = report.plot_equity_curve(show=False)
    if fig_equity:
        st.pyplot(fig_equity)
        plt.close(fig_equity)

    # Drawdown Chart
    fig_dd = report.plot_drawdown(show=False)
    if fig_dd:
        st.pyplot(fig_dd)
        plt.close(fig_dd)

    # Monthly Returns Heatmap
    fig_heatmap = report.plot_monthly_returns_heatmap(show=False)
    if fig_heatmap:
        st.pyplot(fig_heatmap)
        plt.close(fig_heatmap)

    st.divider()

    # Per-Symbol Analysis
    st.subheader("🎯 Per-Symbol Performance")

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
        fig_per_symbol = report.plot_per_symbol_pnl(show=False)
        if fig_per_symbol:
            st.pyplot(fig_per_symbol)
            plt.close(fig_per_symbol)

        # Correlation heatmap
        corr_matrix = report.get_correlation_matrix()
        if corr_matrix is not None and not corr_matrix.empty:
            fig_corr = report.plot_correlation_heatmap(show=False)
            if fig_corr:
                st.pyplot(fig_corr)
                plt.close(fig_corr)

    st.divider()

    # Full Metrics Table
    with st.expander("📊 Detailed Metrics"):
        st.dataframe(metrics_df, use_container_width=True)

    # Trade History
    with st.expander("💼 Trade History"):
        if not result.trades.empty:
            st.dataframe(
                result.trades.head(100),
                use_container_width=True
            )
            st.caption(f"Showing first 100 of {len(result.trades)} trades")
        else:
            st.info("No trades executed")


def main():
    """Main application"""

    # Header
    st.title("🚀 BackT Backtest Runner")
    st.markdown("Professional multi-strategy backtesting platform")

    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "⚙️ Configuration",
        "📈 Strategy",
        "📊 Results"
    ])

    with tab1:
        render_configuration_sheet()

    with tab2:
        render_strategy_sheet()

    with tab3:
        render_results_sheet()

    # Sidebar - Status
    with st.sidebar:
        st.header("📋 Session Status")

        if 'config' in st.session_state:
            st.success("✅ Configuration Set")
            config = st.session_state.config
            st.caption(f"Period: {config['start_date']} to {config['end_date']}")
            st.caption(f"Capital: ${config['initial_capital']:,}")
            st.caption(f"Universe: {len(config['symbols'])} symbols")
        else:
            st.warning("⚠️ No Configuration")

        if 'selected_strategy_name' in st.session_state:
            st.success(f"✅ Strategy: {st.session_state.selected_strategy_name}")
        else:
            st.warning("⚠️ No Strategy Selected")

        if 'backtest_result' in st.session_state:
            st.success("✅ Backtest Complete")
            result = st.session_state.backtest_result
            final_value = result.equity_curve['total_equity'].iloc[-1]
            initial_value = result.equity_curve['total_equity'].iloc[0]
            total_return = (final_value / initial_value - 1) * 100
            st.caption(f"Return: {total_return:.2f}%")
            st.caption(f"Trades: {len(result.trades)}")
        else:
            st.warning("⚠️ No Results")

        st.divider()

        # Reset button
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
