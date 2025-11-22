"""
Complete Streamlit Web Interface for BackT

Run with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add parent directory (project root) to path to find backt module
project_root = Path(r"C:\Users\maxim\Documents\Projects\backtester2")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
from typing import Dict, Any

# Import BackT components
from backt import Backtester, BacktestConfig
from backt.signal import TechnicalIndicators, StrategyHelpers
from backt.data.loaders import YahooDataLoader, CustomDataLoader


# Page configuration
st.set_page_config(
    page_title="BackT Trading Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Compact layout with breathing room
st.markdown("""
<style>
    /* Reduce padding and margins globally */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .element-container {
        margin-bottom: 0.4rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem;
    }
    h1 {
        padding-top: 0rem;
        margin-bottom: 0.6rem;
        font-size: 2.1rem;
    }
    h2 {
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        padding-top: 0.4rem;
        font-size: 1.4rem;
    }
    h3 {
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
        font-size: 1.15rem;
    }
    .stButton button {
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }
    div[data-testid="column"] {
        padding: 0.3rem;
    }
    /* Compact sidebar */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
        padding-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.4rem;
    }
    section[data-testid="stSidebar"] h2 {
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
        font-size: 1.15rem;
    }
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.4rem;
        margin-bottom: 0.3rem;
        font-size: 1.05rem;
    }
    section[data-testid="stSidebar"] label {
        font-size: 0.88rem;
        margin-bottom: 0.3rem;
    }
    section[data-testid="stSidebar"] .stNumberInput,
    section[data-testid="stSidebar"] .stTextInput,
    section[data-testid="stSidebar"] .stDateInput,
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.4rem;
    }
    section[data-testid="stSidebar"] input {
        padding: 0.35rem 0.6rem;
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] .row-widget.stRadio > div {
        gap: 0.6rem;
    }
    section[data-testid="stSidebar"] .row-widget.stRadio label {
        font-size: 0.85rem;
    }
    /* Compact dataframes */
    div[data-testid="stDataFrame"] {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }
    /* Compact expanders */
    .streamlit-expanderHeader {
        font-size: 0.92rem;
    }
    /* Compact metrics */
    div[data-testid="metric-container"] {
        padding: 0.4rem 0.6rem;
    }
    /* Compact charts */
    div[data-testid="stPlotlyChart"] {
        margin-bottom: 0.6rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 0.6rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff6b6b;
    }
    .success-metric {
        border-left-color: #51cf66;
    }
    .warning-metric {
        border-left-color: #ffd43b;
    }
    .danger-metric {
        border-left-color: #ff6b6b;
    }
</style>
""", unsafe_allow_html=True)


def create_synthetic_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Create synthetic data for demo purposes"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    np.random.seed(42)
    n_periods = len(dates)
    base_price = 150.0

    returns = np.random.normal(0.0005, 0.02, n_periods)
    prices = base_price * np.exp(np.cumsum(returns))

    close_prices = prices
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]

    noise = 0.005
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.normal(0, noise, n_periods)))
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.normal(0, noise, n_periods)))

    volumes = np.random.randint(500000, 2000000, n_periods)

    return pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)


def synthetic_data_loader(symbols, start_date, end_date, **kwargs):
    """Synthetic data loader for demo"""
    if isinstance(symbols, str):
        symbols = [symbols]
    return {symbol: create_synthetic_data(symbol, start_date, end_date) for symbol in symbols}


def moving_average_strategy(market_data, current_time, positions, context, params):
    """Moving average crossover strategy"""
    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)

    orders = {}
    for symbol, data in market_data.items():
        if len(data) < long_period:
            continue

        short_ma = TechnicalIndicators.sma(data['close'], short_period)
        long_ma = TechnicalIndicators.sma(data['close'], long_period)

        if StrategyHelpers.is_crossover(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)
        elif StrategyHelpers.is_crossunder(short_ma, long_ma):
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def buy_and_hold_strategy(market_data, current_time, positions, context, params):
    """Simple buy and hold strategy"""
    if 'initialized' not in context:
        context['initialized'] = True
        target_weight = 1.0 / len(market_data)
        return {symbol: StrategyHelpers.create_target_weight_order(weight=target_weight)
                for symbol in market_data.keys()}
    return {}


def rsi_strategy(market_data, current_time, positions, context, params):
    """RSI mean reversion strategy"""
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)

    orders = {}
    for symbol, data in market_data.items():
        if len(data) < rsi_period + 1:
            continue

        rsi = TechnicalIndicators.rsi(data['close'], rsi_period)
        current_rsi = rsi.iloc[-1]

        if current_rsi < rsi_oversold:
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=1.0)
        elif current_rsi > rsi_overbought:
            orders[symbol] = StrategyHelpers.create_target_weight_order(weight=0.0)

    return orders


def create_equity_chart(equity_curve):
    """Create interactive equity curve chart"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve['total_equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=20, t=45, b=45)
    )

    return fig


def create_drawdown_chart(equity_curve):
    """Create drawdown chart"""
    equity = equity_curve['total_equity']
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=drawdown * 100,
        fill='tonexty',
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))

    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=380,
        margin=dict(l=50, r=20, t=45, b=45)
    )

    return fig


def main():
    """Main Streamlit application"""

    # Header
    st.title("🚀 BackT - Professional Trading Backtester")
    st.caption("Real-time strategy backtesting with professional-grade analytics")

    # Sidebar Configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")

        # Date inputs - compact
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start", value=date(2022, 1, 1), max_value=date.today(), label_visibility="visible")
        with col2:
            end_date = st.date_input("End", value=date(2023, 1, 1), max_value=date.today(), label_visibility="visible")

        # Capital and symbols - compact
        initial_capital = st.number_input("Capital ($)", value=100000, min_value=1000, step=10000, format="%d")

        # Common ETFs list
        COMMON_ETFS = [
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO",
            "AGG", "BND", "LQD", "HYG", "TLT", "IEF", "SHY",
            "GLD", "SLV", "USO", "UNG",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
            "EEM", "EFA", "FXI", "EWJ", "EWZ"
        ]

        # Initialize session state for symbols if not exists
        if 'symbols_input' not in st.session_state:
            st.session_state.symbols_input = "AAPL"

        # Button to select all ETFs
        col1, col2 = st.columns([3, 1])
        with col1:
            symbols_input = st.text_input("Symbols", value=st.session_state.symbols_input, placeholder="AAPL, MSFT, TSLA", key="symbols_text_input")
        with col2:
            if st.button("ETFs", help="Select all common ETFs", use_container_width=True):
                st.session_state.symbols_input = ", ".join(COMMON_ETFS)
                st.rerun()

        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

        # Data source - compact
        data_source = st.radio("Data", ["Yahoo Finance", "Synthetic"], horizontal=True)

        # Strategy selection - compact
        st.subheader("📊 Strategy")
        strategy_type = st.selectbox("Type", ["Moving Average Crossover", "Buy and Hold", "RSI Mean Reversion"], label_visibility="collapsed")

        # Strategy parameters - compact
        strategy_params = {}

        if strategy_type == "Moving Average Crossover":
            col1, col2 = st.columns(2)
            with col1:
                strategy_params['short_period'] = st.number_input("Short", value=20, min_value=1, max_value=100, step=1)
            with col2:
                strategy_params['long_period'] = st.number_input("Long", value=50, min_value=1, max_value=200, step=5)

        elif strategy_type == "RSI Mean Reversion":
            strategy_params['rsi_period'] = st.number_input("Period", value=14, min_value=2, max_value=50)
            col1, col2 = st.columns(2)
            with col1:
                strategy_params['rsi_oversold'] = st.number_input("Low", value=30, min_value=10, max_value=50)
            with col2:
                strategy_params['rsi_overbought'] = st.number_input("High", value=70, min_value=50, max_value=90)

        # Advanced settings - compact
        with st.expander("🔧 Advanced"):
            col1, col2 = st.columns(2)
            with col1:
                commission = st.number_input("Commission", value=0.001, min_value=0.0, step=0.001, format="%.3f")
            with col2:
                slippage = st.number_input("Slippage %", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

    # Main content area - compact summary
    col1, col2, col3, col4, col5 = st.columns([1, 1.5, 1, 1.5, 2])
    with col1:
        st.metric("Symbols", len(symbols))
    with col2:
        st.metric("Capital", f"${initial_capital:,}")
    with col3:
        days = (end_date - start_date).days
        st.metric("Days", days)
    with col4:
        st.metric("Strategy", strategy_type.split()[0])
    with col5:
        st.write("")  # Spacer
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
            run_backtest = True
        else:
            run_backtest = False

    # Run backtest
    if run_backtest:

        # Progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Initializing backtest configuration...")
            progress_bar.progress(10)

            # Create configuration
            config = BacktestConfig(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                initial_capital=initial_capital,
                verbose=False  # Disable verbose for Streamlit
            )

            # Update execution config
            config.execution.commission_per_share = commission
            config.execution.slippage_pct = slippage / 100

            status_text.text("Setting up data loader...")
            progress_bar.progress(20)

            # Create data loader
            if data_source == "Yahoo Finance":
                try:
                    data_loader = YahooDataLoader()
                    # Test connection
                    test_data = data_loader.load(symbols[0], start_date.strftime('%Y-%m-%d'), start_date.strftime('%Y-%m-%d'))
                except:
                    st.warning("⚠️ Yahoo Finance failed, using synthetic data")
                    data_loader = CustomDataLoader(synthetic_data_loader)
            else:
                data_loader = CustomDataLoader(synthetic_data_loader)

            status_text.text("Creating backtester...")
            progress_bar.progress(30)

            # Create backtester
            backtester = Backtester(config, data_loader=data_loader)

            # Select strategy function
            if strategy_type == "Moving Average Crossover":
                strategy_func = moving_average_strategy
            elif strategy_type == "Buy and Hold":
                strategy_func = buy_and_hold_strategy
            else:  # RSI
                strategy_func = rsi_strategy

            status_text.text("Running backtest simulation...")
            progress_bar.progress(50)

            # Run backtest
            result = backtester.run(
                strategy=strategy_func,
                universe=symbols,
                strategy_params=strategy_params
            )

            status_text.text("Calculating performance metrics...")
            progress_bar.progress(80)

            # Clear progress indicators
            progress_bar.progress(100)
            status_text.text("✅ Backtest completed successfully!")

            # Display results
            st.success(f"✅ Backtest completed in {result.total_runtime_seconds:.2f}s")

            # Performance metrics
            st.subheader("📈 Performance Metrics")

            metrics = result.performance_metrics

            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)

            total_return = metrics.get('total_return', 0)
            with col1:
                delta_color = "normal" if total_return >= 0 else "inverse"
                st.metric(
                    "Total Return",
                    f"{total_return:.2%}",
                    delta=f"{total_return:.2%}"
                )

            sharpe = metrics.get('sharpe_ratio', 0)
            with col2:
                st.metric("Sharpe Ratio", f"{sharpe:.3f}")

            max_dd = metrics.get('max_drawdown', 0)
            with col3:
                st.metric("Max Drawdown", f"{max_dd:.2%}")

            with col4:
                st.metric("Total Trades", len(result.trades))

            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                cagr = metrics.get('cagr', 0)
                st.metric("CAGR", f"{cagr:.2%}")

            with col2:
                volatility = metrics.get('annualized_volatility', 0)
                st.metric("Volatility", f"{volatility:.2%}")

            with col3:
                sortino = metrics.get('sortino_ratio', 0)
                st.metric("Sortino Ratio", f"{sortino:.3f}")

            with col4:
                calmar = metrics.get('calmar_ratio', 0)
                if np.isfinite(calmar):
                    st.metric("Calmar Ratio", f"{calmar:.3f}")
                else:
                    st.metric("Calmar Ratio", "N/A")

            # Charts
            st.subheader("📊 Performance Charts")

            if not result.equity_curve.empty:
                col1, col2 = st.columns(2)

                with col1:
                    equity_fig = create_equity_chart(result.equity_curve)
                    st.plotly_chart(equity_fig, use_container_width=True)

                with col2:
                    drawdown_fig = create_drawdown_chart(result.equity_curve)
                    st.plotly_chart(drawdown_fig, use_container_width=True)

            # Trade analysis
            if not result.trades.empty:
                st.subheader("💼 Trade Analysis")

                col1, col2, col3, col4 = st.columns(4)
                buy_trades = len(result.trades[result.trades['side'] == 'buy'])
                sell_trades = len(result.trades[result.trades['side'] == 'sell'])

                with col1:
                    st.metric("Buy Trades", buy_trades)
                with col2:
                    st.metric("Sell Trades", sell_trades)
                with col3:
                    st.metric("Avg Trade Size", f"${result.trades['value'].mean():,.0f}")
                with col4:
                    st.metric("Total Commission", f"${result.trades['commission'].sum():,.2f}")

                st.caption("Recent Trades")
                recent_trades = result.trades.tail(5)[['side', 'quantity', 'price', 'value']]
                st.dataframe(recent_trades, use_container_width=True, hide_index=True)

            # Portfolio timeline
            if not result.equity_curve.empty:
                st.subheader("💰 Portfolio Timeline")

                # Summary statistics
                initial_equity = result.equity_curve['total_equity'].iloc[0]
                final_equity = result.equity_curve['total_equity'].iloc[-1]
                max_equity = result.equity_curve['total_equity'].max()
                min_equity = result.equity_curve['total_equity'].min()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Initial Value", f"${initial_equity:,.0f}")
                with col2:
                    st.metric("Final Value", f"${final_equity:,.0f}")
                with col3:
                    st.metric("Peak Value", f"${max_equity:,.0f}")
                with col4:
                    st.metric("Lowest Value", f"${min_equity:,.0f}")

                portfolio_summary = result.equity_curve[['total_equity', 'cash', 'total_pnl']].tail(10)
                st.line_chart(portfolio_summary, height=280)

        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
            st.exception(e)

        finally:
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

    else:
        # Default state - show instructions
        col1, col2 = st.columns([1, 1])

        with col1:
            st.info("👋 **Welcome to BackT!** Configure parameters in the sidebar and click **Run Backtest** to begin.")

        with col2:
            st.markdown("""
            **Features:** 📊 Multiple strategies • 📈 Performance metrics • 🎯 Interactive charts • ⚡ Fast execution
            """)

        # Show sample data
        st.subheader("📋 Current Configuration")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.write(f"**Symbols:** {', '.join(symbols)}")
        with col2:
            st.write(f"**Start:** {start_date}")
        with col3:
            st.write(f"**End:** {end_date}")
        with col4:
            st.write(f"**Capital:** ${initial_capital:,}")
        with col5:
            st.write(f"**Strategy:** {strategy_type.split()[0]}")


if __name__ == "__main__":
    main()