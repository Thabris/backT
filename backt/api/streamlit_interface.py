"""
Streamlit interface for BackT

Provides web-based interface for backtesting.
"""

from typing import Optional, Dict, Any


class StreamlitInterface:
    """Streamlit web interface for BackT"""

    def __init__(self):
        pass

    def create_streamlit_app(self):
        """Create Streamlit application"""
        try:
            import streamlit as st
        except ImportError:
            raise ImportError("Streamlit is required for web interface. Install with: pip install streamlit")

        st.title("BackT - Trading Backtester")
        st.write("Web interface for BackT backtesting framework")

        # Configuration inputs
        with st.sidebar:
            st.header("Configuration")
            start_date = st.date_input("Start Date")
            end_date = st.date_input("End Date")
            initial_capital = st.number_input("Initial Capital", value=100000, min_value=1000)
            symbols = st.text_input("Symbols (comma-separated)", value="AAPL,MSFT")

        # Strategy selection
        st.header("Strategy")
        strategy_type = st.selectbox("Strategy Type", ["Moving Average Crossover", "Buy and Hold"])

        if strategy_type == "Moving Average Crossover":
            short_period = st.number_input("Short MA Period", value=20, min_value=1)
            long_period = st.number_input("Long MA Period", value=50, min_value=1)

        # Run backtest button
        if st.button("Run Backtest"):
            st.write("Backtesting functionality would be implemented here")
            st.write("This requires integration with the main BackT engine")

        # Results display
        st.header("Results")
        st.write("Results would be displayed here after running backtest")