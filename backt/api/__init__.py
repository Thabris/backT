"""
API module for BackT

Provides programmatic and CLI interfaces for running backtests,
as well as integration utilities for web frameworks like Streamlit.
"""

from .cli import BacktestCLI
from .streamlit_interface import StreamlitInterface

__all__ = [
    "BacktestCLI",
    "StreamlitInterface"
]