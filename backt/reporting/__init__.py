"""
Reporting module for BackT

Handles output generation including trade logs, equity curves, performance reports,
and visualization in various formats (CSV, JSON, Parquet, plots).
"""

from .report_generator import ReportGenerator
from .trade_logger import TradeLogger
from .visualization import PlotGenerator

__all__ = [
    "ReportGenerator",
    "TradeLogger",
    "PlotGenerator"
]