"""Test that all imports work correctly"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print(f"Python path includes: {sys.path[0]}")
print()

# Test backt imports
print("Testing backt imports...")
from backt import Backtester, BacktestConfig
from backt.reporting import PerformanceReport
from backt.utils.config import ExecutionConfig
from backt.validation import CPCVValidator, CPCVConfig
from backt.utils import BookEditor, Book
print("[OK] backt imports successful")
print()

# Test strategies imports
print("Testing strategies imports...")
from strategies import momentum, aqr
print("[OK] strategies imports successful")
print()

# Test streamlit import
print("Testing streamlit import...")
import streamlit as st
print(f"[OK] streamlit version: {st.__version__}")
print()

print("=" * 50)
print("ALL IMPORTS SUCCESSFUL!")
print("=" * 50)
print()
print("You can now run:")
print('  streamlit run streamlit_backtest_runner.py')
