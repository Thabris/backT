"""
Execution module for BackT

Handles order execution simulation including market impact modeling,
slippage, commissions, and fill simulation. Designed to be pluggable
for different execution models.
"""

from .mock_execution import MockExecutionEngine
from .execution_interface import ExecutionEngine

__all__ = [
    "MockExecutionEngine",
    "ExecutionEngine"
]