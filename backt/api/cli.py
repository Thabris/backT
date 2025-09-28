"""
Command Line Interface for BackT

Provides CLI access to backtesting functionality.
"""

import click
import json
from pathlib import Path
from typing import Optional

from ..engine.backtester import Backtester
from ..utils.config import BacktestConfig
from ..utils.logging_config import setup_logging


class BacktestCLI:
    """Command line interface for BackT"""

    @staticmethod
    @click.command()
    @click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
    @click.option('--symbols', '-s', multiple=True, help='Symbols to backtest')
    @click.option('--start-date', help='Start date (YYYY-MM-DD)')
    @click.option('--end-date', help='End date (YYYY-MM-DD)')
    @click.option('--initial-capital', type=float, default=100000, help='Initial capital')
    @click.option('--output-dir', '-o', help='Output directory for results')
    @click.option('--verbose', '-v', is_flag=True, help='Verbose output')
    def run_backtest(
        config: Optional[str],
        symbols: tuple,
        start_date: Optional[str],
        end_date: Optional[str],
        initial_capital: float,
        output_dir: Optional[str],
        verbose: bool
    ):
        """Run a backtest from command line"""

        # Load configuration
        if config:
            with open(config, 'r') as f:
                config_dict = json.load(f)
            backtest_config = BacktestConfig.from_dict(config_dict)
        else:
            # Create config from CLI arguments
            if not start_date or not end_date:
                click.echo("Error: start-date and end-date are required when not using config file")
                return

            backtest_config = BacktestConfig(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                verbose=verbose,
                output_dir=output_dir
            )

        if not symbols:
            click.echo("Error: At least one symbol is required")
            return

        # Set up logging
        if verbose:
            setup_logging("INFO")

        # Create and run backtester
        backtester = Backtester(backtest_config)

        # This would need a strategy function - for CLI we'd need to support
        # strategy modules or simple built-in strategies
        click.echo("CLI backtesting requires strategy implementation")
        click.echo("Please use the Python API for full functionality")


if __name__ == "__main__":
    BacktestCLI.run_backtest()