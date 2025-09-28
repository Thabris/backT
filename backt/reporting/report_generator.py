"""
Report generation for BackT

Generates comprehensive reports and outputs from backtest results.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from ..utils.types import BacktestResult
from ..utils.logging_config import LoggerMixin


class ReportGenerator(LoggerMixin):
    """Generates backtest reports in various formats"""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "backtest_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        result: BacktestResult,
        report_name: str = "backtest_report"
    ) -> Dict[str, str]:
        """
        Generate a complete backtest report

        Args:
            result: BacktestResult object
            report_name: Base name for report files

        Returns:
            Dictionary of generated file paths
        """
        generated_files = {}

        # Performance summary (JSON)
        summary_file = self.output_dir / f"{report_name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(result.summary(), f, indent=2, default=str)
        generated_files['summary'] = str(summary_file)

        # Trades CSV
        if not result.trades.empty:
            trades_file = self.output_dir / f"{report_name}_trades.csv"
            result.trades.to_csv(trades_file)
            generated_files['trades'] = str(trades_file)

        # Equity curve CSV
        if not result.equity_curve.empty:
            equity_file = self.output_dir / f"{report_name}_equity.csv"
            result.equity_curve.to_csv(equity_file)
            generated_files['equity'] = str(equity_file)

        # Performance metrics JSON
        metrics_file = self.output_dir / f"{report_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(result.performance_metrics, f, indent=2, default=str)
        generated_files['metrics'] = str(metrics_file)

        self.logger.info(f"Generated report files: {list(generated_files.keys())}")
        return generated_files

    def export_to_excel(
        self,
        result: BacktestResult,
        filename: str = "backtest_results.xlsx"
    ) -> str:
        """Export results to Excel file with multiple sheets"""
        file_path = self.output_dir / filename

        with pd.ExcelWriter(file_path) as writer:
            # Summary sheet
            summary_df = pd.DataFrame([result.summary()])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Trades sheet
            if not result.trades.empty:
                result.trades.to_excel(writer, sheet_name='Trades')

            # Equity curve sheet
            if not result.equity_curve.empty:
                result.equity_curve.to_excel(writer, sheet_name='Equity')

            # Metrics sheet
            metrics_df = pd.DataFrame([result.performance_metrics])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

        self.logger.info(f"Exported results to Excel: {file_path}")
        return str(file_path)