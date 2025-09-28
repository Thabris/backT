"""
Visualization utilities for BackT

Creates plots and charts for backtest analysis.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

from ..utils.types import BacktestResult
from ..utils.logging_config import LoggerMixin


class PlotGenerator(LoggerMixin):
    """Generates plots and visualizations for backtest results"""

    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib and seaborn are required for plotting. Install with: pip install matplotlib seaborn")

        self.style = style
        self.figsize = figsize
        plt.style.use(style)

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ) -> None:
        """Plot equity curve"""
        if equity_curve.empty or 'total_equity' not in equity_curve.columns:
            self.logger.warning("No equity data to plot")
            return

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(equity_curve.index, equity_curve['total_equity'], linewidth=2, label='Portfolio Value')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved equity curve plot to {save_path}")

        plt.show()

    def plot_drawdown(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Drawdown",
        save_path: Optional[str] = None
    ) -> None:
        """Plot drawdown chart"""
        if equity_curve.empty or 'total_equity' not in equity_curve.columns:
            self.logger.warning("No equity data to plot")
            return

        equity = equity_curve['total_equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        fig, ax = plt.subplots(figsize=self.figsize)

        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown, color='darkred', linewidth=1)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved drawdown plot to {save_path}")

        plt.show()

    def create_performance_dashboard(
        self,
        result: BacktestResult,
        save_path: Optional[str] = None
    ) -> None:
        """Create a comprehensive performance dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Backtest Performance Dashboard', fontsize=16, fontweight='bold')

        # Equity curve
        if not result.equity_curve.empty and 'total_equity' in result.equity_curve.columns:
            axes[0, 0].plot(result.equity_curve.index, result.equity_curve['total_equity'])
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True, alpha=0.3)

            # Drawdown
            equity = result.equity_curve['total_equity']
            running_max = equity.expanding().max()
            drawdown = (equity - running_max) / running_max
            axes[0, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.7, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True, alpha=0.3)

        # Monthly returns heatmap (if enough data)
        if not result.equity_curve.empty and len(result.equity_curve) > 30:
            try:
                returns = result.equity_curve['total_equity'].pct_change().dropna()
                monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

                if len(monthly_returns) > 1:
                    monthly_table = monthly_returns.groupby([
                        monthly_returns.index.year,
                        monthly_returns.index.month
                    ]).first().unstack()

                    sns.heatmap(monthly_table, annot=True, fmt='.2%', cmap='RdYlGn',
                              center=0, ax=axes[1, 0], cbar_kws={'label': 'Returns'})
                    axes[1, 0].set_title('Monthly Returns Heatmap')
            except Exception as e:
                axes[1, 0].text(0.5, 0.5, 'Monthly Returns\n(Insufficient Data)',
                              ha='center', va='center', transform=axes[1, 0].transAxes)

        # Performance metrics text
        metrics_text = self._format_metrics_text(result.performance_metrics)
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved dashboard to {save_path}")

        plt.show()

    def _format_metrics_text(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for text display"""
        if not metrics:
            return "No metrics available"

        lines = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'ratio' in key.lower() or 'return' in key.lower():
                    lines.append(f"{key}: {value:.3f}")
                elif 'drawdown' in key.lower():
                    lines.append(f"{key}: {value:.2%}")
                else:
                    lines.append(f"{key}: {value:.4f}")
            else:
                lines.append(f"{key}: {value}")

        return '\n'.join(lines[:15])  # Limit to first 15 metrics