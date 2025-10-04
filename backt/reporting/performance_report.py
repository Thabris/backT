"""
Performance reporting module for BackT

Provides standardized, reusable reporting functionality for metrics and visualizations
that can be used in both Jupyter notebooks and Streamlit applications.
"""

from typing import Dict, Any, Optional, List, Literal
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

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


@dataclass
class ReportConfig:
    """Configuration for performance reports"""

    # Metrics sections to include
    include_returns: bool = True
    include_risk: bool = True
    include_trading: bool = True
    include_portfolio: bool = True
    include_assessment: bool = True

    # Chart types to generate
    include_equity_curve: bool = True
    include_drawdown: bool = True
    include_monthly_returns: bool = True
    include_risk_return: bool = True

    # Detailed metrics options
    verbose_metrics: bool = False  # Include all available metrics

    # Chart styling
    chart_style: str = "seaborn-v0_8"
    figsize: tuple = (15, 10)

    # Benchmark comparison (optional)
    benchmark_return: Optional[float] = None  # Annualized return (e.g., 0.08 for 8%)
    benchmark_volatility: Optional[float] = None  # Annualized volatility (e.g., 0.15 for 15%)
    benchmark_name: str = "Benchmark"


class PerformanceReport(LoggerMixin):
    """
    Generates standardized performance reports with metrics and visualizations.

    Outputs are designed to be reusable across Jupyter notebooks, Streamlit apps,
    and other interfaces.

    Example:
        >>> report = PerformanceReport(result)
        >>>
        >>> # Get formatted text output
        >>> print(report.format_text_report())
        >>>
        >>> # Get metrics as dictionary
        >>> metrics_dict = report.get_metrics_dict()
        >>>
        >>> # Get metrics as DataFrame
        >>> df = report.get_metrics_dataframe()
        >>>
        >>> # Generate charts
        >>> fig = report.generate_charts()
        >>> plt.show()
    """

    def __init__(
        self,
        result: BacktestResult,
        config: Optional[ReportConfig] = None,
        initial_capital: Optional[float] = None
    ):
        """
        Initialize performance report

        Args:
            result: BacktestResult object from backtester
            config: Report configuration (uses defaults if None)
            initial_capital: Initial capital for calculations (extracted from result if None)
        """
        self.result = result
        self.config = config or ReportConfig()
        self.metrics = result.performance_metrics

        # Extract initial capital
        if initial_capital is not None:
            self.initial_capital = initial_capital
        elif not result.equity_curve.empty:
            self.initial_capital = result.equity_curve.iloc[0]
        else:
            self.initial_capital = 100000  # Default fallback

    def format_text_report(self, use_emojis: bool = True) -> str:
        """
        Generate formatted text report

        Args:
            use_emojis: Whether to include emoji icons

        Returns:
            Formatted string ready for printing
        """
        sections = []

        # Header
        emoji_prefix = "📊 " if use_emojis else ""
        sections.append(f"{emoji_prefix}PERFORMANCE ANALYSIS")
        sections.append("=" * 50)

        # Return metrics
        if self.config.include_returns:
            sections.append(self._format_returns_section(use_emojis))

        # Risk metrics
        if self.config.include_risk:
            sections.append(self._format_risk_section(use_emojis))

        # Trading activity
        if self.config.include_trading:
            sections.append(self._format_trading_section(use_emojis))

        # Portfolio summary
        if self.config.include_portfolio:
            sections.append(self._format_portfolio_section(use_emojis))

        # Performance assessment
        if self.config.include_assessment:
            sections.append(self._format_assessment_section(use_emojis))

        return "\n".join(sections)

    def _format_returns_section(self, use_emojis: bool) -> str:
        """Format return metrics section"""
        emoji = "📈 " if use_emojis else ""
        lines = [f"\n{emoji}RETURN METRICS:", "-" * 25]

        metrics_to_show = [
            ('total_return', 'Total Return:', '{:8.2%}'),
            ('annualized_return', 'Annualized Return:', '{:8.2%}'),
            ('cagr', 'CAGR:', '{:8.2%}'),
        ]

        if self.config.verbose_metrics:
            metrics_to_show.extend([
                ('best_day', 'Best Day:', '{:8.2%}'),
                ('worst_day', 'Worst Day:', '{:8.2%}'),
                ('avg_daily_return', 'Avg Daily Return:', '{:8.2%}'),
            ])

        for key, label, fmt in metrics_to_show:
            value = self.metrics.get(key, 0)
            lines.append(f"{label:25} {fmt.format(value)}")

        return "\n".join(lines)

    def _format_risk_section(self, use_emojis: bool) -> str:
        """Format risk metrics section"""
        emoji = "⚠️  " if use_emojis else ""
        lines = [f"\n{emoji}RISK METRICS:", "-" * 20]

        metrics_to_show = [
            ('volatility', 'Volatility:', '{:8.2%}'),
            ('sharpe_ratio', 'Sharpe Ratio:', '{:8.3f}'),
            ('sortino_ratio', 'Sortino Ratio:', '{:8.3f}'),
            ('max_drawdown', 'Maximum Drawdown:', '{:8.2%}'),
            ('calmar_ratio', 'Calmar Ratio:', '{:8.3f}'),
        ]

        if self.config.verbose_metrics:
            metrics_to_show.extend([
                ('value_at_risk_95', 'VaR (95%):', '{:8.2%}'),
                ('conditional_var_95', 'CVaR (95%):', '{:8.2%}'),
                ('downside_deviation', 'Downside Deviation:', '{:8.2%}'),
            ])

        for key, label, fmt in metrics_to_show:
            value = self.metrics.get(key, 0)
            lines.append(f"{label:25} {fmt.format(value)}")

        return "\n".join(lines)

    def _format_trading_section(self, use_emojis: bool) -> str:
        """Format trading activity section"""
        emoji = "📊 " if use_emojis else ""
        lines = [f"\n{emoji}TRADING ACTIVITY:", "-" * 24]

        metrics_to_show = [
            ('total_trades', 'Total Trades:', '{:8.0f}'),
            ('win_rate', 'Win Rate:', '{:8.1%}'),
            ('profit_factor', 'Profit Factor:', '{:8.2f}'),
            ('avg_trade_return', 'Average Trade:', '{:8.2%}'),
        ]

        if self.config.verbose_metrics:
            metrics_to_show.extend([
                ('winning_trades', 'Winning Trades:', '{:8.0f}'),
                ('losing_trades', 'Losing Trades:', '{:8.0f}'),
                ('avg_win', 'Avg Win:', '{:8.2%}'),
                ('avg_loss', 'Avg Loss:', '{:8.2%}'),
                ('largest_win', 'Largest Win:', '{:8.2%}'),
                ('largest_loss', 'Largest Loss:', '{:8.2%}'),
            ])

        for key, label, fmt in metrics_to_show:
            value = self.metrics.get(key, 0)
            lines.append(f"{label:25} {fmt.format(value)}")

        return "\n".join(lines)

    def _format_portfolio_section(self, use_emojis: bool) -> str:
        """Format portfolio summary section"""
        emoji = "💼 " if use_emojis else ""
        lines = [f"\n{emoji}PORTFOLIO SUMMARY:", "-" * 25]

        total_return = self.metrics.get('total_return', 0)
        final_value = self.initial_capital * (1 + total_return)
        profit_loss = final_value - self.initial_capital

        lines.append(f"{'Initial Capital:':25} ${self.initial_capital:>10,.0f}")
        lines.append(f"{'Final Value:':25} ${final_value:>10,.0f}")
        lines.append(f"{'Profit/Loss:':25} ${profit_loss:>10,.0f}")

        if self.config.verbose_metrics:
            peak_value = self.initial_capital * (1 + self.metrics.get('total_return', 0))
            lines.append(f"{'Peak Value:':25} ${peak_value:>10,.0f}")

        return "\n".join(lines)

    def _format_assessment_section(self, use_emojis: bool) -> str:
        """Format performance assessment section"""
        emoji = "🎯 " if use_emojis else ""
        lines = [f"\n{emoji}PERFORMANCE ASSESSMENT:", "-" * 30]

        sharpe = self.metrics.get('sharpe_ratio', 0)
        max_dd = self.metrics.get('max_drawdown', 0)
        total_return = self.metrics.get('total_return', 0)
        final_value = self.initial_capital * (1 + total_return)
        profit_loss = final_value - self.initial_capital

        # Sharpe rating
        if sharpe > 2.0:
            sharpe_rating = "Exceptional"
        elif sharpe > 1.0:
            sharpe_rating = "Excellent"
        elif sharpe > 0.5:
            sharpe_rating = "Good"
        elif sharpe > 0:
            sharpe_rating = "Fair"
        else:
            sharpe_rating = "Poor"

        # Risk rating
        if abs(max_dd) < 0.10:
            dd_rating = "Low Risk"
        elif abs(max_dd) < 0.20:
            dd_rating = "Moderate Risk"
        elif abs(max_dd) < 0.30:
            dd_rating = "High Risk"
        else:
            dd_rating = "Very High Risk"

        lines.append(f"{'Sharpe Rating:':25} {sharpe_rating} ({sharpe:.2f})")
        lines.append(f"{'Risk Rating:':25} {dd_rating} ({max_dd:.1%} max DD)")

        # Final verdict
        verdict_emoji = "✅" if profit_loss > 0 else "❌"
        verdict_emoji = verdict_emoji if use_emojis else ""
        if profit_loss > 0:
            lines.append(f"\n{verdict_emoji} Strategy was PROFITABLE with ${profit_loss:,.0f} gains")
        else:
            lines.append(f"\n{verdict_emoji} Strategy had LOSSES of ${abs(profit_loss):,.0f}")

        return "\n".join(lines)

    def get_metrics_dict(self, include_calculated: bool = True) -> Dict[str, Any]:
        """
        Get metrics as dictionary

        Args:
            include_calculated: Include calculated fields like final_value, profit_loss

        Returns:
            Dictionary of metrics
        """
        metrics = self.metrics.copy()

        if include_calculated:
            total_return = metrics.get('total_return', 0)
            final_value = self.initial_capital * (1 + total_return)

            metrics['initial_capital'] = self.initial_capital
            metrics['final_value'] = final_value
            metrics['profit_loss'] = final_value - self.initial_capital

        return metrics

    def get_metrics_dataframe(
        self,
        transpose: bool = True,
        include_calculated: bool = True
    ) -> pd.DataFrame:
        """
        Get metrics as DataFrame

        Args:
            transpose: If True, metrics are rows; if False, metrics are columns
            include_calculated: Include calculated fields

        Returns:
            DataFrame of metrics
        """
        metrics = self.get_metrics_dict(include_calculated=include_calculated)
        df = pd.DataFrame([metrics])

        if transpose:
            df = df.T
            df.columns = ['Value']

        return df

    def generate_charts(
        self,
        return_fig: bool = True,
        show_plots: bool = False
    ):
        """
        Generate performance visualization charts

        Args:
            return_fig: If True, return the figure object
            show_plots: If True, call plt.show()

        Returns:
            matplotlib Figure object if return_fig=True, else None
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, cannot generate charts")
            return None

        # Determine chart layout
        charts_enabled = [
            self.config.include_equity_curve,
            self.config.include_drawdown,
            self.config.include_monthly_returns,
            self.config.include_risk_return
        ]
        num_charts = sum(charts_enabled)

        if num_charts == 0:
            self.logger.warning("No charts enabled in config")
            return None

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.config.figsize)
        fig.suptitle('BackT Strategy Performance Analysis', fontsize=16, fontweight='bold')

        # Flatten axes for easier indexing
        axes = axes.flatten()
        chart_idx = 0

        # 1. Equity Curve
        if self.config.include_equity_curve:
            self._plot_equity_curve(axes[chart_idx])
            chart_idx += 1

        # 2. Drawdown Chart
        if self.config.include_drawdown:
            self._plot_drawdown(axes[chart_idx])
            chart_idx += 1

        # 3. Monthly Returns
        if self.config.include_monthly_returns:
            self._plot_monthly_returns(axes[chart_idx])
            chart_idx += 1

        # 4. Risk-Return Profile
        if self.config.include_risk_return:
            self._plot_risk_return(axes[chart_idx])
            chart_idx += 1

        # Hide unused subplots
        for idx in range(chart_idx, 4):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if show_plots:
            plt.show()

        if return_fig:
            return fig

        return None

    def _plot_equity_curve(self, ax):
        """Plot equity curve"""
        equity_curve = self.result.equity_curve

        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Portfolio Equity Curve', fontweight='bold')
            return

        # Handle both Series and DataFrame (extract Series if DataFrame)
        if isinstance(equity_curve, pd.DataFrame):
            if 'total_equity' in equity_curve.columns:
                equity_data = equity_curve['total_equity']
            else:
                equity_data = equity_curve.iloc[:, 0]
        else:
            equity_data = equity_curve

        equity_data.plot(ax=ax, linewidth=2, color='blue')
        ax.set_title('Portfolio Equity Curve', fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    def _plot_drawdown(self, ax):
        """Plot drawdown"""
        equity_curve = self.result.equity_curve

        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No equity data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Drawdown Analysis', fontweight='bold')
            return

        # Handle both Series and DataFrame (extract Series if DataFrame)
        if isinstance(equity_curve, pd.DataFrame):
            if 'total_equity' in equity_curve.columns:
                equity_data = equity_curve['total_equity']
            else:
                equity_data = equity_curve.iloc[:, 0]
        else:
            equity_data = equity_curve

        # Calculate drawdown
        rolling_max = equity_data.expanding().max()
        drawdown = (equity_data / rolling_max - 1) * 100

        drawdown.plot(ax=ax, linewidth=2, color='red', alpha=0.7)
        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown Analysis', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

    def _plot_monthly_returns(self, ax):
        """Plot monthly returns heatmap or bar chart"""
        equity_curve = self.result.equity_curve

        if equity_curve.empty:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Monthly Returns', fontweight='bold')
            return

        # Handle both Series and DataFrame (extract Series if DataFrame)
        if isinstance(equity_curve, pd.DataFrame):
            if 'total_equity' in equity_curve.columns:
                equity_data = equity_curve['total_equity']
            else:
                equity_data = equity_curve.iloc[:, 0]
        else:
            equity_data = equity_curve

        # Calculate returns
        returns = equity_data.pct_change().dropna()

        if len(returns) == 0:
            ax.text(0.5, 0.5, 'No returns data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title('Monthly Returns', fontweight='bold')
            return

        monthly_returns = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pct = monthly_returns * 100

        # Create heatmap if we have enough data (>12 months)
        if len(monthly_returns_pct) > 12 and HAS_MATPLOTLIB and sns is not None:
            try:
                monthly_df = monthly_returns_pct.to_frame('returns')
                monthly_df['year'] = monthly_df.index.year
                monthly_df['month'] = monthly_df.index.month

                pivot_table = monthly_df.pivot_table(
                    values='returns',
                    index='year',
                    columns='month'
                )

                sns.heatmap(
                    pivot_table,
                    annot=True,
                    fmt='.1f',
                    cmap='RdYlGn',
                    center=0,
                    ax=ax,
                    cbar_kws={'label': 'Monthly Return (%)'}
                )
                ax.set_title('Monthly Returns Heatmap', fontweight='bold')
                ax.set_xlabel('Month')
                ax.set_ylabel('Year')
            except Exception as e:
                # Fallback to bar chart
                self.logger.debug(f"Heatmap failed: {e}, using bar chart")
                monthly_returns_pct.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_title('Monthly Returns', fontweight='bold')
                ax.set_ylabel('Return (%)')
                ax.tick_params(axis='x', rotation=45)
        else:
            # Bar chart for limited data
            monthly_returns_pct.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Monthly Returns', fontweight='bold')
            ax.set_ylabel('Return (%)')
            ax.tick_params(axis='x', rotation=45)

    def _plot_risk_return(self, ax):
        """Plot risk-return scatter"""
        annual_return = self.metrics.get('annualized_return', 0) * 100
        volatility = self.metrics.get('volatility', 0) * 100
        sharpe = self.metrics.get('sharpe_ratio', 0)

        # Plot strategy point
        ax.scatter(
            volatility, annual_return,
            s=200, c='blue', alpha=0.7,
            label='Strategy',
            edgecolors='black',
            linewidth=2
        )

        # Add benchmark if provided
        if self.config.benchmark_return is not None and self.config.benchmark_volatility is not None:
            bench_return = self.config.benchmark_return * 100
            bench_vol = self.config.benchmark_volatility * 100
            ax.scatter(
                bench_vol, bench_return,
                s=150, c='orange', alpha=0.7,
                label=self.config.benchmark_name,
                edgecolors='black',
                linewidth=2
            )
        else:
            # Add reference point
            ax.scatter(
                15, 8,
                s=100, c='gray', alpha=0.5,
                label='Market Reference (15% vol, 8% ret)'
            )

        # Add reference lines
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Volatility (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title('Risk-Return Profile', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add Sharpe ratio annotation
        ax.annotate(
            f'Sharpe: {sharpe:.2f}',
            xy=(volatility, annual_return),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    def print_report(self, use_emojis: bool = True):
        """Convenience method to print the text report"""
        print(self.format_text_report(use_emojis=use_emojis))

    def show_charts(self):
        """Convenience method to generate and show charts"""
        self.generate_charts(return_fig=False, show_plots=True)
