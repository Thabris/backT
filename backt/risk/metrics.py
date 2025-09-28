"""
Risk and performance metrics calculation for BackT

Provides comprehensive risk analytics and performance metrics
including Sharpe ratio, maximum drawdown, and other standard measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats

from ..utils.config import BacktestConfig
from ..utils.constants import TRADING_DAYS_PER_YEAR
from ..utils.logging_config import LoggerMixin


class MetricsEngine(LoggerMixin):
    """Calculates performance and risk metrics"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.risk_free_rate = config.risk_free_rate

    def calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics

        Args:
            equity_curve: DataFrame with equity time series
            trades: DataFrame with individual trades (optional)

        Returns:
            Dictionary of calculated metrics
        """
        if equity_curve.empty:
            self.logger.warning("Empty equity curve provided")
            return {}

        metrics = {}

        # Basic return metrics
        metrics.update(self._calculate_return_metrics(equity_curve))

        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_curve))

        # Drawdown metrics
        metrics.update(self._calculate_drawdown_metrics(equity_curve))

        # Trade-level metrics (if trades available)
        if trades is not None and not trades.empty:
            metrics.update(self._calculate_trade_metrics(trades))

        return metrics

    def _calculate_return_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate return-based metrics"""
        if 'total_equity' not in equity_curve.columns:
            return {}

        equity = equity_curve['total_equity']
        initial_equity = equity.iloc[0]
        final_equity = equity.iloc[-1]

        # Total return
        total_return = (final_equity - initial_equity) / initial_equity

        # Calculate returns
        returns = equity.pct_change().dropna()

        if returns.empty:
            return {'total_return': total_return}

        # Annualization factor
        periods_per_year = self._get_annualization_factor(equity_curve.index)

        # CAGR (Compound Annual Growth Rate)
        years = len(equity) / periods_per_year
        cagr = (final_equity / initial_equity) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)

        # Sharpe ratio
        excess_returns = returns - self.risk_free_rate / periods_per_year
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year) \
            if excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = (returns.mean() * periods_per_year - self.risk_free_rate) / downside_deviation \
            if downside_deviation > 0 else 0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'annual_return': returns.mean() * periods_per_year,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'positive_days': (returns > 0).sum() / len(returns),
            'negative_days': (returns < 0).sum() / len(returns)
        }

    def _calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-based metrics"""
        if 'total_equity' not in equity_curve.columns:
            return {}

        equity = equity_curve['total_equity']
        returns = equity.pct_change().dropna()

        if returns.empty:
            return {}

        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)

        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)

        # Beta (if benchmark available)
        beta = np.nan
        alpha = np.nan
        if hasattr(self.config, 'benchmark') and self.config.benchmark:
            # This would require benchmark data to be loaded
            # For now, set to NaN
            pass

        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': beta,
            'alpha': alpha
        }

    def _calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown-based metrics"""
        if 'total_equity' not in equity_curve.columns:
            return {}

        equity = equity_curve['total_equity']

        # Calculate running maximum (peak)
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Drawdown duration
        in_drawdown = drawdown < 0
        drawdown_periods = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_periods.append(current_period)
                current_period = 0

        # Add final period if still in drawdown
        if current_period > 0:
            drawdown_periods.append(current_period)

        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0

        # Calmar ratio (CAGR / Max Drawdown)
        years = len(equity) / self._get_annualization_factor(equity_curve.index)
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.inf

        # Recovery factor
        returns = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'num_drawdown_periods': len(drawdown_periods)
        }

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """Calculate trade-level metrics"""
        if trades.empty:
            return {}

        # Trade PnL calculation would need to be implemented based on
        # how trades are structured. For now, basic stats:

        num_trades = len(trades)

        # If we have fill prices and quantities, we can calculate basic stats
        if 'filled_qty' in trades.columns and 'fill_price' in trades.columns:
            trade_values = trades['filled_qty'] * trades['fill_price']

            avg_trade_size = trade_values.abs().mean()
            largest_trade = trade_values.abs().max()

            # Buy/sell breakdown
            buys = trades[trades['filled_qty'] > 0]
            sells = trades[trades['filled_qty'] < 0]

            return {
                'total_trades': num_trades,
                'buy_trades': len(buys),
                'sell_trades': len(sells),
                'avg_trade_size': avg_trade_size,
                'largest_trade': largest_trade,
                'avg_commission': trades['commission'].mean() if 'commission' in trades.columns else 0,
                'total_commission': trades['commission'].sum() if 'commission' in trades.columns else 0
            }

        return {'total_trades': num_trades}

    def _get_annualization_factor(self, date_index: pd.DatetimeIndex) -> float:
        """Determine annualization factor based on data frequency"""
        if len(date_index) < 2:
            return TRADING_DAYS_PER_YEAR

        # Calculate average time delta
        deltas = date_index[1:] - date_index[:-1]
        avg_delta = deltas.mean()

        # Convert to days
        avg_days = avg_delta.total_seconds() / (24 * 3600)

        if avg_days <= 1:
            return TRADING_DAYS_PER_YEAR
        elif avg_days <= 7:
            return 52  # Weekly
        elif avg_days <= 31:
            return 12  # Monthly
        elif avg_days <= 95:
            return 4   # Quarterly
        else:
            return 1   # Annual

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.DataFrame,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics

        Args:
            equity_curve: Equity curve DataFrame
            window: Rolling window size in periods

        Returns:
            DataFrame with rolling metrics
        """
        if 'total_equity' not in equity_curve.columns:
            return pd.DataFrame()

        equity = equity_curve['total_equity']
        returns = equity.pct_change().dropna()

        if len(returns) < window:
            return pd.DataFrame()

        # Rolling metrics
        rolling_metrics = pd.DataFrame(index=returns.index)

        # Rolling returns
        rolling_metrics['rolling_return'] = returns.rolling(window).sum()

        # Rolling volatility
        rolling_metrics['rolling_volatility'] = returns.rolling(window).std() * np.sqrt(252)

        # Rolling Sharpe
        excess_returns = returns - self.risk_free_rate / 252
        rolling_metrics['rolling_sharpe'] = (
            excess_returns.rolling(window).mean() / excess_returns.rolling(window).std() * np.sqrt(252)
        )

        # Rolling max drawdown
        rolling_max = equity.rolling(window).max()
        rolling_dd = (equity - rolling_max) / rolling_max
        rolling_metrics['rolling_max_drawdown'] = rolling_dd.rolling(window).min()

        return rolling_metrics

    def compare_to_benchmark(
        self,
        equity_curve: pd.DataFrame,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compare performance to benchmark

        Args:
            equity_curve: Strategy equity curve
            benchmark_returns: Benchmark return series

        Returns:
            Comparison metrics
        """
        if 'total_equity' not in equity_curve.columns:
            return {}

        strategy_returns = equity_curve['total_equity'].pct_change().dropna()

        # Align series
        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
        if len(common_dates) == 0:
            return {}

        strat_aligned = strategy_returns.loc[common_dates]
        bench_aligned = benchmark_returns.loc[common_dates]

        # Calculate metrics
        correlation = strat_aligned.corr(bench_aligned)

        # Beta
        beta = strat_aligned.cov(bench_aligned) / bench_aligned.var()

        # Alpha
        alpha = strat_aligned.mean() - beta * bench_aligned.mean()

        # Information ratio
        active_returns = strat_aligned - bench_aligned
        information_ratio = active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0

        # Tracking error
        tracking_error = active_returns.std() * np.sqrt(252)

        return {
            'correlation': correlation,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error
        }