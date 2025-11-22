"""
Risk and performance metrics calculation for BackT

Provides comprehensive risk analytics and performance metrics
including Sharpe ratio, maximum drawdown, and other standard measures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from scipy import stats

from ..utils.config import BacktestConfig
from ..utils.constants import TRADING_DAYS_PER_YEAR
from ..utils.logging_config import LoggerMixin

# Import Numba-optimized metrics (optional)
try:
    from .numba_metrics import (
        HAS_NUMBA,
        calculate_sharpe_ratio_fast,
        calculate_sortino_ratio_fast,
        calculate_max_drawdown_fast,
        calculate_var_cvar_fast,
        calculate_return_stats_fast,
        calculate_calmar_ratio_fast,
        calculate_win_rate_fast
    )
except ImportError:
    HAS_NUMBA = False


class MetricsEngine(LoggerMixin):
    """Calculates performance and risk metrics"""

    def __init__(self, config: BacktestConfig, use_numba: bool = True):
        self.config = config
        self.risk_free_rate = config.risk_free_rate
        self.use_numba = use_numba and HAS_NUMBA  # Enable if requested and available

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
        returns = equity.pct_change().dropna()

        if returns.empty:
            initial_equity = equity.iloc[0]
            final_equity = equity.iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity
            return {'total_return': total_return}

        # Annualization factor
        periods_per_year = self._get_annualization_factor(equity_curve.index)

        # Use Numba-optimized versions if available
        if self.use_numba:
            # Fast version using Numba JIT
            equity_array = equity.values
            returns_array = returns.values

            # Calculate actual years from timestamps
            years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25

            total_return, cagr, volatility, best_day, worst_day = calculate_return_stats_fast(
                equity_array, years
            )

            sharpe_ratio = calculate_sharpe_ratio_fast(
                returns_array, self.risk_free_rate, periods_per_year
            )

            sortino_ratio = calculate_sortino_ratio_fast(
                returns_array, self.risk_free_rate, periods_per_year
            )

        else:
            # Original pandas/numpy version
            initial_equity = equity.iloc[0]
            final_equity = equity.iloc[-1]

            # Total return
            total_return = (final_equity - initial_equity) / initial_equity

            # CAGR (Compound Annual Growth Rate)
            # Use actual calendar time instead of data point count
            years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
            if years > 0 and initial_equity > 0 and final_equity > 0:
                cagr = (final_equity / initial_equity) ** (1 / years) - 1
            else:
                cagr = 0.0

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

            best_day = returns.max()
            worst_day = returns.min()

        return {
            'total_return': total_return,
            'cagr': cagr,
            'annualized_volatility': volatility,
            'volatility': volatility,  # Add alias for display compatibility
            'annualized_return': cagr,  # Use CAGR (geometric mean), not arithmetic mean
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'annual_return': cagr,  # Use CAGR (geometric mean), not arithmetic mean
            'best_day': best_day,
            'worst_day': worst_day,
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

        # Use Numba-optimized versions if available
        if self.use_numba:
            returns_array = returns.values

            # Fast VaR/CVaR calculations
            var_95, cvar_95 = calculate_var_cvar_fast(returns_array, 0.95)
            var_99, cvar_99 = calculate_var_cvar_fast(returns_array, 0.99)

            # Skewness and Kurtosis (scipy is already optimized, use as-is)
            skewness = stats.skew(returns_array)
            kurtosis = stats.kurtosis(returns_array)

        else:
            # Original version
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

        # Use Numba-optimized versions if available
        if self.use_numba:
            equity_array = equity.values

            # Fast drawdown calculation
            max_drawdown, max_dd_duration, num_dd_periods = calculate_max_drawdown_fast(equity_array)

            # CAGR for Calmar ratio
            periods_per_year = self._get_annualization_factor(equity_curve.index)
            years = len(equity) / periods_per_year
            if years > 0 and equity_array[0] > 0 and equity_array[-1] > 0:
                cagr = (equity_array[-1] / equity_array[0]) ** (1 / years) - 1
            else:
                cagr = 0.0

            # Calmar ratio (CAGR / Max Drawdown)
            calmar_ratio = calculate_calmar_ratio_fast(cagr, max_drawdown)

            # Recovery factor
            total_return = (equity_array[-1] - equity_array[0]) / equity_array[0]
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

            # Note: avg_drawdown_duration not returned by fast version, calculate separately
            avg_drawdown_duration = 0.0  # Simplified for now
            max_days_to_recover = 0.0  # Not calculated in Numba fast version

        else:
            # Original version
            # Calculate running maximum (peak)
            running_max = equity.expanding().max()

            # Calculate drawdown
            drawdown = (equity - running_max) / running_max

            # Maximum drawdown
            max_drawdown = drawdown.min()

            # Drawdown duration and recovery time
            in_drawdown = drawdown < 0
            drawdown_periods = []
            recovery_periods = []  # Time from trough to recovery
            current_period = 0
            recovery_days = 0
            in_recovery = False
            recovery_start_equity = 0
            recovery_target_equity = 0

            for i, (is_dd, curr_equity) in enumerate(zip(in_drawdown, equity)):
                if is_dd:
                    current_period += 1
                    if not in_recovery:
                        # Track the equity at trough and target for recovery
                        recovery_start_equity = curr_equity
                        recovery_target_equity = running_max.iloc[i]
                else:
                    # Exited drawdown
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        # Start counting recovery time
                        in_recovery = True
                        recovery_days = current_period  # Time in drawdown
                    current_period = 0

                # Track recovery time (from start of drawdown to recovery)
                if in_recovery and curr_equity >= recovery_target_equity:
                    recovery_periods.append(recovery_days)
                    in_recovery = False
                    recovery_days = 0
                elif in_recovery:
                    recovery_days += 1

            # Add final period if still in drawdown
            if current_period > 0:
                drawdown_periods.append(current_period)

            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            num_dd_periods = len(drawdown_periods)

            # Maximum recovery time
            max_days_to_recover = max(recovery_periods) if recovery_periods else 0

            # Calmar ratio (CAGR / Max Drawdown)
            years = len(equity) / self._get_annualization_factor(equity_curve.index)
            if years > 0 and equity.iloc[0] > 0 and equity.iloc[-1] > 0:
                cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
            else:
                cagr = 0.0
            calmar_ratio = cagr / abs(max_drawdown) if max_drawdown < 0 else np.inf

            # Recovery factor
            total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_dd_duration if self.use_numba else max_dd_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_days_to_recover': max_days_to_recover,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'num_drawdown_periods': num_dd_periods if self.use_numba else num_dd_periods
        }

    def _calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate trade-level metrics including PnL analysis
        """
        if trades.empty:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_trade_pnl_pct': 0.0
            }

        num_trades = len(trades)

        # Calculate basic trade statistics
        basic_stats = {
            'total_trades': num_trades,
            'avg_commission': trades['commission'].mean() if 'commission' in trades.columns else 0,
            'total_commission': trades['commission'].sum() if 'commission' in trades.columns else 0
        }

        # Calculate trade PnL if we have sufficient data
        if 'side' in trades.columns and 'quantity' in trades.columns and 'price' in trades.columns:
            # Group trades by symbol to calculate round-trip PnL
            pnl_trades = self._calculate_round_trip_pnl(trades)

            if pnl_trades:
                winning_trades = [pnl for pnl in pnl_trades if pnl > 0]
                losing_trades = [pnl for pnl in pnl_trades if pnl < 0]

                # Win rate
                win_rate = len(winning_trades) / len(pnl_trades) if pnl_trades else 0.0

                # Profit factor (gross profit / gross loss)
                gross_profit = sum(winning_trades) if winning_trades else 0
                gross_loss = abs(sum(losing_trades)) if losing_trades else 0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

                # Average trade PnL
                avg_trade_pnl = sum(pnl_trades) / len(pnl_trades) if pnl_trades else 0.0

                # Average trade PnL percentage (approximate)
                avg_trade_value = trades['value'].mean() if 'value' in trades.columns else 1.0
                avg_trade_pnl_pct = (avg_trade_pnl / avg_trade_value) if avg_trade_value > 0 else 0.0

                basic_stats.update({
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'avg_trade_pnl': avg_trade_pnl,
                    'avg_trade_pnl_pct': avg_trade_pnl_pct,
                    'num_winning_trades': len(winning_trades),
                    'num_losing_trades': len(losing_trades),
                    'gross_profit': gross_profit,
                    'gross_loss': gross_loss
                })
            else:
                # No complete round trips found
                basic_stats.update({
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_trade_pnl': 0.0,
                    'avg_trade_pnl_pct': 0.0
                })

        return basic_stats

    def _calculate_round_trip_pnl(self, trades: pd.DataFrame) -> List[float]:
        """
        Calculate round-trip PnL from individual fills
        Simple FIFO matching for buy/sell pairs
        """
        pnl_list = []

        # Group by symbol
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol].copy()
            symbol_trades = symbol_trades.sort_index()  # Sort by timestamp

            position = 0.0
            cost_basis = 0.0

            for _, trade in symbol_trades.iterrows():
                if trade['side'] == 'buy':
                    # Adding to position
                    new_cost = position * cost_basis + trade['quantity'] * trade['price']
                    position += trade['quantity']
                    cost_basis = new_cost / position if position != 0 else 0

                elif trade['side'] == 'sell':
                    # Reducing position
                    if position > 0:
                        # Calculate PnL for the sold portion
                        sold_qty = min(trade['quantity'], position)
                        pnl = sold_qty * (trade['price'] - cost_basis)
                        pnl_list.append(pnl)

                        # Update position
                        position -= sold_qty

        return pnl_list

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

    def calculate_per_symbol_metrics(
        self,
        per_symbol_equity_curves: Dict[str, pd.DataFrame],
        trades: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for each symbol individually

        Args:
            per_symbol_equity_curves: Dictionary of symbol -> equity curve DataFrame
            trades: DataFrame with individual trades (optional)

        Returns:
            Dictionary of symbol -> metrics dictionary
        """
        per_symbol_metrics = {}

        for symbol, equity_curve in per_symbol_equity_curves.items():
            if equity_curve.empty:
                continue

            # Filter trades for this symbol if available
            symbol_trades = None
            if trades is not None and not trades.empty and 'symbol' in trades.columns:
                symbol_trades = trades[trades['symbol'] == symbol]

            # Calculate metrics using total_pnl as the equity measure
            metrics = self._calculate_symbol_specific_metrics(equity_curve, symbol_trades)
            per_symbol_metrics[symbol] = metrics

        return per_symbol_metrics

    def _calculate_symbol_specific_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Calculate metrics for a single symbol's equity curve"""
        if equity_curve.empty:
            return {}

        metrics = {}

        # Use total_equity if available (proper equity curve), otherwise fall back to total_pnl
        equity_column = 'total_equity' if 'total_equity' in equity_curve.columns else 'total_pnl'

        if equity_column in equity_curve.columns:
            equity_series = equity_curve[equity_column]

            # Calculate percentage returns from equity curve
            returns = equity_series.pct_change().dropna()

            if not returns.empty and len(returns) > 1:
                periods_per_year = self._get_annualization_factor(equity_curve.index)

                # Total return and CAGR
                initial_equity = equity_series.iloc[0]
                final_equity = equity_series.iloc[-1]

                if initial_equity > 0:
                    total_return = (final_equity / initial_equity) - 1

                    # Calculate time period for CAGR
                    start_date = equity_curve.index[0]
                    end_date = equity_curve.index[-1]
                    num_years = (end_date - start_date).days / 365.25

                    if num_years > 0:
                        # Protect against invalid power operations when losses exceed 100%
                        # (e.g., from shorting volatility ETFs or leveraged instruments)
                        if total_return <= -1.0:
                            # Total loss or worse - CAGR is undefined, use large negative value
                            cagr = -1.0
                        else:
                            cagr = (1 + total_return) ** (1 / num_years) - 1
                    else:
                        cagr = 0
                else:
                    total_return = 0
                    cagr = 0

                # Volatility (annualized)
                volatility = returns.std() * np.sqrt(periods_per_year)

                # Sharpe ratio (using proper percentage returns)
                mean_return = returns.mean()
                sharpe_ratio = (mean_return / returns.std() * np.sqrt(periods_per_year)) \
                    if returns.std() > 0 else 0

                # Sortino ratio (downside deviation)
                downside_returns = returns[returns < 0]
                downside_deviation = downside_returns.std()
                sortino_ratio = (mean_return / downside_deviation * np.sqrt(periods_per_year)) \
                    if downside_deviation > 0 else 0

                # Drawdown metrics
                running_max = equity_series.expanding().max()
                drawdown = (equity_series - running_max) / running_max
                max_drawdown = drawdown.min()

                # Best and worst periods
                best_period = returns.max()
                worst_period = returns.min()

                # Win rate
                win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

                # PnL metrics
                if 'total_pnl' in equity_curve.columns:
                    total_pnl = equity_curve['total_pnl'].iloc[-1] - equity_curve['total_pnl'].iloc[0]
                    final_pnl = equity_curve['total_pnl'].iloc[-1]
                else:
                    total_pnl = final_equity - initial_equity
                    final_pnl = total_pnl

                metrics.update({
                    'total_return': total_return,
                    'cagr': cagr,
                    'total_pnl': total_pnl,
                    'final_pnl': final_pnl,
                    'initial_equity': initial_equity,
                    'final_equity': final_equity,
                    'annualized_volatility': volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'best_period': best_period,
                    'worst_period': worst_period,
                    'win_rate': win_rate
                })

        # Add trade-level metrics if available
        if trades is not None and not trades.empty:
            trade_metrics = self._calculate_trade_metrics(trades)
            metrics.update({
                f'trades_{k}': v for k, v in trade_metrics.items()
            })

        return metrics

    def calculate_returns_correlation_matrix(
        self,
        per_symbol_equity_curves: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix of percentage returns between symbols

        Uses percentage changes in total_pnl to normalize for position size differences.
        Correlation is calculated using Pearson correlation coefficient on the
        percentage return time series.

        Args:
            per_symbol_equity_curves: Dictionary of symbol -> equity curve DataFrame

        Returns:
            Correlation matrix as DataFrame with Pearson correlation coefficients
        """
        returns_dict = {}

        # Extract percentage returns for each symbol
        for symbol, equity_curve in per_symbol_equity_curves.items():
            if equity_curve.empty:
                continue

            # Use total_pnl percentage changes as returns
            if 'total_pnl' in equity_curve.columns:
                pnl_series = equity_curve['total_pnl']

                # Calculate percentage change in PnL
                # pct_change() = (pnl[t] - pnl[t-1]) / pnl[t-1]
                # We need to handle the case where previous PnL is 0 or near 0

                # Shift to get previous values
                pnl_prev = pnl_series.shift(1)

                # Calculate absolute change
                pnl_change = pnl_series - pnl_prev

                # Calculate percentage change with safety for division by zero
                # Use absolute value of previous PnL to avoid sign issues
                pnl_prev_abs = pnl_prev.abs()

                # Where previous PnL is very small (< $1), use absolute change instead
                # This prevents huge percentage spikes when crossing zero
                min_base = 100.0  # Minimum base for percentage calculation
                pnl_base = pnl_prev_abs.where(pnl_prev_abs >= min_base, min_base)

                # Calculate percentage returns
                pct_returns = (pnl_change / pnl_base) * 100  # In percentage points

                # Drop NaN values (first row and any other invalid values)
                pct_returns = pct_returns.dropna()

                # Only include if we have valid returns
                if len(pct_returns) > 0:
                    returns_dict[symbol] = pct_returns

        if not returns_dict:
            return pd.DataFrame()

        # Create returns DataFrame (aligned by timestamp)
        returns_df = pd.DataFrame(returns_dict)

        # Drop any rows with NaN values to ensure clean correlation calculation
        returns_df = returns_df.dropna()

        # Calculate Pearson correlation matrix
        # Pearson correlation: corr(X,Y) = cov(X,Y) / (std(X) * std(Y))
        correlation_matrix = returns_df.corr(method='pearson')

        return correlation_matrix