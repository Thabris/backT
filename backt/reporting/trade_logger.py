"""
Trade logging for BackT

Handles logging and tracking of all trades and fills
for reporting and analysis purposes.
"""

from typing import List, Dict, Optional
import pandas as pd

from ..utils.types import Fill
from ..utils.logging_config import LoggerMixin


class TradeLogger(LoggerMixin):
    """Logs and manages trade history"""

    def __init__(self):
        self.fills: List[Fill] = []

    def log_fill(self, fill: Fill) -> None:
        """Log a trade fill"""
        self.fills.append(fill)
        self.logger.debug(f"Logged fill: {fill.side} {abs(fill.filled_qty)} {fill.symbol} at ${fill.fill_price:.2f}")

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get all trades as a pandas DataFrame"""
        if not self.fills:
            return pd.DataFrame()

        trades_data = []
        for fill in self.fills:
            trade_record = {
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'side': fill.side,
                'quantity': abs(fill.filled_qty),
                'price': fill.fill_price,
                'value': abs(fill.filled_qty * fill.fill_price),
                'commission': fill.commission,
                'slippage': fill.slippage,
                'net_value': abs(fill.filled_qty * fill.fill_price) - fill.commission,
                'order_id': fill.order_id
            }

            # Add metadata if available
            if fill.meta:
                for key, value in fill.meta.items():
                    trade_record[f'meta_{key}'] = value

            trades_data.append(trade_record)

        df = pd.DataFrame(trades_data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

        return df

    def get_trade_summary(self) -> Dict:
        """Get summary statistics of all trades"""
        if not self.fills:
            return {}

        df = self.get_trades_dataframe()

        if df.empty:
            return {}

        # Basic statistics
        total_trades = len(df)
        total_volume = df['value'].sum()
        total_commission = df['commission'].sum()
        avg_trade_size = df['value'].mean()

        # Buy/sell breakdown
        buys = df[df['side'] == 'buy']
        sells = df[df['side'] == 'sell']

        summary = {
            'total_trades': total_trades,
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'avg_trade_size': avg_trade_size,
            'avg_commission_per_trade': total_commission / total_trades if total_trades > 0 else 0,
            'commission_as_pct_of_volume': total_commission / total_volume if total_volume > 0 else 0
        }

        # Symbol breakdown
        symbol_stats = df.groupby('symbol').agg({
            'quantity': 'sum',
            'value': 'sum',
            'commission': 'sum'
        }).to_dict('index')

        summary['by_symbol'] = symbol_stats

        # Time analysis
        if len(df) > 1:
            time_between_trades = df.index.to_series().diff().dt.total_seconds().dropna()
            summary['avg_time_between_trades_seconds'] = time_between_trades.mean()
            summary['min_time_between_trades_seconds'] = time_between_trades.min()
            summary['max_time_between_trades_seconds'] = time_between_trades.max()

        return summary

    def clear(self) -> None:
        """Clear all logged trades"""
        self.fills.clear()
        self.logger.info("Trade log cleared")

    def export_to_csv(self, filename: str) -> None:
        """Export trades to CSV file"""
        df = self.get_trades_dataframe()
        if not df.empty:
            df.to_csv(filename)
            self.logger.info(f"Exported {len(df)} trades to {filename}")
        else:
            self.logger.warning("No trades to export")