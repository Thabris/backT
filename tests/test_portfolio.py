"""
Tests for portfolio management module
"""

import pytest
import pandas as pd
from backt.portfolio.portfolio_manager import PortfolioManager
from backt.utils.config import BacktestConfig
from backt.utils.types import Fill, Position


class TestPortfolioManager:
    """Test PortfolioManager class"""

    def create_test_config(self):
        """Create test configuration"""
        return BacktestConfig(
            start_date='2020-01-01',
            end_date='2022-01-01',
            initial_capital=100000
        )

    def create_test_fill(self, symbol='AAPL', qty=100, price=150.0):
        """Create test fill"""
        return Fill(
            symbol=symbol,
            filled_qty=qty,
            fill_price=price,
            commission=1.0,
            slippage=0.1,
            timestamp=pd.Timestamp('2020-01-01'),
            order_id='test_001',
            side='buy' if qty > 0 else 'sell'
        )

    def test_portfolio_initialization(self):
        """Test portfolio manager initialization"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        assert portfolio.cash == 100000
        assert portfolio.initial_capital == 100000
        assert len(portfolio.positions) == 0

    def test_process_fill_new_position(self):
        """Test processing fill for new position"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        fill = self.create_test_fill(qty=100, price=150.0)
        current_prices = {'AAPL': 155.0}

        portfolio.process_fill(fill, current_prices)

        assert 'AAPL' in portfolio.positions
        position = portfolio.positions['AAPL']
        assert position.qty == 100
        assert position.avg_price == 150.0
        assert portfolio.cash == 100000 - (100 * 150.0) - 1.0  # Initial - trade value - commission

    def test_process_fill_add_to_position(self):
        """Test adding to existing position"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # First fill
        fill1 = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill1, {'AAPL': 150.0})

        # Second fill
        fill2 = self.create_test_fill(qty=50, price=160.0)
        portfolio.process_fill(fill2, {'AAPL': 160.0})

        position = portfolio.positions['AAPL']
        assert position.qty == 150
        # Average price should be weighted average
        expected_avg = (100 * 150.0 + 50 * 160.0) / 150
        assert abs(position.avg_price - expected_avg) < 0.01

    def test_process_fill_close_position(self):
        """Test closing a position"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Open position
        fill1 = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill1, {'AAPL': 150.0})

        # Close position
        fill2 = self.create_test_fill(qty=-100, price=160.0)
        portfolio.process_fill(fill2, {'AAPL': 160.0})

        position = portfolio.positions['AAPL']
        assert position.qty == 0
        assert position.realized_pnl == (160.0 - 150.0) * 100  # Profit from price appreciation

    def test_get_portfolio_value(self):
        """Test portfolio value calculation"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Add position
        fill = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill, {'AAPL': 150.0})

        current_prices = {'AAPL': 160.0}
        portfolio_value = portfolio.get_portfolio_value(current_prices)

        expected_value = portfolio.cash + (100 * 160.0)
        assert abs(portfolio_value - expected_value) < 0.01

    def test_update_positions(self):
        """Test updating position values"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Add position
        fill = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill, {'AAPL': 150.0})

        # Update with new prices
        current_prices = {'AAPL': 155.0}
        timestamp = pd.Timestamp('2020-01-02')
        portfolio.update_positions(current_prices, timestamp)

        position = portfolio.positions['AAPL']
        expected_unrealized = (155.0 - 150.0) * 100
        assert abs(position.unrealized_pnl - expected_unrealized) < 0.01

    def test_get_equity_snapshot(self):
        """Test equity snapshot creation"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Add position
        fill = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill, {'AAPL': 150.0})

        current_prices = {'AAPL': 155.0}
        timestamp = pd.Timestamp('2020-01-01')
        snapshot = portfolio.get_equity_snapshot(current_prices, timestamp)

        assert snapshot['timestamp'] == timestamp
        assert 'cash' in snapshot
        assert 'total_equity' in snapshot
        assert 'unrealized_pnl' in snapshot

    def test_check_risk_limits(self):
        """Test risk limit checking"""
        config = self.create_test_config()
        config.max_position_size = 0.1  # 10% max position size
        portfolio = PortfolioManager(config)

        # Test position within limits
        assert portfolio.check_risk_limits('AAPL', 50, 200.0) is True

        # Test position exceeding limits
        assert portfolio.check_risk_limits('AAPL', 1000, 200.0) is False

    def test_can_afford_trade(self):
        """Test affordability checking"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Affordable trade
        assert portfolio.can_afford_trade('AAPL', 100, 500.0, 10.0) is True

        # Unaffordable trade
        assert portfolio.can_afford_trade('AAPL', 1000, 500.0, 10.0) is False

    def test_get_leverage(self):
        """Test leverage calculation"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Add position
        fill = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill, {'AAPL': 150.0})

        current_prices = {'AAPL': 150.0}
        leverage = portfolio.get_leverage(current_prices)

        # Should be close to (100 * 150) / total_equity
        expected_leverage = (100 * 150.0) / portfolio.get_portfolio_value(current_prices)
        assert abs(leverage - expected_leverage) < 0.01

    def test_reset(self):
        """Test portfolio reset"""
        config = self.create_test_config()
        portfolio = PortfolioManager(config)

        # Add some activity
        fill = self.create_test_fill(qty=100, price=150.0)
        portfolio.process_fill(fill, {'AAPL': 150.0})

        # Reset
        portfolio.reset()

        assert portfolio.cash == config.initial_capital
        assert len(portfolio.positions) == 0
        assert len(portfolio.equity_history) == 0