"""
Tests for configuration module
"""

import pytest
import pandas as pd
from backt.utils.config import BacktestConfig, ExecutionConfig


class TestBacktestConfig:
    """Test BacktestConfig class"""

    def test_valid_config(self):
        """Test creating a valid configuration"""
        config = BacktestConfig(
            start_date='2020-01-01',
            end_date='2022-01-01',
            initial_capital=100000
        )

        assert config.start_date == '2020-01-01'
        assert config.end_date == '2022-01-01'
        assert config.initial_capital == 100000
        assert config.start_datetime == pd.to_datetime('2020-01-01')
        assert config.end_datetime == pd.to_datetime('2022-01-01')

    def test_invalid_dates(self):
        """Test configuration with invalid dates"""
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='invalid_date',
                end_date='2022-01-01',
                initial_capital=100000
            )

    def test_start_after_end(self):
        """Test configuration with start date after end date"""
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='2022-01-01',
                end_date='2020-01-01',
                initial_capital=100000
            )

    def test_negative_capital(self):
        """Test configuration with negative initial capital"""
        with pytest.raises(ValueError):
            BacktestConfig(
                start_date='2020-01-01',
                end_date='2022-01-01',
                initial_capital=-1000
            )

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = BacktestConfig(
            start_date='2020-01-01',
            end_date='2022-01-01',
            initial_capital=50000
        )

        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['start_date'] == '2020-01-01'
        assert config_dict['initial_capital'] == 50000

    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            'start_date': '2020-01-01',
            'end_date': '2022-01-01',
            'initial_capital': 75000,
            'execution': {
                'spread': 0.02,
                'commission_per_share': 0.01
            }
        }

        config = BacktestConfig.from_dict(config_dict)
        assert config.start_date == '2020-01-01'
        assert config.initial_capital == 75000
        assert config.execution.spread == 0.02


class TestExecutionConfig:
    """Test ExecutionConfig class"""

    def test_valid_execution_config(self):
        """Test creating valid execution configuration"""
        config = ExecutionConfig(
            spread=0.01,
            slippage_pct=0.001,
            commission_per_share=0.005
        )

        assert config.spread == 0.01
        assert config.slippage_pct == 0.001
        assert config.commission_per_share == 0.005

    def test_negative_spread(self):
        """Test execution config with negative spread"""
        with pytest.raises(ValueError):
            ExecutionConfig(spread=-0.01)

    def test_negative_slippage(self):
        """Test execution config with negative slippage"""
        with pytest.raises(ValueError):
            ExecutionConfig(slippage_pct=-0.001)