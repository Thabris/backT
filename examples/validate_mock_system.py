"""
Validation Script for BackT Mock Data System

Quick validation to ensure mock data system works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_mock_data_generation():
    """Test basic mock data generation"""
    print("Testing mock data generation...")

    try:
        from backt.data.mock_data import MockDataLoader

        # Test basic data generation
        loader = MockDataLoader(scenario='normal', seed=42)
        data = loader.load(['SPY', 'TLT'], '2020-01-01', '2020-12-31')

        print(f"  ✓ Generated data for {len(data)} symbols")

        for symbol, df in data.items():
            print(f"  ✓ {symbol}: {len(df)} data points, price range ${df['close'].min():.2f}-${df['close'].max():.2f}")

        return True

    except Exception as e:
        print(f"  ✗ Mock data generation failed: {e}")
        return False

def test_mock_integration():
    """Test integration with BackT engine"""
    print("\nTesting BackT integration...")

    try:
        from backt import Backtester, BacktestConfig

        # Simple strategy for testing
        def test_strategy(market_data, current_time, positions, context, params):
            return {'SPY': {'action': 'target_weight', 'weight': 1.0}}

        # Create config with mock data
        config = BacktestConfig(
            start_date='2020-01-01',
            end_date='2020-06-30',
            initial_capital=100000,
            use_mock_data=True,
            mock_scenario='normal',
            mock_seed=42,
            verbose=False
        )

        # Run quick backtest
        backtester = Backtester(config)
        result = backtester.run(test_strategy, ['SPY', 'TLT'])

        print(f"  ✓ Backtest completed successfully")
        print(f"  ✓ Final portfolio value: ${result.portfolio_value:,.0f}")
        print(f"  ✓ Total return: {result.performance_metrics.get('total_return', 0):.2%}")

        return True

    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scenarios():
    """Test different market scenarios"""
    print("\nTesting market scenarios...")

    scenarios = ['normal', 'bull', 'bear', 'volatile']

    try:
        from backt.data.mock_data import MockDataLoader

        for scenario in scenarios:
            loader = MockDataLoader(scenario=scenario, seed=42)
            data = loader.load(['SPY'], '2020-01-01', '2020-12-31')

            spy_data = data['SPY']
            start_price = spy_data['close'].iloc[0]
            end_price = spy_data['close'].iloc[-1]
            total_return = (end_price / start_price) - 1

            print(f"  ✓ {scenario.capitalize()}: {total_return:.1%} return, volatility: {spy_data['close'].pct_change().std() * (252**0.5):.1%}")

        return True

    except Exception as e:
        print(f"  ✗ Scenario test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("="*50)
    print("BackT Mock Data System Validation")
    print("="*50)

    tests = [
        test_mock_data_generation,
        test_mock_integration,
        test_scenarios
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n{'='*50}")
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All tests passed! Mock data system is working correctly.")
        print("\nYou can now run:")
        print("  python mock_data_example.py")
        print("  python mock_data_example.py --scenario bull")
        print("  python mock_data_example.py --compare")
        return True
    else:
        print("✗ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)