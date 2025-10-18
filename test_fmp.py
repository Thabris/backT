"""
Test FMP integration with fundamentals loader
"""

import os
from backt.data.fundamentals import FundamentalsLoader, calculate_quality_score, calculate_value_score

def test_fmp_integration():
    """Test FMP data loading and fallback"""

    print("=" * 60)
    print("Testing FinancialModelingPrep Integration")
    print("=" * 60)

    # Check for API key
    fmp_key = os.environ.get('FMP_API_KEY')
    if fmp_key:
        print(f"\nOK FMP API key found: {fmp_key[:8]}...")
    else:
        print("\nWARNING: No FMP API key found - will use yfinance only")

    print("\n" + "-" * 60)
    print("Test 1: Auto mode (FMP with yfinance fallback)")
    print("-" * 60)

    loader = FundamentalsLoader(data_source='auto')

    # Test with AAPL (should work with both FMP and yfinance)
    print("\nLoading AAPL...")
    aapl_data = loader.get_fundamentals('AAPL')

    if aapl_data:
        print(f"OK Loaded AAPL from: {aapl_data.get('data_source', 'unknown')}")
        print(f"  - ROE: {aapl_data.get('roe')}")
        print(f"  - ROA: {aapl_data.get('roa')}")
        print(f"  - Profit Margin: {aapl_data.get('profit_margin')}")
        print(f"  - Debt/Equity: {aapl_data.get('debt_to_equity')}")
        print(f"  - P/E Ratio: {aapl_data.get('trailing_pe')}")
        print(f"  - P/B Ratio: {aapl_data.get('price_to_book')}")
        print(f"  - Beta: {aapl_data.get('beta')}")
        print(f"  - Sector: {aapl_data.get('sector')}")

        # Calculate scores
        quality = calculate_quality_score(aapl_data)
        value = calculate_value_score(aapl_data)
        print(f"\n  Quality Score: {quality:.4f}" if quality else "  Quality Score: N/A")
        print(f"  Value Score: {value:.4f}" if value else "  Value Score: N/A")
    else:
        print("FAIL Failed to load AAPL")

    # Test with TSLA
    print("\nLoading TSLA...")
    tsla_data = loader.get_fundamentals('TSLA')

    if tsla_data:
        print(f"OK Loaded TSLA from: {tsla_data.get('data_source', 'unknown')}")
        print(f"  - ROE: {tsla_data.get('roe')}")
        print(f"  - Debt/Equity: {tsla_data.get('debt_to_equity')}")
        print(f"  - P/E Ratio: {tsla_data.get('trailing_pe')}")
    else:
        print("FAIL Failed to load TSLA")

    # Test with MSFT
    print("\nLoading MSFT...")
    msft_data = loader.get_fundamentals('MSFT')

    if msft_data:
        print(f"OK Loaded MSFT from: {msft_data.get('data_source', 'unknown')}")
        print(f"  - ROE: {msft_data.get('roe')}")
        print(f"  - Current Ratio: {msft_data.get('current_ratio')}")
    else:
        print("FAIL Failed to load MSFT")

    # Check API usage stats
    print("\n" + "-" * 60)
    print("API Usage Statistics")
    print("-" * 60)
    stats = loader.get_api_usage_stats()
    print(f"Data source: {stats['data_source']}")
    print(f"Has FMP key: {stats['has_fmp_key']}")
    print(f"FMP calls today: {stats['fmp_calls_today']}/250")
    print(f"FMP calls remaining: {stats['fmp_calls_remaining']}")
    print(f"Cache size: {stats['cache_size']} symbols")

    # Test cache
    print("\n" + "-" * 60)
    print("Test 2: Cache functionality")
    print("-" * 60)

    print("\nLoading AAPL again (should use cache)...")
    aapl_data2 = loader.get_fundamentals('AAPL')

    if aapl_data2:
        print(f"OK Loaded from cache")
        stats2 = loader.get_api_usage_stats()
        if stats2['fmp_calls_today'] == stats['fmp_calls_today']:
            print("  OK No additional API call made (cache working)")
        else:
            print("  WARNING API call was made (cache may not be working)")

    # Test yfinance-only mode
    print("\n" + "-" * 60)
    print("Test 3: yfinance-only mode")
    print("-" * 60)

    yf_loader = FundamentalsLoader(data_source='yfinance')
    print("\nLoading GOOGL with yfinance only...")
    googl_data = yf_loader.get_fundamentals('GOOGL')

    if googl_data:
        print(f"OK Loaded GOOGL from: {googl_data.get('data_source', 'unknown')}")
        print(f"  - ROE: {googl_data.get('roe')}")
    else:
        print("FAIL Failed to load GOOGL")

    print("\n" + "=" * 60)
    print("Tests Complete!")
    print("=" * 60)

    # Summary
    print("\nSummary:")
    if stats['has_fmp_key']:
        print(f"OK FMP integration active - {stats['fmp_calls_today']} API calls used")
    else:
        print("WARNING Using yfinance only (no FMP key)")
    print(f"OK {stats['cache_size']} symbols cached")
    print("\nRecommendation:")
    if not stats['has_fmp_key']:
        print("  -> Get a free FMP API key for better data quality")
        print("  -> See FMP_SETUP.md for instructions")
    else:
        print("  -> FMP is working! You have more reliable fundamental data.")

if __name__ == '__main__':
    test_fmp_integration()
