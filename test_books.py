"""
Test script for the Books feature

This script tests the basic functionality of the Book and BookManager classes.
"""

from backt.utils.books import Book, BookManager, create_book_from_session


def test_books():
    """Test the books functionality"""
    print("=" * 60)
    print("Testing Books Feature")
    print("=" * 60)

    # 1. Create BookManager
    print("\n1. Creating BookManager...")
    manager = BookManager()
    print(f"   [OK] Books directory: {manager.books_dir}")

    # 2. Create a sample book
    print("\n2. Creating sample book...")
    book = create_book_from_session(
        name="Test_Momentum_Strategy",
        strategy_module="MOMENTUM",
        strategy_name="ma_crossover_long_only",
        strategy_params={
            'fast_ma': 20,
            'slow_ma': 50,
            'min_periods': 60
        },
        symbols=["AAPL", "MSFT", "GOOGL"],
        description="Test momentum strategy for tech stocks",
        tags=["momentum", "long-only", "tech"]
    )
    print(f"   [OK] Created book: {book}")

    # 3. Save the book
    print("\n3. Saving book...")
    manager.save_book(book, overwrite=True)
    print(f"   [OK] Book saved successfully")

    # 4. List all books
    print("\n4. Listing all books...")
    books = manager.list_books()
    print(f"   [OK] Found {len(books)} book(s):")
    for b in books:
        print(f"     - {b}")

    # 5. Load the book
    print("\n5. Loading book...")
    loaded_book = manager.load_book("Test_Momentum_Strategy")
    print(f"   [OK] Loaded book: {loaded_book.name}")
    print(f"     - Strategy: {loaded_book.strategy_module}.{loaded_book.strategy_name}")
    print(f"     - Parameters: {loaded_book.strategy_params}")
    print(f"     - Symbols: {loaded_book.symbols}")
    print(f"     - Description: {loaded_book.description}")
    print(f"     - Tags: {loaded_book.tags}")

    # 6. Verify book data
    print("\n6. Verifying book data...")
    assert loaded_book.name == book.name, "Book name mismatch"
    assert loaded_book.strategy_module == book.strategy_module, "Strategy module mismatch"
    assert loaded_book.strategy_name == book.strategy_name, "Strategy name mismatch"
    assert loaded_book.strategy_params == book.strategy_params, "Parameters mismatch"
    assert loaded_book.symbols == book.symbols, "Symbols mismatch"
    assert loaded_book.description == book.description, "Description mismatch"
    assert loaded_book.tags == book.tags, "Tags mismatch"
    print("   [OK] All data verified successfully")

    # 7. Test book exists
    print("\n7. Testing book_exists()...")
    assert manager.book_exists("Test_Momentum_Strategy"), "Book should exist"
    assert not manager.book_exists("NonExistentBook"), "Book should not exist"
    print("   [OK] book_exists() working correctly")

    # 8. Get books by tag
    print("\n8. Testing get_books_by_tag()...")
    momentum_books = manager.get_books_by_tag("momentum")
    print(f"   [OK] Found {len(momentum_books)} book(s) with tag 'momentum'")

    # 9. Get books by strategy
    print("\n9. Testing get_books_by_strategy()...")
    ma_books = manager.get_books_by_strategy("ma_crossover_long_only")
    print(f"   [OK] Found {len(ma_books)} book(s) using 'ma_crossover_long_only' strategy")

    # 10. Test to_dict and from_dict
    print("\n10. Testing to_dict() and from_dict()...")
    book_dict = book.to_dict()
    recreated_book = Book.from_dict(book_dict)
    assert recreated_book.name == book.name, "Recreated book name mismatch"
    print("   [OK] Serialization/deserialization working correctly")

    # 11. Test updating book symbols
    print("\n11. Testing symbol update functionality...")
    original_symbols = book.symbols.copy()
    print(f"   Original symbols: {original_symbols}")

    # Add new symbols
    new_symbols = original_symbols + ["NVDA", "TSLA"]
    book.symbols = new_symbols
    manager.save_book(book, overwrite=True)
    print(f"   Updated symbols: {new_symbols}")

    # Reload and verify
    reloaded_book = manager.load_book("Test_Momentum_Strategy")
    assert reloaded_book.symbols == new_symbols, "Symbol update failed"
    assert len(reloaded_book.symbols) == 5, f"Expected 5 symbols, got {len(reloaded_book.symbols)}"
    print(f"   [OK] Successfully updated from {len(original_symbols)} to {len(new_symbols)} symbols")

    # Restore original symbols
    book.symbols = original_symbols
    manager.save_book(book, overwrite=True)
    print("   [OK] Restored original symbols")

    print("\n" + "=" * 60)
    print("[SUCCESS] All tests passed!")
    print("=" * 60)

    # Optional: Clean up - uncomment to delete test book
    # print("\nCleaning up...")
    # manager.delete_book("Test_Momentum_Strategy")
    # print("   [OK] Test book deleted")


if __name__ == "__main__":
    try:
        test_books()
    except Exception as e:
        print(f"\n[FAILED] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
