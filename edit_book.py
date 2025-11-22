"""
Interactive CLI for editing saved Books

Usage:
    python edit_book.py <book_name>
    python edit_book.py --list
    python edit_book.py --help

Examples:
    # List all books
    python edit_book.py --list

    # Edit a book interactively
    python edit_book.py "MACD_Top5_2024"

    # Quick edit symbols
    python edit_book.py "MACD_Top5_2024" --add-symbols SPY,QQQ --remove-symbols XLU

    # Quick edit parameters
    python edit_book.py "MACD_Top5_2024" --params fast_period=12 slow_period=26
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from backt.utils.book_editor import BookEditor, quick_edit_symbols, quick_edit_params
from backt.utils.books import Book
from typing import Dict, Any


def interactive_menu(editor: BookEditor, book: Book) -> Book:
    """
    Interactive menu for editing a book.

    Args:
        editor: The book editor instance
        book: The book to edit

    Returns:
        The modified book
    """
    original_book = Book.from_dict(book.to_dict())  # Keep a copy for diff

    while True:
        print("\n" + "=" * 80)
        print(f"EDITING: {book.name}")
        print("=" * 80)
        print("\nOptions:")
        print("  1. View current book")
        print("  2. Add symbols")
        print("  3. Remove symbols")
        print("  4. Replace all symbols")
        print("  5. Edit parameters")
        print("  6. Edit metadata (description, tags)")
        print("  7. Show changes (diff)")
        print("  8. Validate book")
        print("  9. Save and exit")
        print("  0. Exit without saving")
        print()

        choice = input("Choose an option (0-9): ").strip()

        if choice == '1':
            # View book
            print(editor.preview_changes(book))

        elif choice == '2':
            # Add symbols
            symbols_str = input("Enter symbols to add (comma-separated): ").strip()
            if symbols_str:
                symbols = [s.strip().upper() for s in symbols_str.split(',')]
                book = editor.add_symbols(book, symbols)

        elif choice == '3':
            # Remove symbols
            print(f"\nCurrent symbols: {', '.join(book.symbols)}")
            symbols_str = input("Enter symbols to remove (comma-separated): ").strip()
            if symbols_str:
                symbols = [s.strip().upper() for s in symbols_str.split(',')]
                book = editor.remove_symbols(book, symbols)

        elif choice == '4':
            # Replace symbols
            symbols_str = input("Enter ALL symbols (comma-separated): ").strip()
            if symbols_str:
                symbols = [s.strip().upper() for s in symbols_str.split(',')]
                confirm = input(f"Replace {len(book.symbols)} symbols with {len(symbols)} new symbols? (y/n): ")
                if confirm.lower() == 'y':
                    book = editor.replace_symbols(book, symbols)

        elif choice == '5':
            # Edit parameters
            print(f"\nCurrent parameters:")
            for key, value in sorted(book.strategy_params.items()):
                print(f"  {key} = {value}")

            print("\nEnter parameter updates (format: key=value, comma-separated)")
            print("Example: fast_period=12, slow_period=26")
            params_str = input("Parameters: ").strip()

            if params_str:
                try:
                    params = {}
                    for item in params_str.split(','):
                        key, value = item.split('=')
                        key = key.strip()
                        value = value.strip()

                        # Try to parse as number
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            # Keep as string, handle boolean
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False

                        params[key] = value

                    book = editor.update_parameters(book, params)

                except Exception as e:
                    print(f"Error parsing parameters: {e}")

        elif choice == '6':
            # Edit metadata
            print("\nMetadata options:")
            print("  1. Update description")
            print("  2. Add tags")
            print("  3. Remove tags")
            meta_choice = input("Choose option (1-3): ").strip()

            if meta_choice == '1':
                desc = input("Enter new description: ").strip()
                book = editor.set_description(book, desc)

            elif meta_choice == '2':
                tags_str = input("Enter tags to add (comma-separated): ").strip()
                if tags_str:
                    tags = [t.strip() for t in tags_str.split(',')]
                    book = editor.add_tags(book, tags)

            elif meta_choice == '3':
                print(f"Current tags: {', '.join(book.tags)}")
                tags_str = input("Enter tags to remove (comma-separated): ").strip()
                if tags_str:
                    tags = [t.strip() for t in tags_str.split(',')]
                    book = editor.remove_tags(book, tags)

        elif choice == '7':
            # Show diff
            print(editor.show_diff(original_book, book))

        elif choice == '8':
            # Validate
            is_valid, warnings = editor.validate_book(book)
            print("\n" + "=" * 80)
            print("VALIDATION")
            print("=" * 80)
            if is_valid:
                print("[OK] Book is valid - no issues found")
            else:
                print("[WARNING] Warnings found:")
                for warning in warnings:
                    print(f"  - {warning}")
            print("=" * 80)

        elif choice == '9':
            # Save and exit
            print(editor.show_diff(original_book, book))
            confirm = input("\nSave these changes? (y/n): ")
            if confirm.lower() == 'y':
                editor.save_book(book)
                print(f"\n[OK] Book '{book.name}' saved successfully")
                return book
            else:
                print("Save cancelled")

        elif choice == '0':
            # Exit without saving
            confirm = input("Exit without saving changes? (y/n): ")
            if confirm.lower() == 'y':
                print("Exited without saving")
                return original_book
        else:
            print("Invalid choice")

    return book


def list_all_books(editor: BookEditor):
    """List all available books with details."""
    books = editor.manager.get_all_books()

    if not books:
        print("No books found in saved_books/")
        return

    print("\n" + "=" * 100)
    print("SAVED BOOKS")
    print("=" * 100)
    print(f"{'Name':<30} {'Strategy':<25} {'Symbols':<10} {'Updated':<20}")
    print("-" * 100)

    for book in books:
        strategy = f"{book.strategy_module}.{book.strategy_name}"
        updated = book.updated_at.split('T')[0]  # Just the date
        print(f"{book.name:<30} {strategy:<25} {len(book.symbols):<10} {updated:<20}")

    print("=" * 100)
    print(f"Total books: {len(books)}")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive book editor for BackT strategy configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('book_name', nargs='?', help='Name of the book to edit')
    parser.add_argument('--list', action='store_true', help='List all available books')

    # Quick edit options
    parser.add_argument('--add-symbols', type=str, help='Add symbols (comma-separated)')
    parser.add_argument('--remove-symbols', type=str, help='Remove symbols (comma-separated)')
    parser.add_argument('--replace-symbols', type=str, help='Replace all symbols (comma-separated)')
    parser.add_argument('--params', nargs='+', help='Update parameters (format: key=value)')
    parser.add_argument('--no-preview', action='store_true', help='Skip preview before saving')

    args = parser.parse_args()

    editor = BookEditor()

    # List mode
    if args.list:
        list_all_books(editor)
        return

    # Require book name for editing
    if not args.book_name:
        parser.print_help()
        return

    # Check if book exists
    if not editor.manager.book_exists(args.book_name):
        print(f"Error: Book '{args.book_name}' not found")
        print("\nAvailable books:")
        list_all_books(editor)
        return

    # Quick edit mode
    if args.add_symbols or args.remove_symbols or args.replace_symbols:
        add = [s.strip().upper() for s in args.add_symbols.split(',')] if args.add_symbols else None
        remove = [s.strip().upper() for s in args.remove_symbols.split(',')] if args.remove_symbols else None
        replace = [s.strip().upper() for s in args.replace_symbols.split(',')] if args.replace_symbols else None

        quick_edit_symbols(
            args.book_name,
            add=add,
            remove=remove,
            replace=replace,
            preview=not args.no_preview
        )
        return

    if args.params:
        params = {}
        for param_str in args.params:
            try:
                key, value = param_str.split('=')
                # Try to parse as number
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Handle boolean or keep as string
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False

                params[key.strip()] = value
            except ValueError:
                print(f"Error: Invalid parameter format '{param_str}' (expected key=value)")
                return

        quick_edit_params(args.book_name, params, preview=not args.no_preview)
        return

    # Interactive mode
    book = editor.load_book(args.book_name)
    print(f"\nLoading book: {book.name}")
    print(editor.preview_changes(book))

    interactive_menu(editor, book)


if __name__ == '__main__':
    main()
