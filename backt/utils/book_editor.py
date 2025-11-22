"""
Book Editor - Interactive editing of saved strategy configurations

Provides CLI and programmatic interfaces to edit books:
- Add/remove symbols
- Update strategy parameters
- Modify metadata and tags
- Preview changes before saving
- Backup on edit
"""

from typing import Dict, Any, List, Optional, Tuple
import json
import shutil
from datetime import datetime
from pathlib import Path
from backt.utils.books import Book, BookManager


class BookEditor:
    """
    Interactive editor for Book configurations.

    Features:
    - Edit symbols (add, remove, replace)
    - Edit strategy parameters
    - Edit metadata and tags
    - Preview changes before saving
    - Automatic backups
    - Validation
    """

    def __init__(self, books_dir: Optional[str] = None):
        """
        Initialize the book editor.

        Args:
            books_dir: Directory containing books (defaults to ./saved_books/)
        """
        self.manager = BookManager(books_dir)
        self.backup_dir = Path(self.manager.books_dir) / "_backups"
        self.backup_dir.mkdir(exist_ok=True)

    def load_book(self, name: str) -> Book:
        """
        Load a book for editing.

        Args:
            name: Name of the book

        Returns:
            The loaded book
        """
        return self.manager.load_book(name)

    def save_book(self, book: Book, create_backup: bool = True) -> bool:
        """
        Save an edited book with optional backup.

        Args:
            book: The book to save
            create_backup: If True, creates a backup before saving

        Returns:
            True if saved successfully
        """
        if create_backup and self.manager.book_exists(book.name):
            self._create_backup(book.name)

        book.updated_at = datetime.now().isoformat()
        return self.manager.save_book(book, overwrite=True)

    # Symbol Editing Methods
    # =====================

    def add_symbols(self, book: Book, symbols: List[str]) -> Book:
        """
        Add symbols to a book.

        Args:
            book: The book to edit
            symbols: List of symbols to add

        Returns:
            The modified book
        """
        current_symbols = set(book.symbols)
        new_symbols = [s for s in symbols if s not in current_symbols]

        if new_symbols:
            book.symbols.extend(new_symbols)
            book.symbols.sort()
            print(f"Added {len(new_symbols)} symbols: {', '.join(new_symbols)}")
        else:
            print("No new symbols to add (all already exist)")

        return book

    def remove_symbols(self, book: Book, symbols: List[str]) -> Book:
        """
        Remove symbols from a book.

        Args:
            book: The book to edit
            symbols: List of symbols to remove

        Returns:
            The modified book
        """
        removed = []
        for symbol in symbols:
            if symbol in book.symbols:
                book.symbols.remove(symbol)
                removed.append(symbol)

        if removed:
            print(f"Removed {len(removed)} symbols: {', '.join(removed)}")
        else:
            print("No symbols removed (none were in the book)")

        return book

    def replace_symbols(self, book: Book, new_symbols: List[str]) -> Book:
        """
        Replace all symbols in a book.

        Args:
            book: The book to edit
            new_symbols: New list of symbols

        Returns:
            The modified book
        """
        old_count = len(book.symbols)
        book.symbols = sorted(list(set(new_symbols)))  # Remove duplicates and sort
        print(f"Replaced {old_count} symbols with {len(book.symbols)} symbols")
        return book

    # Parameter Editing Methods
    # ========================

    def update_parameters(self, book: Book, params: Dict[str, Any]) -> Book:
        """
        Update strategy parameters.

        Args:
            book: The book to edit
            params: Dictionary of parameters to update

        Returns:
            The modified book
        """
        updated = []
        for key, value in params.items():
            old_value = book.strategy_params.get(key)
            book.strategy_params[key] = value

            if old_value != value:
                updated.append(f"{key}: {old_value} -> {value}")

        if updated:
            print(f"Updated {len(updated)} parameters:")
            for change in updated:
                print(f"  - {change}")
        else:
            print("No parameters changed")

        return book

    def remove_parameter(self, book: Book, param_name: str) -> Book:
        """
        Remove a parameter from the book.

        Args:
            book: The book to edit
            param_name: Name of the parameter to remove

        Returns:
            The modified book
        """
        if param_name in book.strategy_params:
            old_value = book.strategy_params.pop(param_name)
            print(f"Removed parameter '{param_name}' (was: {old_value})")
        else:
            print(f"Parameter '{param_name}' not found")

        return book

    # Metadata Editing Methods
    # =======================

    def update_metadata(self, book: Book, metadata: Dict[str, Any]) -> Book:
        """
        Update book metadata.

        Args:
            book: The book to edit
            metadata: Dictionary of metadata to update

        Returns:
            The modified book
        """
        book.metadata.update(metadata)
        print(f"Updated {len(metadata)} metadata fields")
        return book

    def add_tags(self, book: Book, tags: List[str]) -> Book:
        """
        Add tags to a book.

        Args:
            book: The book to edit
            tags: List of tags to add

        Returns:
            The modified book
        """
        new_tags = [t for t in tags if t not in book.tags]
        book.tags.extend(new_tags)

        if new_tags:
            print(f"Added {len(new_tags)} tags: {', '.join(new_tags)}")

        return book

    def remove_tags(self, book: Book, tags: List[str]) -> Book:
        """
        Remove tags from a book.

        Args:
            book: The book to edit
            tags: List of tags to remove

        Returns:
            The modified book
        """
        removed = []
        for tag in tags:
            if tag in book.tags:
                book.tags.remove(tag)
                removed.append(tag)

        if removed:
            print(f"Removed {len(removed)} tags: {', '.join(removed)}")

        return book

    def set_description(self, book: Book, description: str) -> Book:
        """
        Set the book description.

        Args:
            book: The book to edit
            description: New description

        Returns:
            The modified book
        """
        book.description = description
        print(f"Updated description")
        return book

    # Utility Methods
    # ==============

    def preview_changes(self, book: Book) -> str:
        """
        Generate a preview of the book's current state.

        Args:
            book: The book to preview

        Returns:
            Formatted string showing book details
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"BOOK PREVIEW: {book.name}")
        lines.append("=" * 80)
        lines.append(f"Strategy:     {book.strategy_module}.{book.strategy_name}")
        lines.append(f"Symbols:      {len(book.symbols)} symbols")
        lines.append(f"              {', '.join(book.symbols[:10])}")
        if len(book.symbols) > 10:
            lines.append(f"              ... and {len(book.symbols) - 10} more")
        lines.append(f"\nParameters:")
        for key, value in sorted(book.strategy_params.items()):
            lines.append(f"  {key:20s} = {value}")
        lines.append(f"\nMetadata:")
        lines.append(f"  Description:  {book.description or '(none)'}")
        lines.append(f"  Tags:         {', '.join(book.tags) if book.tags else '(none)'}")
        lines.append(f"  Created:      {book.created_at}")
        lines.append(f"  Updated:      {book.updated_at}")
        if book.metadata:
            lines.append(f"  Custom metadata:")
            for key, value in sorted(book.metadata.items()):
                if key not in ['ranking_date', 'ranking_period']:  # Skip common fields
                    lines.append(f"    {key}: {value}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def show_diff(self, old_book: Book, new_book: Book) -> str:
        """
        Show differences between two book versions.

        Args:
            old_book: Original book
            new_book: Modified book

        Returns:
            Formatted string showing changes
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CHANGES")
        lines.append("=" * 80)

        # Symbols changes
        old_symbols = set(old_book.symbols)
        new_symbols = set(new_book.symbols)
        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols

        if added or removed:
            lines.append("\nSymbols:")
            if added:
                lines.append(f"  + Added ({len(added)}):   {', '.join(sorted(added))}")
            if removed:
                lines.append(f"  - Removed ({len(removed)}): {', '.join(sorted(removed))}")

        # Parameters changes
        param_changes = []
        all_param_keys = set(old_book.strategy_params.keys()) | set(new_book.strategy_params.keys())
        for key in sorted(all_param_keys):
            old_val = old_book.strategy_params.get(key, "(not set)")
            new_val = new_book.strategy_params.get(key, "(not set)")
            if old_val != new_val:
                param_changes.append(f"  {key}: {old_val} -> {new_val}")

        if param_changes:
            lines.append("\nParameters:")
            lines.extend(param_changes)

        # Metadata changes
        if old_book.description != new_book.description:
            lines.append(f"\nDescription:")
            lines.append(f"  Old: {old_book.description or '(none)'}")
            lines.append(f"  New: {new_book.description or '(none)'}")

        if old_book.tags != new_book.tags:
            lines.append(f"\nTags:")
            lines.append(f"  Old: {', '.join(old_book.tags) if old_book.tags else '(none)'}")
            lines.append(f"  New: {', '.join(new_book.tags) if new_book.tags else '(none)'}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def validate_book(self, book: Book) -> Tuple[bool, List[str]]:
        """
        Validate a book for common issues.

        Args:
            book: The book to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check for empty symbols
        if not book.symbols:
            warnings.append("Book has no symbols")

        # Check for duplicate symbols
        if len(book.symbols) != len(set(book.symbols)):
            duplicates = [s for s in book.symbols if book.symbols.count(s) > 1]
            warnings.append(f"Duplicate symbols found: {', '.join(set(duplicates))}")

        # Check for empty strategy name
        if not book.strategy_name:
            warnings.append("Strategy name is empty")

        # Check for empty parameters
        if not book.strategy_params:
            warnings.append("No strategy parameters defined")

        # Check for very long symbol lists
        if len(book.symbols) > 50:
            warnings.append(f"Large number of symbols ({len(book.symbols)}) may slow down backtests")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    def _create_backup(self, book_name: str) -> Path:
        """
        Create a backup of a book before editing.

        Args:
            book_name: Name of the book to backup

        Returns:
            Path to the backup file
        """
        source = self.manager._get_book_filepath(book_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{book_name}_{timestamp}.json"
        backup_path = self.backup_dir / backup_name

        shutil.copy2(source, backup_path)
        return backup_path

    def list_backups(self, book_name: Optional[str] = None) -> List[Path]:
        """
        List available backups.

        Args:
            book_name: If provided, only list backups for this book

        Returns:
            List of backup file paths
        """
        if book_name:
            pattern = f"{book_name}_*.json"
        else:
            pattern = "*.json"

        backups = sorted(self.backup_dir.glob(pattern), reverse=True)
        return backups

    def restore_backup(self, backup_path: Path) -> Book:
        """
        Restore a book from a backup.

        Args:
            backup_path: Path to the backup file

        Returns:
            The restored book
        """
        with open(backup_path, 'r') as f:
            data = json.load(f)

        book = Book.from_dict(data)
        self.manager.save_book(book, overwrite=True)
        print(f"Restored book '{book.name}' from backup: {backup_path.name}")

        return book


# Convenience functions for quick edits
# ====================================

def quick_edit_symbols(
    book_name: str,
    add: List[str] = None,
    remove: List[str] = None,
    replace: List[str] = None,
    preview: bool = True,
    save: bool = True
) -> Book:
    """
    Quick function to edit symbols in a book.

    Args:
        book_name: Name of the book to edit
        add: Symbols to add
        remove: Symbols to remove
        replace: Replace all symbols with this list
        preview: If True, show preview before saving
        save: If True, save changes to disk

    Returns:
        The edited book

    Example:
        >>> quick_edit_symbols("MACD_Top5_2024", add=["AAPL"], remove=["XLU"])
    """
    editor = BookEditor()
    book = editor.load_book(book_name)

    if replace is not None:
        book = editor.replace_symbols(book, replace)
    else:
        if add:
            book = editor.add_symbols(book, add)
        if remove:
            book = editor.remove_symbols(book, remove)

    if preview:
        print(editor.preview_changes(book))

    if save:
        editor.save_book(book)
        print(f"\n[OK] Book '{book_name}' saved successfully")

    return book


def quick_edit_params(
    book_name: str,
    params: Dict[str, Any],
    preview: bool = True,
    save: bool = True
) -> Book:
    """
    Quick function to edit parameters in a book.

    Args:
        book_name: Name of the book to edit
        params: Dictionary of parameters to update
        preview: If True, show preview before saving
        save: If True, save changes to disk

    Returns:
        The edited book

    Example:
        >>> quick_edit_params("MACD_Top5_2024", {"fast_period": 12, "slow_period": 26})
    """
    editor = BookEditor()
    book = editor.load_book(book_name)
    book = editor.update_parameters(book, params)

    if preview:
        print(editor.preview_changes(book))

    if save:
        editor.save_book(book)
        print(f"\n[OK] Book '{book_name}' saved successfully")

    return book
