"""
Book Management System for BackT

A "book" represents a saved strategy configuration that can be reused across backtests.
Books are designed to be modular components that can later be combined into portfolios.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path
from datetime import datetime


@dataclass
class Book:
    """
    Represents a saved strategy configuration.

    A book captures:
    - The strategy being used
    - All strategy parameters
    - The universe of symbols
    - Metadata for organization

    Note: Dates are NOT saved in books - they come from the backtest config.
    This allows you to test the same book across different time periods.
    """
    name: str                           # User-friendly name for the book
    strategy_module: str                # e.g., "MOMENTUM", "AQR"
    strategy_name: str                  # e.g., "ma_crossover_long_only"
    strategy_params: Dict[str, Any]     # Strategy parameters
    symbols: List[str]                  # Universe of symbols
    description: str = ""               # Optional description
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)  # For categorization (e.g., ["momentum", "long-only"])
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extensible metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert book to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Book':
        """Create book from dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        return f"Book(name='{self.name}', strategy='{self.strategy_module}.{self.strategy_name}', symbols={len(self.symbols)})"


class BookManager:
    """
    Manages saving, loading, and organizing books.

    Books are stored as individual JSON files in a books directory.
    This allows for easy version control and manual editing if needed.
    """

    def __init__(self, books_dir: Optional[str] = None):
        """
        Initialize the book manager.

        Args:
            books_dir: Directory to store books. Defaults to ./saved_books/
        """
        if books_dir is None:
            # Default to saved_books in the project root
            books_dir = os.path.join(os.getcwd(), "saved_books")

        self.books_dir = Path(books_dir)
        self.books_dir.mkdir(parents=True, exist_ok=True)

    def save_book(self, book: Book, overwrite: bool = False) -> bool:
        """
        Save a book to disk.

        Args:
            book: The book to save
            overwrite: If False, raises error if book already exists

        Returns:
            True if saved successfully

        Raises:
            FileExistsError: If book exists and overwrite=False
        """
        book.updated_at = datetime.now().isoformat()

        filepath = self._get_book_filepath(book.name)

        if filepath.exists() and not overwrite:
            raise FileExistsError(f"Book '{book.name}' already exists. Set overwrite=True to replace it.")

        with open(filepath, 'w') as f:
            json.dump(book.to_dict(), f, indent=2)

        return True

    def load_book(self, name: str) -> Book:
        """
        Load a book from disk.

        Args:
            name: Name of the book to load

        Returns:
            The loaded book

        Raises:
            FileNotFoundError: If book doesn't exist
        """
        filepath = self._get_book_filepath(name)

        if not filepath.exists():
            raise FileNotFoundError(f"Book '{name}' not found.")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return Book.from_dict(data)

    def list_books(self) -> List[str]:
        """
        Get list of all saved book names.

        Returns:
            List of book names (sorted alphabetically)
        """
        book_files = self.books_dir.glob("*.json")
        names = [f.stem for f in book_files]
        return sorted(names)

    def get_all_books(self) -> List[Book]:
        """
        Load all books from disk.

        Returns:
            List of all books
        """
        books = []
        for name in self.list_books():
            try:
                books.append(self.load_book(name))
            except Exception as e:
                print(f"Warning: Could not load book '{name}': {e}")
        return books

    def delete_book(self, name: str) -> bool:
        """
        Delete a book from disk.

        Args:
            name: Name of the book to delete

        Returns:
            True if deleted successfully

        Raises:
            FileNotFoundError: If book doesn't exist
        """
        filepath = self._get_book_filepath(name)

        if not filepath.exists():
            raise FileNotFoundError(f"Book '{name}' not found.")

        filepath.unlink()
        return True

    def book_exists(self, name: str) -> bool:
        """
        Check if a book exists.

        Args:
            name: Name of the book

        Returns:
            True if book exists
        """
        return self._get_book_filepath(name).exists()

    def rename_book(self, old_name: str, new_name: str) -> bool:
        """
        Rename a book.

        Args:
            old_name: Current name of the book
            new_name: New name for the book

        Returns:
            True if renamed successfully
        """
        book = self.load_book(old_name)
        book.name = new_name
        self.save_book(book)
        self.delete_book(old_name)
        return True

    def get_books_by_tag(self, tag: str) -> List[Book]:
        """
        Get all books with a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            List of books with the tag
        """
        books = self.get_all_books()
        return [b for b in books if tag in b.tags]

    def get_books_by_strategy(self, strategy_name: str) -> List[Book]:
        """
        Get all books using a specific strategy.

        Args:
            strategy_name: The strategy name to filter by

        Returns:
            List of books using the strategy
        """
        books = self.get_all_books()
        return [b for b in books if b.strategy_name == strategy_name]

    def _get_book_filepath(self, name: str) -> Path:
        """Get the filepath for a book."""
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip()
        return self.books_dir / f"{safe_name}.json"


def create_book_from_session(
    name: str,
    strategy_module: str,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    symbols: List[str],
    description: str = "",
    tags: List[str] = None
) -> Book:
    """
    Convenience function to create a book from session data.

    Args:
        name: Book name
        strategy_module: Strategy module (e.g., "MOMENTUM")
        strategy_name: Strategy function name
        strategy_params: Strategy parameters dict
        symbols: List of symbols
        description: Optional description
        tags: Optional tags for categorization

    Returns:
        A new Book instance
    """
    return Book(
        name=name,
        strategy_module=strategy_module,
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        symbols=symbols,
        description=description,
        tags=tags or []
    )
