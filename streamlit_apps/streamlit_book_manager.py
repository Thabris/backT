"""
Streamlit Book Manager - Comprehensive book editing interface

Launch with:
    streamlit run streamlit_book_manager.py
"""

import sys
from pathlib import Path
# Add parent directory (project root) to path to find backt module
project_root = Path(r"C:\Users\maxim\Documents\Projects\backtester2")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from datetime import datetime
from backt.utils import BookEditor, BookManager, Book
from typing import Optional


# Page config
st.set_page_config(
    page_title="Book Manager - BackT",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'selected_book' not in st.session_state:
        st.session_state.selected_book = None
    if 'book_modified' not in st.session_state:
        st.session_state.book_modified = False
    if 'original_book' not in st.session_state:
        st.session_state.original_book = None


def load_book(editor: BookEditor, book_name: str):
    """Load a book into session state."""
    book = editor.load_book(book_name)
    st.session_state.selected_book = book
    st.session_state.original_book = Book.from_dict(book.to_dict())
    st.session_state.book_modified = False


def render_book_list(editor: BookEditor):
    """Render the book list sidebar."""
    st.sidebar.title("📚 Book Manager")
    st.sidebar.caption("Manage your saved strategy configurations")

    # Get all books
    books = editor.manager.get_all_books()

    if not books:
        st.sidebar.info("No books found. Create books using the backtesting interface or CLI tools.")
        return None

    # Create a dataframe for better display
    book_data = []
    for book in books:
        book_data.append({
            "Name": book.name,
            "Strategy": f"{book.strategy_module}.{book.strategy_name}",
            "Symbols": len(book.symbols),
            "Updated": book.updated_at[:10]
        })

    df = pd.DataFrame(book_data)

    # Display summary
    st.sidebar.metric("Total Books", len(books))

    # Filter options
    st.sidebar.subheader("Filter Books")

    # Filter by strategy
    all_strategies = sorted(list(set([b.strategy_name for b in books])))
    selected_strategy = st.sidebar.selectbox(
        "Strategy",
        ["All"] + all_strategies,
        key="strategy_filter"
    )

    # Filter by tag
    all_tags = sorted(list(set([tag for b in books for tag in b.tags])))
    selected_tag = st.sidebar.selectbox(
        "Tag",
        ["All"] + all_tags,
        key="tag_filter"
    )

    # Apply filters
    filtered_books = books
    if selected_strategy != "All":
        filtered_books = [b for b in filtered_books if b.strategy_name == selected_strategy]
    if selected_tag != "All":
        filtered_books = [b for b in filtered_books if selected_tag in b.tags]

    # Book selection
    st.sidebar.subheader("Select Book")
    book_names = [b.name for b in filtered_books]

    if not book_names:
        st.sidebar.warning("No books match the filter criteria")
        return None

    selected_name = st.sidebar.selectbox(
        "Book Name",
        book_names,
        key="book_selector"
    )

    # Load button
    if st.sidebar.button("📖 Load Book", use_container_width=True, type="primary"):
        load_book(editor, selected_name)

    # Quick actions
    st.sidebar.subheader("Quick Actions")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("➕ New", use_container_width=True):
            st.info("Use the backtesting interface to create new books")

    return filtered_books


def render_symbol_editor(book: Book):
    """Render symbol editing interface."""
    st.subheader("📋 Symbols")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Display current symbols
        symbols_text = st.text_area(
            "Symbol List (comma-separated)",
            value=", ".join(book.symbols),
            height=150,
            help="Edit symbols by adding or removing from this list",
            key="symbols_editor"
        )

        # Parse edited symbols
        new_symbols = [s.strip().upper() for s in symbols_text.split(',') if s.strip()]
        new_symbols = sorted(list(set(new_symbols)))  # Remove duplicates and sort

    with col2:
        st.metric("Total Symbols", len(new_symbols))

        # Quick add
        add_symbol = st.text_input("Quick Add", placeholder="SPY", key="quick_add_symbol")
        if st.button("➕ Add", use_container_width=True, key="add_symbol_btn"):
            if add_symbol and add_symbol.upper() not in new_symbols:
                new_symbols.append(add_symbol.upper())
                new_symbols.sort()
                book.symbols = new_symbols
                st.session_state.book_modified = True
                st.rerun()

        # Quick remove
        if book.symbols:
            remove_symbol = st.selectbox("Quick Remove", book.symbols, key="quick_remove_symbol")
            if st.button("➖ Remove", use_container_width=True, key="remove_symbol_btn"):
                if remove_symbol in new_symbols:
                    new_symbols.remove(remove_symbol)
                    book.symbols = new_symbols
                    st.session_state.book_modified = True
                    st.rerun()

    # Check if changed
    if set(new_symbols) != set(book.symbols):
        st.warning(f"⚠️ Symbols changed: {len(book.symbols)} → {len(new_symbols)}")

        # Show diff
        with st.expander("View Changes"):
            col1, col2 = st.columns(2)
            with col1:
                st.caption("**Original**")
                st.code(", ".join(book.symbols), language=None)
            with col2:
                st.caption("**New**")
                st.code(", ".join(new_symbols), language=None)

            added = set(new_symbols) - set(book.symbols)
            removed = set(book.symbols) - set(new_symbols)

            if added:
                st.success(f"➕ Added: {', '.join(sorted(added))}")
            if removed:
                st.error(f"➖ Removed: {', '.join(sorted(removed))}")

        if st.button("✅ Apply Symbol Changes", type="primary"):
            book.symbols = new_symbols
            st.session_state.book_modified = True
            st.success("Symbols updated (not saved to disk yet)")
            st.rerun()


def render_parameter_editor(book: Book):
    """Render parameter editing interface."""
    st.subheader("⚙️ Strategy Parameters")

    params = book.strategy_params.copy()

    # Create editable form
    with st.form("param_editor_form"):
        st.caption(f"**Strategy:** {book.strategy_module}.{book.strategy_name}")

        updated_params = {}
        param_changed = False

        # Display each parameter with appropriate input widget
        for key, value in sorted(params.items()):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.text(key)

            with col2:
                # Choose widget based on type
                if isinstance(value, bool):
                    new_value = st.checkbox(
                        "Value",
                        value=value,
                        key=f"param_{key}",
                        label_visibility="collapsed"
                    )
                elif isinstance(value, int):
                    new_value = st.number_input(
                        "Value",
                        value=value,
                        step=1,
                        key=f"param_{key}",
                        label_visibility="collapsed"
                    )
                elif isinstance(value, float):
                    new_value = st.number_input(
                        "Value",
                        value=value,
                        step=0.01,
                        format="%.4f",
                        key=f"param_{key}",
                        label_visibility="collapsed"
                    )
                else:
                    new_value = st.text_input(
                        "Value",
                        value=str(value),
                        key=f"param_{key}",
                        label_visibility="collapsed"
                    )

                updated_params[key] = new_value

                if new_value != value:
                    param_changed = True

            with col3:
                if new_value != value:
                    st.caption(f"✏️ {value} → {new_value}")

        # Add new parameter
        st.divider()
        st.caption("**Add New Parameter**")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            new_param_key = st.text_input("Parameter Name", key="new_param_key")
        with col2:
            new_param_value = st.text_input("Value", key="new_param_value")
        with col3:
            new_param_type = st.selectbox(
                "Type",
                ["str", "int", "float", "bool"],
                key="new_param_type",
                label_visibility="collapsed"
            )

        # Submit button
        submitted = st.form_submit_button("💾 Apply Parameter Changes", type="primary")

        if submitted:
            # Apply changes
            if new_param_key and new_param_value:
                # Add new parameter
                if new_param_type == "int":
                    updated_params[new_param_key] = int(new_param_value)
                elif new_param_type == "float":
                    updated_params[new_param_key] = float(new_param_value)
                elif new_param_type == "bool":
                    updated_params[new_param_key] = new_param_value.lower() in ['true', '1', 'yes']
                else:
                    updated_params[new_param_key] = new_param_value

            book.strategy_params = updated_params
            st.session_state.book_modified = True
            st.success("Parameters updated (not saved to disk yet)")
            st.rerun()


def render_metadata_editor(book: Book):
    """Render metadata editing interface."""
    st.subheader("📝 Metadata")

    # Description
    new_description = st.text_area(
        "Description",
        value=book.description,
        height=100,
        help="Describe this book's purpose or strategy",
        key="description_editor"
    )

    if new_description != book.description:
        if st.button("💾 Update Description", type="primary"):
            book.description = new_description
            st.session_state.book_modified = True
            st.success("Description updated")
            st.rerun()

    # Tags
    st.caption("**Tags**")
    col1, col2 = st.columns([3, 1])

    with col1:
        tags_text = st.text_input(
            "Tags (comma-separated)",
            value=", ".join(book.tags),
            help="Add tags for categorization",
            key="tags_editor"
        )
        new_tags = [t.strip() for t in tags_text.split(',') if t.strip()]

    with col2:
        if set(new_tags) != set(book.tags):
            if st.button("💾 Update Tags", type="primary"):
                book.tags = new_tags
                st.session_state.book_modified = True
                st.success("Tags updated")
                st.rerun()

    # Created/Updated info
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"**Created:** {book.created_at[:19]}")
    with col2:
        st.caption(f"**Updated:** {book.updated_at[:19]}")


def render_book_overview(book: Book, original: Optional[Book]):
    """Render book overview and save controls."""
    st.title(f"📚 {book.name}")

    # Status badge
    if st.session_state.book_modified:
        st.warning("⚠️ Unsaved changes - remember to save!")
    else:
        st.success("✅ No unsaved changes")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Strategy", book.strategy_name)
    with col2:
        st.metric("Symbols", len(book.symbols))
    with col3:
        st.metric("Parameters", len(book.strategy_params))
    with col4:
        st.metric("Tags", len(book.tags))

    # Save controls
    st.divider()
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

    with col1:
        if st.button("💾 Save Changes", disabled=not st.session_state.book_modified, type="primary", use_container_width=True):
            editor = BookEditor()
            editor.save_book(book, create_backup=True)
            st.session_state.book_modified = False
            st.session_state.original_book = Book.from_dict(book.to_dict())
            st.success(f"✅ Book '{book.name}' saved (backup created)")
            st.rerun()

    with col2:
        if st.button("↩️ Revert Changes", disabled=not st.session_state.book_modified, use_container_width=True):
            st.session_state.selected_book = Book.from_dict(original.to_dict())
            st.session_state.book_modified = False
            st.info("Changes reverted")
            st.rerun()

    with col3:
        if st.button("🔍 Validate", use_container_width=True):
            editor = BookEditor()
            is_valid, warnings = editor.validate_book(book)

            if is_valid:
                st.success("✅ Book is valid - no issues found")
            else:
                st.warning(f"⚠️ {len(warnings)} warnings found:")
                for warning in warnings:
                    st.caption(f"  • {warning}")

    with col4:
        if st.button("🗑️ Delete Book", use_container_width=True, type="secondary"):
            st.session_state.show_delete_confirm = True

    # Delete confirmation
    if st.session_state.get('show_delete_confirm', False):
        st.error(f"⚠️ **Delete '{book.name}'?** This cannot be undone!")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm Delete", type="primary", use_container_width=True):
                editor = BookEditor()
                editor.manager.delete_book(book.name)
                st.success(f"Book '{book.name}' deleted")
                st.session_state.selected_book = None
                st.session_state.show_delete_confirm = False
                st.rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_delete_confirm = False
                st.rerun()

    # Show diff if modified
    if st.session_state.book_modified and original:
        with st.expander("📊 View All Changes", expanded=False):
            editor = BookEditor()
            diff = editor.show_diff(original, book)
            st.code(diff, language=None)


def render_backup_manager():
    """Render backup management interface."""
    st.subheader("🔄 Backups")

    editor = BookEditor()
    all_backups = editor.list_backups()

    if not all_backups:
        st.info("No backups found")
        return

    # Group backups by book name
    backup_groups = {}
    for backup in all_backups:
        # Extract book name from filename (format: BookName_YYYYMMDD_HHMMSS.json)
        parts = backup.stem.split('_')
        if len(parts) >= 3:
            book_name = '_'.join(parts[:-2])
            timestamp = '_'.join(parts[-2:])

            if book_name not in backup_groups:
                backup_groups[book_name] = []
            backup_groups[book_name].append({
                'path': backup,
                'timestamp': timestamp,
                'date': datetime.strptime(timestamp, '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            })

    # Display backups
    for book_name, backups in sorted(backup_groups.items()):
        with st.expander(f"📚 {book_name} ({len(backups)} backups)"):
            for backup in sorted(backups, key=lambda x: x['timestamp'], reverse=True):
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.caption(f"**{backup['date']}**")
                with col2:
                    st.caption(f"`{backup['path'].name}`")
                with col3:
                    if st.button("♻️ Restore", key=f"restore_{backup['path'].stem}"):
                        restored = editor.restore_backup(backup['path'])
                        st.success(f"Restored '{restored.name}' from backup")
                        st.rerun()


def main():
    """Main application."""
    init_session_state()

    editor = BookEditor()

    # Sidebar - Book list
    books = render_book_list(editor)

    # Main area
    if st.session_state.selected_book is None:
        st.title("📚 Book Manager")
        st.info("👈 Select a book from the sidebar to start editing")

        # Show overview
        if books:
            st.subheader("Available Books")
            book_data = [{
                "Name": b.name,
                "Strategy": b.strategy_name,
                "Symbols": len(b.symbols),
                "Params": len(b.strategy_params),
                "Tags": ", ".join(b.tags) if b.tags else "-",
                "Updated": b.updated_at[:10]
            } for b in books]
            st.dataframe(pd.DataFrame(book_data), use_container_width=True, hide_index=True)

        # Backup manager
        st.divider()
        render_backup_manager()

    else:
        book = st.session_state.selected_book
        original = st.session_state.original_book

        # Book overview and controls
        render_book_overview(book, original)

        # Tabs for different editors
        tab1, tab2, tab3 = st.tabs(["📋 Symbols", "⚙️ Parameters", "📝 Metadata"])

        with tab1:
            render_symbol_editor(book)

        with tab2:
            render_parameter_editor(book)

        with tab3:
            render_metadata_editor(book)


if __name__ == "__main__":
    main()
