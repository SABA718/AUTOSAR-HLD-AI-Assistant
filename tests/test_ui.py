"""Streamlit App Test using streamlit.testing.v1 (Phase 7).

Tests verify:
1. Streamlit app initializes cleanly without exceptions.
2. Sidebar controls and navigation radio button are present.
3. Views render expected titles and components.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


class TestStreamlitApp:
    """UI integration tests using Streamlit AppTest."""

    def test_app_initialization_and_sidebar(self) -> None:
        """Verify app runs and sidebar displays navigation options."""
        app_path = Path(__file__).parent.parent / "frontend" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # Check there are no unhandled exceptions in the script
        assert not at.exception

        # Verify title and sidebar elements
        assert len(at.sidebar.radio) >= 1
        nav_radio = at.sidebar.radio[0]
        assert "Document Upload" in nav_radio.options
        assert "Engineering Q&A" in nav_radio.options
        assert "Architecture Explorer" in nav_radio.options

    def test_navigate_to_qa_view(self) -> None:
        """Verify navigation to Q&A view renders question input."""
        app_path = Path(__file__).parent.parent / "frontend" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # Select Engineering Q&A
        at.sidebar.radio[0].set_value("Engineering Q&A").run()
        assert not at.exception
        # Check text area for entering question exists
        assert len(at.text_area) >= 1

    def test_navigate_to_explorer_view(self) -> None:
        """Verify navigation to Architecture Explorer renders document input."""
        app_path = Path(__file__).parent.parent / "frontend" / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()

        # Select Architecture Explorer
        at.sidebar.radio[0].set_value("Architecture Explorer").run()
        assert not at.exception
        # Check text input for document id
        assert len(at.text_input) >= 1
