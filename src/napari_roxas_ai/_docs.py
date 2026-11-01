"""
Helper for opening the online ROXAS AI user documentation.
"""

import webbrowser
from napari.utils.notifications import show_info

DOCS_URL = "https://roxas-ai.github.io/napari-roxas-ai/"


def open_user_documentation() -> None:
    """Open the online user documentation in the default web browser."""
    show_info("Opening ROXAS AI documentation in your browser...")
    webbrowser.open(DOCS_URL)
