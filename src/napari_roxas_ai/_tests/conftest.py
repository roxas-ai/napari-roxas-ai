"""
Configure Qt for headless testing environments.
"""

import os
import sys


def pytest_configure(config):
    """Force offscreen mode for headless Qt tests"""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Apply macOS-specific fixes without affecting Linux or Windows
    if sys.platform == "darwin":
        os.environ["QT_QPA_FONTDIR"] = "/System/Library/Fonts/Supplemental/"
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = (
            "/usr/local/Cellar/qt/6.*/plugins/platforms/"
        )
        os.environ["QT_DEBUG_PLUGINS"] = "1"  # Debugging (optional)
