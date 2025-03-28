"""
Configure Qt for headless testing environments.
"""

import os


def pytest_configure(config):
    """Force offscreen mode for headless Qt tests"""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
