"""
Configure Qt for headless testing environments.

This setup:
1. Forces Qt to use 'offscreen' rendering to avoid GUI dependency crashes
2. Provides font fallback paths to prevent missing font warnings
3. Ensures consistent Qt behavior across CI systems and local test runs

Required when testing PyQt/PySide components without a physical display,
common in CI pipelines (GitHub Actions, GitLab CI, etc.) and headless servers.
"""

import os


def pytest_configure(config):
    """Force offscreen mode for all tests"""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
