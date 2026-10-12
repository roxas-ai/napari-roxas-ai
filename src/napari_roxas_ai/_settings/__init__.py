"""
Plugin settings.

``SettingsWidget`` lives in a submodule and is deliberately not imported here:
this package is imported by a dozen modules that only need the settings values,
and pulling a Qt widget (and with it magicgui, superqt and the napari GUI) into
that path would make reading a setting depend on a running GUI. The widget is
referenced through the lazy import table in the package root instead.
"""

from ._settings_manager import (
    DEFAULT_SETTINGS,
    SettingsManager,
    upgrade_settings,
)

# Expose these classes/functions for importing from the module
__all__ = [
    "SettingsManager",
    "DEFAULT_SETTINGS",
    "upgrade_settings",
]
