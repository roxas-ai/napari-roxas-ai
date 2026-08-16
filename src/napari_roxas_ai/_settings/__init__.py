from ._settings_manager import (
    DEFAULT_SETTINGS,
    SettingsManager,
    open_settings_file,
    upgrade_settings,
)

# Expose these classes/functions for importing from the module
__all__ = [
    "SettingsManager",
    "open_settings_file",
    "DEFAULT_SETTINGS",
    "upgrade_settings",
]
