import json
import os
from pathlib import Path
from typing import Any, Dict


class SettingsManager:
    """
    Manages plugin settings, including loading, saving, and accessing settings.

    This class implements the Singleton pattern to ensure only one instance
    exists throughout the application. This guarantees that settings are
    consistent across all components of the plugin.

    The settings are stored in a JSON file located in the _settings module
    directory. The file is automatically created with default values if it
    doesn't exist.

    Usage:
        # Get the settings manager instance
        settings = SettingsManager()

        # Get a setting value (supports dot notation for nested settings)
        value = settings.get('samples_metadata.default_scale', default_value)

        # Set a setting value (supports dot notation for nested settings)
        settings.set('samples_metadata.default_scale', new_value)
    """

    # Class variables for Singleton implementation
    _instance = None  # Stores the single instance
    _settings = None  # Stores the loaded settings
    _settings_file = None  # Stores the path to settings file

    def __new__(cls):
        """
        Singleton implementation: ensures only one instance of SettingsManager exists.
        Returns the existing instance if already created, or creates a new one.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initialize the settings manager by loading settings from file.
        Called only once when the singleton instance is first created.
        """
        self._settings = {}
        self._load_settings()

    @property
    def settings_file(self) -> Path:
        """
        Get the path to the settings file.

        Returns:
            Path: The full path to the settings.json file
        """
        if self._settings_file is None:
            # Use the _settings module directory to store the settings file
            settings_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self._settings_file = settings_dir / "settings.json"
        return self._settings_file

    def _load_settings(self):
        """
        Load settings from the JSON file.

        If the file exists, loads settings from it.
        If the file doesn't exist or is corrupted, creates default settings.
        """
        if self.settings_file.exists():
            try:
                # Try to load existing settings
                with open(self.settings_file) as f:
                    self._settings = json.load(f)
            except json.JSONDecodeError:
                # If file is corrupted, use default settings
                self.reset()
        else:
            # Create default settings if file doesn't exist
            self.reset()

    def save_settings(self):
        """
        Save current settings to the JSON file.

        Writes the settings dictionary to the settings file with pretty formatting.
        """
        with open(self.settings_file, "w") as f:
            json.dump(self._settings, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.
        Supports nested settings using dot notation (e.g., 'samples_metadata.default_scale')

        Args:
            key: The setting key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The setting value or the default if key doesn't exist
        """
        if "." not in key:
            return self._settings.get(key, default)

        # Handle nested keys
        keys = key.split(".")
        current = self._settings

        for k in keys[:-1]:
            if k not in current:
                return default
            current = current[k]

        if not isinstance(current, dict) or keys[-1] not in current:
            return default

        return current.get(keys[-1], default)

    def set(self, key: str, value: Any):
        """
        Set a setting value and save to file.
        Supports nested settings using dot notation (e.g., 'samples_metadata.default_scale')

        Args:
            key: The setting key to set
            value: The value to assign to the setting
        """
        if "." not in key:
            self._settings[key] = value
            self.save_settings()
            return

        # Handle nested keys
        keys = key.split(".")
        current = self._settings

        # Navigate to the correct nested dictionary
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the value
        current[keys[-1]] = value
        self.save_settings()

    def update(self, settings_dict: Dict[str, Any]):
        """
        Update multiple settings at once and save to file.

        Args:
            settings_dict: Dictionary of settings to update
        """
        self._settings.update(settings_dict)
        self.save_settings()

    def reset(self):
        """
        Reset all settings to default values and save to file.
        """
        self._settings = {
            # Metadata settings with fields for UI
            "samples_metadata": {
                "fields": [
                    {
                        "id": "sample_name",
                        "label": "Sample Name",
                        "widget_type": "QLineEdit",
                        "read_only": True,
                        "required": True,
                    },
                    {
                        "id": "sample_type",
                        "label": "Sample Type",
                        "widget_type": "QComboBox",
                        "items": ["conifer", "angiosperm"],
                        "editable": True,
                        "required": True,
                    },
                    {
                        "id": "sample_geometry",
                        "label": "Sample Geometry",
                        "widget_type": "QComboBox",
                        "items": ["linear", "circular"],
                        "editable": True,
                        "required": True,
                    },
                    {
                        "id": "sample_scale",
                        "label": "Sample Scale",
                        "widget_type": "QDoubleSpinBox",
                        "default": 2.2675,
                        "min": 0.001,
                        "max": 1000.0,
                        "step": 0.01,
                        "decimals": 4,
                        "required": True,
                    },
                    {
                        "id": "sample_angle",
                        "label": "Sample Angle",
                        "widget_type": "QDoubleSpinBox",
                        "default": 0.0,
                        "min": -360.0,
                        "max": 360.0,
                        "step": 0.1,
                        "decimals": 2,
                        "required": True,
                    },
                    {
                        "id": "sample_outmost_complete_ring_year",
                        "label": "Sample Outmost Complete Ring Year",
                        "widget_type": "QSpinBox",
                        "default": 9999,
                        "min": -10000,
                        "max": 9999,
                        "special_value_text": "Not set",
                        "required": False,
                    },
                ]
            },
            # File extension settings
            "file_extensions": {
                "scan_file_extension": [
                    ".scan",
                    ".jpg",
                ],  # Parts of scan file extension
                "metadata_file_extension": [
                    ".metadata",
                    ".json",
                ],  # Parts of metadata file extension
                "cells_file_extension": [
                    ".cells",
                    ".png",
                ],  # Parts of cells file extension
                "rings_file_extension": [
                    ".rings",
                    ".tif",
                ],  # Parts of rings file extension
                "crossdating_file_extension": [
                    ".crossdating",
                    ".txt",
                ],  # Parts of tucson file extension
                "roxas_file_extensions": [
                    ".scan",
                    ".cells",
                    ".rings",
                    ".metadata",
                ],  # roxas file extensions
                "image_file_extensions": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".tif",
                    ".tiff",
                    ".bmp",
                    ".jp2",
                ],  # Supported image file extensions
                "text_file_extensions": [
                    ".rwl",  # for tucson files
                    ".tuc",  # for tucson files
                    ".txt",
                    ".csv",
                    ".tsv",
                ],  # Supported text file extensions
            },
            # Image processing settings
            "JPEG_compression": {
                "quality": 95,  # Default JPEG quality
                "optimize": True,  # Default optimize flag
                "progressive": False,  # Default progressive flag
            },
        }
        self.save_settings()


def open_settings_file():
    """
    Opens the settings file in the system's default text editor.

    This function is used as an entry point for the plugin menu item.
    It ensures the settings file exists and then opens it using the
    appropriate system command based on the user's operating system.

    Returns:
        Path: The path to the settings file that was opened
    """
    # Get the settings file path
    settings_manager = SettingsManager()
    settings_file = settings_manager.settings_file

    # Ensure the file exists
    if not settings_file.exists():
        settings_manager.save_settings()

    # Open the file with the system's default application based on OS
    import subprocess
    import sys

    if sys.platform == "win32":
        # Windows
        os.startfile(settings_file)
    elif sys.platform == "darwin":
        # macOS
        subprocess.call(["open", settings_file])
    else:
        # Linux and other UNIX-like systems
        subprocess.call(["xdg-open", settings_file])

    return settings_file
