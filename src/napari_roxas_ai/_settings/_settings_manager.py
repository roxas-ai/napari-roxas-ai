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

        # Get a setting value
        value = settings.get('setting_name', default_value)

        # Set a setting value
        settings.set('setting_name', new_value)
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

        Args:
            key: The setting key to retrieve
            default: Value to return if key doesn't exist

        Returns:
            The setting value or the default if key doesn't exist
        """
        return self._settings.get(key, default)

    def set(self, key: str, value: Any):
        """
        Set a setting value and save to file.

        Args:
            key: The setting key to set
            value: The value to assign to the setting
        """
        self._settings[key] = value
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
            "excluded_path_words": [
                "annotated",
                "Preview",
                "ReferenceSeries",
                "ReferenceSeriesLong",
            ],
            # Metadata settings
            "authorised_sample_types": [
                "conifer",
                "angiosperm",
            ],
            "authorised_sample_geometries": ["linear", "circular"],
            "default_scale": 2.2675,  # Default value: 2.2675 pixels/μm
            "default_angle": 0.0,  # Default value: 0 degrees
            # JPEG compression parameters
            "jpeg_quality": 95,  # JPEG quality (0-100)
            "jpeg_optimize": True,  # Optimize JPEG files
            "jpeg_progressive": True,  # Use progressive JPEG format
            # File extension settings
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
            "roxas_file_extensions": [
                ".scan",
                ".cells",
                ".rings",
                ".metadata",
            ],  # roxas file extensions
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
