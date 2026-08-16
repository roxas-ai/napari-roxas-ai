import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH for default settings. When adding a new setting, add
# it here and nowhere else: at runtime every key missing from a user's
# settings.json is filled in from this dictionary, while the values the user
# already has are left untouched (see SettingsManager._load_settings).
#
# Never mutate this dictionary; hand out deepcopy()s of it.
DEFAULT_SETTINGS: Dict[str, Any] = {
    # Metadata settings with fields for UI
    "samples_metadata": {
        "fields": [
            {
                "id": "sample_name",
                "label": "Name",
                "widget_type": "QLineEdit",
                "read_only": True,
                "required": True,
            },
            # In anticipation of better inclusion of the stem path
            # {
            #     "id": "sample_stem_path",
            #     "label": "Stem Path",
            #     "widget_type": "QLineEdit",
            #     "read_only": True,
            #     "required": True,
            # },
            {
                "id": "sample_type",
                "label": "Type",
                "widget_type": "QComboBox",
                "items": ["conifer", "angiosperm"],
                "editable": True,
                "required": True,
            },
            {
                "id": "meas_geometry",
                "label": "Geometry",
                "widget_type": "QComboBox",
                "items": ["linear", "circular"],
                "editable": True,
                "required": True,
            },
            {
                "id": "spatial_resolution",
                "label": "Spatial Resolution (px/µm)",
                "widget_type": "QDoubleSpinBox",
                "default": 2.2675,
                "min": 0.001,
                "max": 1000.0,
                "step": 0.01,
                "decimals": 4,
                "required": True,
            },
            {
                "id": "rings_outmost_complete_year",
                "label": "Outmost Complete Ring Year",
                "widget_type": "QSpinBox",
                "default": 9999,
                "min": -10000,
                "max": 9999,
                "special_value_text": "Not set",
                "required": True,
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
        "cells_table_file_extension": [
            ".cells_table",
            ".csv",
        ],  # And those of the cells table
        "rings_file_extension": [
            ".rings",
            ".tif",
        ],  # Parts of rings file extension
        "rings_table_file_extension": [
            ".rings_table",
            ".csv",
        ],  # And those of the rings table
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
    # Tabular data settings
    "tables": {
        "index_column": "id",
        "separator": ";",
    },
    # Image processing settings
    "JPEG_compression": {
        "quality": 95,  # Default JPEG quality
        "optimize": True,  # Default optimize flag
        "progressive": False,  # Default progressive flag
    },
    "processing": {
        "try_to_use_gpu": False,  # Try to use GPU if available
        "try_to_use_autocast": False,  # Try to use autocast if available
    },
    "vectorization": {
        "cells_tolerance": 1,  # Default tolerance in pixels for cells vectorization
        "cells_edge_width": 5,  # Default line thickness in pixels for vector shapes visualization
        "cells_edge_color": "blue",  # Default color for vector shapes visualization
        "cells_face_color": "cyan",  # Default color for vector shapes visualization (also used for cells edition in raster mode)
        "rings_tolerance": 5,  # Default tolerance in pixels for rings vectorization
        "rings_edge_width": 5,  # Default line thickness in pixels for vector shapes visualization
        "rings_edge_color": "red",  # Default color for vector shapes visualization
        "rerun_interactive_edge_color": "lime",  # Default color for vector shapes visualization
    },
    "rasterization": {
        "uncomplete_ring_value": -1,
        "uncomplete_ring_color": "red",
        "rings_color_sequence": [
            "blue",
            "green",
            "yellow",
            "purple",
            "orange",
            "cyan",
            "brown",
            "pink",
            "gray",
            "lime",
        ],
        "cells_color": "lime",
    },
    "measurements": {
        "cluster_dbl_cwt_threshold": 3.0,  # Default cluster DBL/CWT threshold in µm
        "cells_smoothing_kernel_size": 5,  # Default smoothing kernel size (1 to disable)
        "relwidth_cwt_integration": 0.75,  # Default wall fraction for thickness measurement
        "cells_tangential_angle": 0.0,  # Default sample angle in degrees (clockwise)
        "lower_limit_cwt_iqr_multiplier": 1.5,  # IQR multiplier for the lower CWT outlier fence
        "upper_limit_cwt_iqr_multiplier": 3.0,  # IQR multiplier for the upper CWT outlier fence
        "opposite_cwt_ratio_limit": 1.5,  # Max CWT ratio between opposite cell sides
        "adjacent_cwt_ratio_limit": 3.0,  # Max CWT ratio between a side and its adjacent sides
    },
    "project_directory": None,  # Current project directory
}


# ---------------------------------------------------------------------------
# Legacy settings migration
# ---------------------------------------------------------------------------
# The settings file lives next to the installed package and survives a
# `pip uninstall`, so a settings.json written by any past version can show up
# under a current install. Settings variable names are not renamed any more,
# but the renames that already happened still have to be applied, otherwise the
# stale names silently keep their old (now unused) meaning.
#
# Entries in "measurements" that were renamed. The user's value is carried over
# to the new name.
LEGACY_MEASUREMENTS_KEY_RENAMES = {
    "cells_cluster_separation_threshold": "cluster_dbl_cwt_threshold",
    "cells_integration_interval": "relwidth_cwt_integration",
}

# Sample metadata field ids that were renamed. These ids end up verbatim as the
# keys of a sample's .metadata.json, so a stale id here is what makes freshly
# prepared samples unreadable by the current code.
LEGACY_METADATA_FIELD_ID_RENAMES = {
    "sample_geometry": "meas_geometry",
    "sample_scale": "spatial_resolution",
}


def _merge_defaults(defaults: Dict[str, Any], stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    Complete `stored` with everything missing from `defaults`.

    Values already present in `stored` always win, so a user never loses a
    setting they changed (e.g. "try_to_use_gpu": true). Keys that only exist in
    `stored` are kept as well. Lists are treated as single values and are never
    merged element-wise, because a user is expected to be able to shorten e.g.
    "rings_color_sequence"; the one list that does get merged is the sample
    metadata field list, handled separately by _merge_metadata_fields().
    """
    merged = deepcopy(stored)
    for key, default_value in defaults.items():
        if key not in merged:
            merged[key] = deepcopy(default_value)
        elif isinstance(default_value, dict) and isinstance(merged[key], dict):
            merged[key] = _merge_defaults(default_value, merged[key])
    return merged


def _merge_metadata_fields(
    default_fields: List[Dict[str, Any]], stored_fields: Any
) -> List[Dict[str, Any]]:
    """
    Merge the sample metadata field definitions by field id.

    New fields shipped with a newer version are inserted at their default
    position, existing fields keep the properties the user set, and fields the
    user added themselves are kept at the end of the list.
    """
    if not isinstance(stored_fields, list):
        return deepcopy(default_fields)

    stored_by_id = {
        field["id"]: field
        for field in stored_fields
        if isinstance(field, dict) and isinstance(field.get("id"), str)
    }

    merged = []
    for default_field in default_fields:
        stored_field = stored_by_id.pop(default_field["id"], None)
        if stored_field is None:
            merged.append(deepcopy(default_field))
        else:
            merged.append(_merge_defaults(default_field, stored_field))

    # Fields that are not part of the defaults are user additions, keep them
    merged.extend(deepcopy(field) for field in stored_by_id.values())
    return merged


def _migrate_legacy_field_ids(fields: List[Any]) -> List[Any]:
    """
    Rename sample metadata fields whose id changed in a past version.

    A rename means the field definition itself changed (label, widget), so the
    current definition is taken as a whole; only the user's chosen `default`
    value is carried over, as it keeps its meaning across the rename.
    """
    defaults_by_id = {
        field["id"]: field
        for field in DEFAULT_SETTINGS["samples_metadata"]["fields"]
    }
    present_ids = {
        field.get("id") for field in fields if isinstance(field, dict)
    }

    migrated = []
    for field in fields:
        if not isinstance(field, dict):
            continue

        new_id = LEGACY_METADATA_FIELD_ID_RENAMES.get(field.get("id"))
        if new_id is None:
            migrated.append(field)
            continue

        if new_id in present_ids:
            # The current field is already there, drop the stale duplicate
            continue

        new_field = deepcopy(
            defaults_by_id.get(new_id, {**field, "id": new_id})
        )
        if "default" in field and "default" in new_field:
            new_field["default"] = field["default"]
        migrated.append(new_field)

    return migrated


def _migrate_legacy_settings(stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply every rename that happened in a past version to a loaded settings
    dictionary, carrying the user's values over to the current names.
    """
    migrated = deepcopy(stored)

    measurements = migrated.get("measurements")
    if isinstance(measurements, dict):
        for old_key, new_key in LEGACY_MEASUREMENTS_KEY_RENAMES.items():
            if old_key in measurements:
                value = measurements.pop(old_key)
                measurements.setdefault(new_key, value)

    samples_metadata = migrated.get("samples_metadata")
    if isinstance(samples_metadata, dict) and isinstance(
        samples_metadata.get("fields"), list
    ):
        samples_metadata["fields"] = _migrate_legacy_field_ids(
            samples_metadata["fields"]
        )

    return migrated


def upgrade_settings(stored: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bring a settings dictionary loaded from disk up to date.

    Renames legacy keys, then fills in everything that is missing from the
    defaults. Values the user already has are never overwritten.

    Parameters
    ----------
    stored : dict
        Settings as loaded from a settings.json file.

    Returns
    -------
    dict
        A new, complete settings dictionary.
    """
    migrated = _migrate_legacy_settings(stored)
    merged = _merge_defaults(DEFAULT_SETTINGS, migrated)

    # The field list is a list of dicts and needs to be merged by field id
    merged["samples_metadata"]["fields"] = _merge_metadata_fields(
        DEFAULT_SETTINGS["samples_metadata"]["fields"],
        merged["samples_metadata"].get("fields"),
    )

    return merged


class SettingsManager:
    """
    Manages plugin settings, including loading, saving, and accessing settings.

    This class implements the Singleton pattern to ensure only one instance
    exists throughout the application. This guarantees that settings are
    consistent across all components of the plugin.

    The settings are stored in a JSON file located in the _settings module
    directory. The file is automatically created with default values if it
    doesn't exist, and an existing file is upgraded in place on load: legacy
    key names are renamed and settings added by a newer version are filled in
    from the defaults, while values the user changed are kept.

    Usage:
        # Get the settings manager instance
        settings = SettingsManager()

        # Get a setting value (supports dot notation for nested settings)
        value = settings.get('samples_metadata.fields', default_value)

        # Set a setting value (supports dot notation for nested settings)
        settings.set('processing.try_to_use_gpu', True)
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
            settings_dir = Path(__file__).parent.absolute()
            self._settings_file = settings_dir / "settings.json"
        return self._settings_file

    def _load_settings(self):
        """
        Load settings from the JSON file.

        If the file exists, its content is upgraded to the current schema
        (legacy keys renamed, missing keys filled in from the defaults) and
        written back if anything changed, so that a settings.json left behind
        by an older install keeps working without losing user values.
        If the file doesn't exist or is corrupted, creates default settings.
        """
        if not self.settings_file.exists():
            # Create default settings if file doesn't exist
            self.reset()
            return

        try:
            with open(self.settings_file, encoding="utf-8") as f:
                stored = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # If file is corrupted, use default settings
            self.reset()
            return

        if not isinstance(stored, dict):
            self.reset()
            return

        self._settings = upgrade_settings(stored)

        # Only touch the file when the upgrade actually changed something
        if self._settings != stored:
            self.save_settings()

    def save_settings(self):
        """
        Save current settings to the JSON file.

        Writes the settings dictionary to the settings file with pretty formatting.
        """
        with open(self.settings_file, "w", encoding="utf-8") as f:
            json.dump(self._settings, f, indent=4)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a setting value by key.
        Supports nested settings using dot notation (e.g., 'processing.try_to_use_gpu')

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
        Supports nested settings using dot notation (e.g., 'processing.try_to_use_gpu')

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

        IMPORTANT for developers:
        New default settings belong in DEFAULT_SETTINGS at the top of this
        module, which is the single source of truth. At runtime, any keys
        missing from the user's settings.json are populated from there by
        _load_settings().

        Args:
            settings_dict: Dictionary of settings to update
        """
        self._settings.update(settings_dict)
        self.save_settings()

    def reset(self):
        """
        Reset all settings to default values and save to file.
        """
        self._settings = deepcopy(DEFAULT_SETTINGS)
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
        # Windows - use Path.open() instead of os.startfile
        import webbrowser

        webbrowser.open(str(settings_file))
    elif sys.platform == "darwin":
        # macOS
        subprocess.call(["open", str(settings_file)])
    else:
        # Linux and other UNIX-like systems
        subprocess.call(["xdg-open", str(settings_file)])

    return settings_file
