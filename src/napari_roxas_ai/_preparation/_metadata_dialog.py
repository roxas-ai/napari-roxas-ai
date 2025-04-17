"""
Dialog for entering sample metadata.
"""

from typing import Dict, Optional, Tuple

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .._settings._settings_manager import SettingsManager


class MetadataDialog(QDialog):
    """
    Dialog for entering sample metadata.

    This dialog allows the user to edit metadata for a sample and
    optionally apply the same metadata to all remaining samples.
    """

    def __init__(self, filename: str, parent: Optional[QWidget] = None):
        """
        Initialize the metadata dialog.

        Args:
            filename: Base filename for the sample (without extension)
            parent: Parent widget
        """
        super().__init__(parent)

        self.setWindowTitle("Sample Metadata")
        self.resize(400, 300)

        # Store the filename
        self.filename = filename

        # Load settings
        self._load_settings()

        # Create layout and widgets
        self._create_ui()

        # Apply default values
        self._apply_defaults()

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Get authorized sample types and geometries
        self.sample_types = (
            self.settings_manager.get(
                "samples_metadata.authorised_sample_types"
            )
            or []
        )
        self.sample_geometries = (
            self.settings_manager.get(
                "samples_metadata.authorised_sample_geometries"
            )
            or []
        )

        # Get default scale and angle
        self.default_scale = (
            self.settings_manager.get("samples_metadata.default_scale") or 1.0
        )
        self.default_angle = (
            self.settings_manager.get("samples_metadata.default_angle") or 0.0
        )

        # Get default outmost year
        self.default_outmost_year = self.settings_manager.get(
            "samples_metadata.default_outmost_year"
        )

        # Get file extensions
        scan_file_extension = self.settings_manager.get(
            "file_extensions.scan_file_extension"
        )
        self.scan_content_extension = (
            scan_file_extension[0] if scan_file_extension else ".scan"
        )

        metadata_file_extension_parts = self.settings_manager.get(
            "file_extensions.metadata_file_extension"
        )
        self.metadata_file_extension = (
            "".join(metadata_file_extension_parts) or ".metadata.json"
        )

    def _create_ui(self):
        """Create the UI elements."""
        # Main layout
        layout = QVBoxLayout(self)

        # Form layout for metadata fields
        form_layout = QFormLayout()

        # Sample name (read-only)
        self.sample_name_edit = QLineEdit()
        self.sample_name_edit.setReadOnly(True)
        form_layout.addRow("Sample Name:", self.sample_name_edit)

        # Sample type (combo box)
        self.sample_type_combo = QComboBox()
        if self.sample_types:
            self.sample_type_combo.addItems(self.sample_types)
        else:
            self.sample_type_combo.setEditable(True)
        form_layout.addRow("Sample Type:", self.sample_type_combo)

        # Sample geometry (combo box)
        self.sample_geometry_combo = QComboBox()
        if self.sample_geometries:
            self.sample_geometry_combo.addItems(self.sample_geometries)
        else:
            self.sample_geometry_combo.setEditable(True)
        form_layout.addRow("Sample Geometry:", self.sample_geometry_combo)

        # Sample scale (double spin box)
        self.sample_scale_spin = QDoubleSpinBox()
        self.sample_scale_spin.setRange(0.001, 1000.0)
        self.sample_scale_spin.setSingleStep(0.01)
        self.sample_scale_spin.setDecimals(4)
        form_layout.addRow("Sample Scale:", self.sample_scale_spin)

        # Sample angle (double spin box)
        self.sample_angle_spin = QDoubleSpinBox()
        self.sample_angle_spin.setRange(-360.0, 360.0)
        self.sample_angle_spin.setSingleStep(0.1)
        self.sample_angle_spin.setDecimals(2)
        form_layout.addRow("Sample Angle:", self.sample_angle_spin)

        # Sample outmost year (spin box)
        self.sample_outmost_year_spin = QSpinBox()
        self.sample_outmost_year_spin.setRange(
            -10000, 3000
        )  # Wide range for historical data
        self.sample_outmost_year_spin.setSpecialValueText(
            "Not set"
        )  # Display "Not set" for value 0
        form_layout.addRow(
            "Sample Outmost Year:", self.sample_outmost_year_spin
        )

        # Add form layout to main layout
        layout.addLayout(form_layout)

        # "Apply to all" checkbox
        self.apply_to_all_checkbox = QCheckBox(
            "Apply to all remaining samples"
        )
        layout.addWidget(self.apply_to_all_checkbox)

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _apply_defaults(self):
        """Apply default values to the widgets."""
        # Sample name
        self.sample_name_edit.setText(self.filename)

        # Sample type
        if self.sample_types:
            self.sample_type_combo.setCurrentIndex(0)
        elif self.sample_type_combo.isEditable():
            self.sample_type_combo.setEditText("")

        # Sample geometry
        if self.sample_geometries:
            self.sample_geometry_combo.setCurrentIndex(0)
        elif self.sample_geometry_combo.isEditable():
            self.sample_geometry_combo.setEditText("")

        # Sample scale
        self.sample_scale_spin.setValue(self.default_scale)

        # Sample angle
        self.sample_angle_spin.setValue(self.default_angle)

        # Sample outmost year
        if (
            self.default_outmost_year is not None
            and self.default_outmost_year != 0
        ):
            self.sample_outmost_year_spin.setValue(self.default_outmost_year)
        else:
            self.sample_outmost_year_spin.setValue(
                0
            )  # Will display as "Not set"

    def get_result(self) -> Tuple[Dict, bool]:
        """
        Get the metadata and apply_to_all flag from the dialog.

        Returns:
            Tuple[Dict, bool]: (metadata_dict, apply_to_all_flag)
        """
        # Build metadata dictionary with all sample_ fields
        metadata = {
            "sample_name": self.sample_name_edit.text(),
            "sample_type": self.sample_type_combo.currentText(),
            "sample_geometry": self.sample_geometry_combo.currentText(),
            "sample_scale": self.sample_scale_spin.value(),
            "sample_angle": self.sample_angle_spin.value(),
            # Add the sample_files field with content extensions
            "sample_files": [
                self.scan_content_extension,
                self.metadata_file_extension.split(".json")[0],
            ],
        }

        # Only include outmost year if it's set (not zero)
        if self.sample_outmost_year_spin.value() != 0:
            metadata["sample_outmost_year"] = (
                self.sample_outmost_year_spin.value()
            )

        # Get apply_to_all flag
        apply_to_all = self.apply_to_all_checkbox.isChecked()

        return metadata, apply_to_all
