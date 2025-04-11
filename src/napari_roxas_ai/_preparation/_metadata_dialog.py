from typing import Dict

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from .._settings._settings_manager import SettingsManager


class MetadataDialog(QDialog):
    """Dialog for entering sample metadata."""

    def __init__(
        self,
        default_metadata=None,
        authorized_sample_types=None,
        authorized_sample_geometries=None,
    ):
        super().__init__()

        settings = SettingsManager()

        if authorized_sample_types is None:
            authorized_sample_types = settings.get(
                "samples_metadata.authorised_sample_types"
            )

        if authorized_sample_geometries is None:
            authorized_sample_geometries = settings.get(
                "samples_metadata.authorised_sample_geometries"
            )

        self.setWindowTitle("Sample Metadata")

        # Create form layout
        form_layout = QFormLayout()

        # Sample name (read-only)
        self.sample_name = default_metadata.get("sample_name", "")

        # Sample type dropdown with authorized values
        self.sample_type = QComboBox()
        # Add authorized sample types to the dropdown
        for sample_type in authorized_sample_types:
            self.sample_type.addItem(sample_type)

        # Set default value if provided and exists in authorized types
        default_type = default_metadata.get("sample_type", "")
        default_index = self.sample_type.findText(default_type)
        if default_index >= 0:
            self.sample_type.setCurrentIndex(default_index)
        elif self.sample_type.count() > 0:
            # Default to first option if provided value isn't valid
            self.sample_type.setCurrentIndex(0)

        form_layout.addRow("Sample Type:", self.sample_type)

        # Sample geometry dropdown with authorized values
        self.sample_geometry = QComboBox()
        # Add authorized sample geometries to the dropdown
        for sample_geometry in authorized_sample_geometries:
            self.sample_geometry.addItem(sample_geometry)

        # Set default value if provided and exists in authorized geometries
        default_geometry = default_metadata.get("sample_geometry", "")
        default_geo_index = self.sample_geometry.findText(default_geometry)
        if default_geo_index >= 0:
            self.sample_geometry.setCurrentIndex(default_geo_index)
        elif self.sample_geometry.count() > 0:
            # Default to first option if provided value isn't valid
            self.sample_geometry.setCurrentIndex(0)

        form_layout.addRow("Sample Geometry:", self.sample_geometry)

        # Sample scale - updated to pixels/μm units
        self.sample_scale = QDoubleSpinBox()
        self.sample_scale.setRange(0.01, 100.0)
        self.sample_scale.setSingleStep(
            0.00001
        )  # Smaller step for more precision
        self.sample_scale.setDecimals(
            8
        )  # More decimals to display the exact value
        self.sample_scale.setValue(default_metadata.get("sample_scale"))
        form_layout.addRow("Scale (pixels/μm):", self.sample_scale)

        # Sample angle
        self.sample_angle = QDoubleSpinBox()
        self.sample_angle.setRange(-180.0, 180.0)
        self.sample_angle.setSingleStep(0.1)
        self.sample_angle.setValue(default_metadata.get("sample_angle"))
        self.sample_angle.setDecimals(1)
        form_layout.addRow("Angle (degrees):", self.sample_angle)

        # Apply to all checkbox
        self.apply_to_all_checkbox = QCheckBox(
            "Apply these settings to all remaining files"
        )

        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # Main layout
        layout = QVBoxLayout()

        # Add a header label
        header = QLabel(f"Enter metadata for: {self.sample_name}")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        layout.addLayout(form_layout)
        layout.addWidget(self.apply_to_all_checkbox)
        layout.addWidget(self.button_box)

        self.setLayout(layout)

        # Set a reasonable default size
        self.resize(400, 250)

    def get_metadata(self) -> Dict:
        """Get the metadata values from the dialog."""
        metadata = {
            "sample_name": self.sample_name,
            "sample_type": self.sample_type.currentText(),
            "sample_geometry": self.sample_geometry.currentText(),
            "sample_scale": self.sample_scale.value(),
            "sample_angle": self.sample_angle.value(),
        }

        # Add the sample_files field using content extensions
        # Get the settings for file extensions
        settings_manager = SettingsManager()
        scan_file_extension = settings_manager.get(
            "file_extensions.scan_file_extension"
        )
        scan_content_extension = (
            scan_file_extension[0] if scan_file_extension else None
        )

        metadata_file_extension_parts = settings_manager.get(
            "file_extensions.metadata_file_extension"
        )
        metadata_content_extension = (
            metadata_file_extension_parts[0]
            if metadata_file_extension_parts
            else None
        )

        metadata["sample_files"] = [
            scan_content_extension,
            metadata_content_extension,
        ]

        return metadata

    def apply_to_all(self) -> bool:
        """Check if the user wants to apply these settings to all files."""
        return self.apply_to_all_checkbox.isChecked()
