from typing import Any, Dict, List

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class MetadataDialog(QDialog):
    """Dialog for entering sample metadata."""

    def __init__(
        self,
        default_metadata: Dict[str, Any],
        authorized_sample_types: List[str],
        authorized_sample_geometries: List[str],
    ):
        """
        Initialize the metadata dialog.

        Args:
            default_metadata: Initial values for metadata fields
            authorized_sample_types: List of authorized sample types
            authorized_sample_geometries: List of authorized sample geometries
        """
        super().__init__()

        self.setWindowTitle("Sample Metadata")
        self.setMinimumWidth(400)

        # Create the main layout
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Create form layout for metadata fields
        form_layout = QFormLayout()

        # Sample name field - as non-editable QLabel
        self.sample_name = QLabel(default_metadata.get("sample_name", ""))
        form_layout.addRow("Sample Name:", self.sample_name)

        # Sample type dropdown
        self.sample_type = QComboBox()
        self.sample_type.addItems(authorized_sample_types)
        if default_metadata.get("sample_type") in authorized_sample_types:
            self.sample_type.setCurrentText(
                default_metadata.get("sample_type")
            )
        form_layout.addRow("Sample Type:", self.sample_type)

        # Sample geometry dropdown
        self.sample_geometry = QComboBox()
        self.sample_geometry.addItems(authorized_sample_geometries)
        if (
            default_metadata.get("sample_geometry")
            in authorized_sample_geometries
        ):
            self.sample_geometry.setCurrentText(
                default_metadata.get("sample_geometry")
            )
        form_layout.addRow("Sample Geometry:", self.sample_geometry)

        # Sample scale field
        self.sample_scale = QDoubleSpinBox()
        self.sample_scale.setDecimals(6)
        self.sample_scale.setRange(0.000001, 1000.0)
        self.sample_scale.setValue(default_metadata.get("sample_scale", 1.0))
        form_layout.addRow("Sample Scale (pixels/μm):", self.sample_scale)

        # Sample angle field
        self.sample_angle = QDoubleSpinBox()
        self.sample_angle.setRange(-360.0, 360.0)
        self.sample_angle.setValue(default_metadata.get("sample_angle", 0.0))
        form_layout.addRow("Sample Angle (°):", self.sample_angle)

        # Outmost year field
        self.sample_outmost_year = QSpinBox()
        self.sample_outmost_year.setRange(
            -10000, 3000
        )  # Allow for ancient samples
        self.sample_outmost_year.setValue(
            default_metadata.get("sample_outmost_year") or 0
        )
        form_layout.addRow("Sample Outmost Year:", self.sample_outmost_year)

        # Sample files information (read-only)
        if "sample_files" in default_metadata:
            sample_files_text = ", ".join(default_metadata["sample_files"])
            self.sample_files = QLabel(sample_files_text)
            form_layout.addRow("Sample Files:", self.sample_files)

        # Apply to all checkbox
        self.apply_to_all_checkbox = QCheckBox("Apply to all remaining files")
        form_layout.addRow("", self.apply_to_all_checkbox)

        # Add form layout to main layout
        main_layout.addLayout(form_layout)

        # Add dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata from the dialog fields.

        Returns:
            Dict: Dictionary containing the metadata values
        """
        return {
            "sample_name": self.sample_name.text(),
            "sample_type": self.sample_type.currentText(),
            "sample_geometry": self.sample_geometry.currentText(),
            "sample_scale": self.sample_scale.value(),
            "sample_angle": self.sample_angle.value(),
            "sample_outmost_year": self.sample_outmost_year.value(),
            "sample_files": self.sample_files.text(),
        }

    def apply_to_all(self) -> bool:
        """
        Check if the 'apply to all' checkbox is selected.

        Returns:
            bool: True if the checkbox is checked, False otherwise
        """
        return self.apply_to_all_checkbox.isChecked()
