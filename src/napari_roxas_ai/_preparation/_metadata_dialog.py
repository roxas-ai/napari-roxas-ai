"""
Dialog for entering sample metadata.
"""

from typing import Any, Dict, Optional, Tuple

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

from napari_roxas_ai._settings._settings_manager import SettingsManager


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

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Get metadata field definitions
        self.metadata_fields = (
            self.settings_manager.get("samples_metadata.fields") or []
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
            "".join(metadata_file_extension_parts)
            if metadata_file_extension_parts
            else ".metadata.json"
        )

    def _create_ui(self):
        """Create the UI elements dynamically based on settings."""
        # Main layout
        layout = QVBoxLayout(self)

        # Form layout for metadata fields
        form_layout = QFormLayout()

        # Dictionary to store all created widgets
        self.widgets = {}

        # Create widgets dynamically based on metadata field definitions
        for field_def in self.metadata_fields:
            widget = self._create_widget_from_definition(field_def)
            if widget:
                form_layout.addRow(f"{field_def['label']}:", widget)
                self.widgets[field_def["id"]] = {
                    "widget": widget,
                    "definition": field_def,
                }

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

    def _create_widget_from_definition(
        self, field_def: Dict[str, Any]
    ) -> Optional[QWidget]:
        """
        Create a widget based on field definition from settings.

        Args:
            field_def: Dictionary with widget definition

        Returns:
            Created widget or None if widget type is not supported
        """
        widget_type = field_def.get("widget_type")
        widget_id = field_def.get("id")

        if widget_type == "QLineEdit":
            widget = QLineEdit()
            if field_def.get("read_only", False):
                widget.setReadOnly(True)

            # Set default value for sample_name
            if widget_id == "sample_name":
                widget.setText(self.filename)

            return widget

        elif widget_type == "QComboBox":
            widget = QComboBox()

            # Get items directly from field definition
            items = field_def.get("items", [])
            if items:
                widget.addItems(items)

            # Set editability
            if field_def.get("editable", False):
                widget.setEditable(True)

            return widget

        elif widget_type == "QDoubleSpinBox":
            widget = QDoubleSpinBox()

            # Configure range, step, and precision
            if "min" in field_def:
                widget.setMinimum(field_def["min"])
            if "max" in field_def:
                widget.setMaximum(field_def["max"])
            if "step" in field_def:
                widget.setSingleStep(field_def["step"])
            if "decimals" in field_def:
                widget.setDecimals(field_def["decimals"])

            # Set default value directly from definition
            if "default" in field_def:
                widget.setValue(field_def["default"])

            return widget

        elif widget_type == "QSpinBox":
            widget = QSpinBox()

            # Configure range
            if "min" in field_def:
                widget.setMinimum(field_def["min"])
            if "max" in field_def:
                widget.setMaximum(field_def["max"])

            # Set special value text if specified
            if "special_value_text" in field_def:
                widget.setSpecialValueText(field_def["special_value_text"])

            # Set default value directly from definition
            if "default" in field_def:
                widget.setValue(field_def["default"])

            return widget

        return None

    def get_result(self) -> Tuple[Dict, bool]:
        """
        Get the metadata and apply_to_all flag from the dialog.

        Returns:
            Tuple[Dict, bool]: (metadata_dict, apply_to_all_flag)
        """
        # Build metadata dictionary with values from all widgets
        metadata = {}

        for field_id, widget_data in self.widgets.items():
            widget = widget_data["widget"]
            field_def = widget_data["definition"]

            # Get value based on widget type
            value = None
            widget_type = field_def.get("widget_type")

            if widget_type == "QLineEdit":
                value = widget.text()
            elif widget_type == "QComboBox":
                value = widget.currentText()
            elif widget_type == "QDoubleSpinBox" or widget_type == "QSpinBox":
                value = widget.value()
                # Don't include if it's a special value and not required
                if (
                    widget_type == "QSpinBox"
                    and "special_value_text" in field_def
                    and value == widget.minimum()
                    and not field_def.get("required", True)
                ):
                    continue

            # Add to metadata
            metadata[field_id] = value

        # Get apply_to_all flag
        apply_to_all = self.apply_to_all_checkbox.isChecked()

        return metadata, apply_to_all
