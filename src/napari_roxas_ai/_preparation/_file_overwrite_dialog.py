import os

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class FileOverwriteDialog(QDialog):
    """
    Dialog to confirm file overwrite when a duplicate file is detected.

    Provides options to keep or overwrite the file, and to apply the choice
    to all subsequent duplicate files.
    """

    def __init__(self, filename: str):
        """
        Initialize the overwrite confirmation dialog.

        Args:
            filename: Name of the file being processed
        """
        super().__init__()

        self.setWindowTitle("Duplicate File Detected")

        # Set up the dialog layout
        layout = QVBoxLayout()

        # Add file info
        layout.addWidget(
            QLabel(f"File already exists: {os.path.basename(filename)}")
        )
        layout.addWidget(QLabel("What would you like to do?"))

        # Add buttons
        self.keep_button = QPushButton("Keep Original")
        self.overwrite_button = QPushButton("Overwrite")

        self.keep_button.clicked.connect(self._keep_clicked)
        self.overwrite_button.clicked.connect(self._overwrite_clicked)

        layout.addWidget(self.keep_button)
        layout.addWidget(self.overwrite_button)

        # Add "apply to all" checkbox
        self.apply_to_all_checkbox = QCheckBox("Apply to all duplicate files")
        layout.addWidget(self.apply_to_all_checkbox)

        self.setLayout(layout)

        # Result storage
        self.overwrite = False  # Default to keeping original

    def _keep_clicked(self):
        """Handle the 'Keep Original' button click."""
        self.overwrite = False
        self.accept()

    def _overwrite_clicked(self):
        """Handle the 'Overwrite' button click."""
        self.overwrite = True
        self.accept()

    def should_overwrite(self) -> bool:
        """Return whether the file should be overwritten."""
        return self.overwrite

    def apply_to_all(self) -> bool:
        """Return whether to apply this choice to all duplicate files."""
        return self.apply_to_all_checkbox.isChecked()
