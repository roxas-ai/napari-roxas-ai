from typing import TYPE_CHECKING

from magicgui.widgets import (
    Container,
    PushButton,
)
from PIL import Image
from qtpy.QtWidgets import QFileDialog

from napari_roxas_ai._settings import SettingsManager

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

settings = SettingsManager()


class ProjectDirectorySettingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self.project_directory_path = settings.get("project_directory")

        # Get input directory
        self._project_directory_dialog_button = PushButton(
            text=f"Input Directory: {self.project_directory_path}",
        )
        self._project_directory_dialog_button.changed.connect(
            self._open_project_directory_dialog
        )

        # Append the widgets to the container
        self.extend(
            [
                self._project_directory_dialog_button,
            ]
        )

    def _open_project_directory_dialog(self):
        """Open a file dialog to select the input directory path."""
        self.project_directory_path = QFileDialog.getExistingDirectory(
            parent=self.project_directory_path,
            caption="Select Project Directory",
        )
        self._project_directory_dialog_button.text = (
            f"Input Directory: {self.project_directory_path}"
        )
        settings.set("project_directory", self.project_directory_path)
