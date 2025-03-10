"""
This code creates the widget to launch the rings-related analysis scripts
"""

from typing import TYPE_CHECKING

from magicgui.widgets import Container, PushButton
from qtpy.QtWidgets import QFileDialog

from .rings_measurements import measure_rings

if TYPE_CHECKING:
    import napari


class RingsMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Open File Dialog")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Append the widgets to the container
        self.extend([self._file_dialog_button])

    def _open_file_dialog(self):
        # Open a file dialog and get the selected file path
        file_path, _ = QFileDialog.getOpenFileName(
            parent=None, caption="Select a File", filter="All Files (*)"
        )

        if file_path:
            measure_rings(file_path)
