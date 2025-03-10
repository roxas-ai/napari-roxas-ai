"""
This code is just a generic template to create a file dialog widget
"""

from typing import TYPE_CHECKING

from magicgui.widgets import Container, PushButton, create_widget
from qtpy.QtWidgets import QFileDialog

if TYPE_CHECKING:
    import napari


class FileDialog(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create a widget for selecting a label layer
        self._label_layer_combo = create_widget(
            label="Label Layer", annotation="napari.layers.Labels"
        )

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Open File Dialog")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Append the widgets to the container
        self.extend([self._label_layer_combo, self._file_dialog_button])

    def _open_file_dialog(self):
        # Open a file dialog and get the selected file path
        file_path, _ = QFileDialog.getOpenFileName(
            parent=None, caption="Select a File", filter="All Files (*)"
        )

        if file_path:
            print(f"Selected file: {file_path}")
            # You can add additional logic here to handle the selected file
