"""
This code creates the widget to launch the cell wall thickness-related analysis scripts
"""

from typing import TYPE_CHECKING

from magicgui.widgets import Container, PushButton
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog

from ._cwt_measurements import measure_cwt

if TYPE_CHECKING:
    import napari


class Worker(QObject):
    finished = Signal()

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        measure_cwt(self.file_path)
        self.finished.emit()


class CellsMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Open File Dialog")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Append the widgets to the container
        self.extend([self._file_dialog_button])

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            parent=None, caption="Select a File", filter="All Files (*)"
        )

        if file_path:
            self._run_in_thread(file_path)

    def _run_in_thread(self, file_path):
        self.worker_thread = QThread()
        self.worker = Worker(file_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
