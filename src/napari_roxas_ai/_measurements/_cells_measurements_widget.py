"""
This code creates the widget to launch the cell wall thickness-related analysis scripts
"""

from typing import TYPE_CHECKING

from magicgui.widgets import Container, PushButton
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog

from ._cwt_measurements import measure_cells

if TYPE_CHECKING:
    import napari


class Worker(QObject):
    finished = Signal()

    def __init__(self, config, input_array, output_file_path):
        super().__init__()
        self.config = config
        self.input_array = input_array
        self.output_file_path = output_file_path

    def run(self):
        measure_cells(self.config, self.input_array, self.output_file_path)
        self.finished.emit()


class CellsMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create input fields for the following config parameters:
        # config: Dictionary containing analysis parameters:
        #         - pixels_per_um: Conversion factor from pixels to micrometers
        #         - cluster_separation_threshold: Minimum distance between clusters (µm)
        #         - smoothing_kernel_size: Size of morphological operation kernel
        #         - integration_interval: Fraction of wall used for thickness measurement
        #         - tangential_angle : Sample angle (degrees, clockwise)

        # Create a layer selection widget for label layers to use to get the input array

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Open File Dialog")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Append the widgets to the container
        self.extend([self._file_dialog_button])

    def _open_file_dialog(self):
        self.output_file_path, _ = QFileDialog.getOpenFileName(
            parent=None,
            caption="Select a File",
            filter="All Files (*);;Comma Separated Variables (*.csv)",
        )

    def _run_in_thread(self, config, input_array, output_file_path):
        self.worker_thread = QThread()
        self.worker = Worker(config, input_array, output_file_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()
