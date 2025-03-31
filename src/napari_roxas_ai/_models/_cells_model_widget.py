import os
from typing import TYPE_CHECKING

from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    create_widget,
)
from PIL import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog

from .._utils import make_binary_labels_colormap
from ._cells_model import CellsSegmentationModel

if TYPE_CHECKING:
    import napari

module_path = os.path.abspath(__file__).rsplit("/", 1)[0]


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object)  # One images to send

    def __init__(
        self, input_array, output_file_path: str, model_weights_file: str
    ):
        super().__init__()
        self.input_array = input_array
        self.output_file_path = output_file_path
        self.model_weights_file = (
            f"{module_path}/_weights/{model_weights_file}"
        )

    def run(self):
        # Set up model
        model = CellsSegmentationModel()
        model.load_weights(self.model_weights_file)

        # Perform inference
        predicted_labels = model.infer(self.input_array)

        # Save labels if path is selected
        if self.output_file_path is not None:
            Image.fromarray(predicted_labels.astype("uint8")).save(
                self.output_file_path
            )

        # Emit results to be added as a layer
        self.result_ready.emit(predicted_labels)

        self.finished.emit()


class CellsModelWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create a layer selection widget for image layers
        self._input_layer_combo = create_widget(
            label="Thin Section", annotation="napari.layers.Image"
        )

        # Scan weights directory for weight files (currrently no restiction on file names)
        self._model_weights_file = ComboBox(
            choices=tuple(os.listdir(f"{module_path}/_weights/")),
            label="Model Weights",
        )

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Output File: None")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Model")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._model_weights_file,
                self._file_dialog_button,
                self._run_analysis_button,
            ]
        )

        # Initialize output file path
        self.output_file_path = None

    def _open_file_dialog(self):
        """Open a file dialog to select the output file path."""
        self.output_file_path, _ = QFileDialog.getSaveFileName(
            parent=None,
            caption="Select Output File",
            filter="Portable Network Graphics (*.png);;All Files (*)",
        )
        self._file_dialog_button.text = f"Output File: {self.output_file_path}"

    def _run_analysis(self):
        """Run the analysis in a separate thread."""

        # Get the selected label layer
        self.input_layer = self._input_layer_combo.value
        if self.input_layer is None:
            raise ValueError("Input layer is not set.")

        # Run the analysis in a separate thread
        self._run_in_thread(
            self.input_layer.data,
            self.output_file_path,
            self._model_weights_file.value,
        )

    def _run_in_thread(
        self, input_array, output_file_path: str, model_weights_file: str
    ):
        """Run the analysis in a separate thread."""
        self.worker_thread = QThread()
        self.worker = Worker(input_array, output_file_path, model_weights_file)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self._add_result_layers)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def _add_result_layers(self, predicted_labels):
        """Add result images to the viewer."""
        self._viewer.add_labels(
            (predicted_labels / 255).astype("uint8"),
            name="Segmented Cells",
            colormap=make_binary_labels_colormap(),
        )
