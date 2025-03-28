import os
from typing import TYPE_CHECKING

from magicgui.widgets import (
    Container,
    PushButton,
    create_widget,
)
from qtpy.QtCore import QObject, QThread, Signal

from .._utils import make_binary_labels_colormap
from ._cells_model import CellsSegmentationModel

if TYPE_CHECKING:
    import napari

module_path = os.path.abspath(__file__).rsplit("/", 1)[0]


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object)  # One images to send

    def __init__(self, input_array):
        super().__init__()
        self.model_weights_file = (
            f"{module_path}/_weights/conifer_cells_segmentation.pth"
        )
        self.input_array = input_array

    def run(self):
        # Set up model
        model = CellsSegmentationModel()
        model.load_weights(self.model_weights_file)

        # Perform inference
        predicted_labels = model.infer(self.input_array)

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

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Model")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._run_analysis_button,
            ]
        )

    def _run_analysis(self):
        """Run the analysis in a separate thread."""

        # Get the selected label layer
        self.input_layer = self._input_layer_combo.value
        if self.input_layer is None:
            raise ValueError("Input layer is not set.")

        # Run the analysis in a separate thread
        self._run_in_thread(self.input_layer.data)

    def _run_in_thread(self, input_array):
        """Run the analysis in a separate thread."""
        self.worker_thread = QThread()
        self.worker = Worker(input_array)
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
