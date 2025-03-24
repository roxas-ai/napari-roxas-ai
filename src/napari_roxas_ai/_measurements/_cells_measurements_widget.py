from typing import TYPE_CHECKING, Any, Dict

from magicgui.widgets import (
    Container,
    FloatSpinBox,
    PushButton,
    SpinBox,
    create_widget,
)
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog

from .._utils import make_binary_labels_colormap
from ._cwt_measurements import CellAnalyzer

if TYPE_CHECKING:
    import napari


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object, object)  # Two images to send

    def __init__(
        self, config: Dict[str, Any], input_array, output_file_path: str
    ):
        super().__init__()
        self.config = config
        self.input_array = input_array
        self.output_file_path = output_file_path

    def run(self):
        analyzer = CellAnalyzer(self.config)
        analyzer.cells_array = self.input_array.astype("uint8") * 255
        analyzer._smooth_image()
        analyzer._find_contours()
        analyzer._analyze_lumina()
        analyzer._analyze_cell_walls()
        analyzer._cluster_cells()
        analyzer._get_results_df()
        analyzer._draw_contours()

        # Save results as CSV
        if self.output_file_path is not None:
            analyzer.df.to_csv(self.output_file_path, index_label="cell_id")

        # Emit results to be added as layers
        self.result_ready.emit(
            analyzer.lumen_contours_image, analyzer.cell_walls_contours_image
        )

        self.finished.emit()


class CellsMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create input fields for the config parameters
        self._pixels_per_um = FloatSpinBox(value=2.2675, label="Pixels per µm")
        self._cluster_separation_threshold = FloatSpinBox(
            value=3.0, label="Cluster Separation Threshold (µm)"
        )
        self._smoothing_kernel_size = SpinBox(
            value=5, label="Smoothing Kernel Size (1 to disable)"
        )
        self._integration_interval = FloatSpinBox(
            value=0.75, label="Wall Fraction for Thickness Measurement"
        )
        self._tangential_angle = FloatSpinBox(
            value=0.0, label="Sample Angle (degrees, clockwise)"
        )

        # Create a layer selection widget for label layers
        self._input_layer_combo = create_widget(
            label="Cell Labels", annotation="napari.layers.Labels"
        )

        # Create a button to open a file dialog
        self._file_dialog_button = PushButton(text="Output File: None")
        self._file_dialog_button.changed.connect(self._open_file_dialog)

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Analysis")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._pixels_per_um,
                self._cluster_separation_threshold,
                self._smoothing_kernel_size,
                self._integration_interval,
                self._tangential_angle,
                self._input_layer_combo,
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
            filter="Comma Separated Variables (*.csv);;All Files (*)",
        )
        self._file_dialog_button.text = f"Output File: {self.output_file_path}"

    def _run_analysis(self):
        """Run the analysis in a separate thread."""

        # Get the selected label layer
        self.input_layer = self._input_layer_combo.value
        if self.input_layer is None:
            raise ValueError("Input layer is not set.")

        # Create the config dictionary
        config = {
            "pixels_per_um": self._pixels_per_um.value,
            "cluster_separation_threshold": self._cluster_separation_threshold.value,
            "smoothing_kernel_size": self._smoothing_kernel_size.value,
            "integration_interval": self._integration_interval.value,
            "tangential_angle": self._tangential_angle.value,
        }

        # Run the analysis in a separate thread
        self._run_in_thread(
            config, self.input_layer.data, self.output_file_path
        )

    def _run_in_thread(
        self, config: Dict[str, Any], input_array, output_file_path: str
    ):
        """Run the analysis in a separate thread."""
        self.worker_thread = QThread()
        self.worker = Worker(config, input_array, output_file_path)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self._add_result_layers)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def _add_result_layers(self, lumen_image, walls_image):
        """Add result images to the viewer."""
        self._viewer.add_labels(
            (lumen_image / 255).astype("uint8"),
            name="Lumen Contours",
            colormap=make_binary_labels_colormap(),
        )
        self._viewer.add_labels(
            (walls_image / 255).astype("uint8"),
            name="Cell Wall Contours",
            colormap=make_binary_labels_colormap(),
        )
