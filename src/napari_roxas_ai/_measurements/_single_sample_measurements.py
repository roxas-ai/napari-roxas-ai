from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FloatSpinBox,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QThread, Signal

from napari_roxas_ai._settings import SettingsManager

from ._sample_measurer import SampleAnalyzer

settings = SettingsManager()

if TYPE_CHECKING:
    import napari


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object, object)

    def __init__(
        self,
        config: Dict[str, Any],
        cells_array: np.array,
        rings_table: pd.DataFrame,
        cells_table: pd.DataFrame,
        measurement: str,
    ):
        super().__init__()

        self.analyzer = SampleAnalyzer(
            config, cells_array, rings_table, cells_table
        )
        self.measurement = measurement

    def run(self):
        if self.measurement == "both":
            cells_table, rings_table = self.analyzer.analyze_sample()
        elif self.measurement == "cells":
            cells_table = self.analyzer.analyze_cells()
            rings_table = pd.DataFrame()
        elif self.measurement == "rings":
            cells_table = pd.DataFrame()
            rings_table = self.analyzer.analyze_rings()
        # Emit results to be added as layers
        self.result_ready.emit(cells_table, rings_table)

        self.finished.emit()


class SingleSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Create a layer selection widget for label layers
        self._input_sample_combo = ComboBox(
            label="Sample", choices=self._get_valid_layers
        )

        # Cells measurement checkbox and settings (currently hidden as we want the user to compute cells first anyway)
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )
        self._measure_cells_checkbox.changed.connect(
            self._update_cells_settings_visibility
        )
        # self._measure_cells_checkbox.visible = False

        # Create input fields for the config parameters
        self._cluster_separation_threshold = FloatSpinBox(
            value=settings.get(
                "measurements.cells_cluster_separation_threshold"
            ),
            label="Cluster Separation Threshold (µm)",
        )
        self._smoothing_kernel_size = SpinBox(
            value=settings.get("measurements.cells_smoothing_kernel_size"),
            label="Smoothing Kernel Size (1 to disable)",
        )
        self._integration_interval = FloatSpinBox(
            value=settings.get("measurements.cells_integration_interval"),
            label="Wall Fraction for Thickness Measurement",
        )

        # Create a container for the cells measurements settings
        self._cells_measurements_settings = Container()
        self._cells_measurements_settings.extend(
            [
                self._cluster_separation_threshold,
                self._smoothing_kernel_size,
                self._integration_interval,
            ]
        )

        # Rings measurement checkbox and settings
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rigns"
        )

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Analysis")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._input_sample_combo,
                self._measure_cells_checkbox,
                self._cells_measurements_settings,
                self._measure_rings_checkbox,
                self._run_analysis_button,
            ]
        )

    def _get_valid_layers(self, widget=None) -> list:
        """Get layers names"""

        return list(
            {layer.metadata["sample_name"] for layer in self._viewer.layers}
        )

    def _update_cells_settings_visibility(self):
        self._cells_measurements_settings.visible = (
            self._measure_cells_checkbox.value
        )

    def _run_analysis(self):
        """Run the analysis in a separate thread."""

        # Get the selected label layer
        if self._input_sample_combo.value is None:
            raise ValueError("Input sample is not set.")

        # Check if the layers exists
        self._cells_layer_name = (
            self._input_sample_combo.value
            + settings.get("file_extensions.cells_file_extension")[0]
        )
        if self._cells_layer_name not in self._viewer.layers:
            raise ValueError(
                f"Layer {self._cells_layer_name} not found in the viewer. Please load the sample first or disable cells processing."
            )

        self._rings_layer_name = (
            self._input_sample_combo.value
            + settings.get("file_extensions.rings_file_extension")[0]
        )
        if self._rings_layer_name not in self._viewer.layers:
            raise ValueError(
                f"Layer {self._rings_layer_name} not found in the viewer. Please load the sample first or disable rings processing."
            )

        if (
            self._measure_cells_checkbox.value
            and self._measure_rings_checkbox.value
        ):
            measurement = "both"
            self._cells_input_layer = self._viewer.layers[
                self._cells_layer_name
            ]
            self._rings_input_layer = self._viewer.layers[
                self._rings_layer_name
            ]
            scale = self._cells_input_layer.metadata["sample_scale"]
            cells_array = self._cells_input_layer.data
            rings_table = self._rings_input_layer.features
            cells_table = self._cells_input_layer.features

        elif (
            self._measure_cells_checkbox.value
            and not self._measure_rings_checkbox.value
        ):
            measurement = "cells"
            self._cells_input_layer = self._viewer.layers[
                self._cells_layer_name
            ]
            scale = self._cells_input_layer.metadata["sample_scale"]
            cells_array = self._cells_input_layer.data
            rings_table = pd.DataFrame()
            cells_table = pd.DataFrame()

        elif (
            self._measure_rings_checkbox.value
            and not self._measure_cells_checkbox.value
        ):
            measurement = "rings"
            self._rings_input_layer = self._viewer.layers[
                self._rings_layer_name
            ]
            scale = self._rings_input_layer.metadata["sample_scale"]
            cells_array = np.zeros_like(self._rings_input_layer.data)
            rings_table = self._rings_input_layer.features
            cells_table = pd.DataFrame()

        else:
            raise ValueError("Choose a measurement to compute.")

        config = {
            "pixels_per_um": scale,
            "cluster_separation_threshold": self._cluster_separation_threshold.value,
            "smoothing_kernel_size": self._smoothing_kernel_size.value,
            "integration_interval": self._integration_interval.value,
            "tangential_angle": settings.get(
                "measurements.cells_tangential_angle"
            ),
        }

        # Run the analysis in a separate thread

        self.worker_thread = QThread()
        self.worker = Worker(
            config,
            cells_array.astype("uint8") * 255,
            rings_table,
            cells_table,
            measurement,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self._add_result_layers)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def _add_result_layers(self, cells_table, rings_table):
        """Add result images to the viewer."""

        if not cells_table.empty:
            self._cells_input_layer.features = cells_table
            # TODO: add metadata update

        if not rings_table.empty:
            self._rings_input_layer.features = rings_table
            # TODO: add metadata update

        show_info("measurements completed successfully.")
