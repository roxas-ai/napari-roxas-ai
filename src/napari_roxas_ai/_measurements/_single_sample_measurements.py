from typing import TYPE_CHECKING, Any, Dict
from datetime import datetime
from pathlib import Path
from qtpy.QtCore import QTimer
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
from .._utils._version_utils import get_software_version
from napari_roxas_ai._writer import write_single_layer


import numpy as np
import pandas as pd


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

        self._spinner_frames = ["|", "/", "-", "\\"]
        self._spinner_index = 0

        self._spinner_timer = QTimer()
        self._spinner_timer.setInterval(150)  # update speed in ms
        self._spinner_timer.timeout.connect(self._update_spinner)

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
        self._cluster_dbl_cwt_threshold = FloatSpinBox(
            value=settings.get("measurements.cluster_dbl_cwt_threshold"),
            label="Cluster DBL CWT Threshold (µm)",
        )
        self._smoothing_kernel_size = SpinBox(
            value=settings.get("measurements.cells_smoothing_kernel_size"),
            label="Smoothing Kernel Size (1 to disable)",
        )
        self._relwidth_cwt_integration = FloatSpinBox(
            value=settings.get("measurements.relwidth_cwt_integration"),
            label="Wall Fraction for Thickness Measurement",
        )

        # Create a container for the cells measurements settings
        self._cells_measurements_settings = Container()
        self._cells_measurements_settings.extend(
            [
                self._cluster_dbl_cwt_threshold,
                self._smoothing_kernel_size,
                self._relwidth_cwt_integration,
            ]
        )

        # Rings measurement checkbox and settings
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rings"
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

    def _update_spinner(self):
        frame = self._spinner_frames[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self._run_analysis_button.text = f"{self._current_status} {frame}"

    def _get_valid_layers(self, widget=None) -> list:
        names = set()

        for layer in self._viewer.layers:
            meta = getattr(layer, "metadata", None)
            if not isinstance(meta, dict):
                continue

            sample_name = meta.get("sample_name")
            if isinstance(sample_name, str) and sample_name:
                names.add(sample_name)

        return sorted(names)

    def _update_cells_settings_visibility(self):
        self._cells_measurements_settings.visible = (
            self._measure_cells_checkbox.value
        )

    def _run_analysis(self):
        self._run_analysis_button.enabled = False
        self._current_status = "Processing"
        self._spinner_index = 0
        self._spinner_timer.start()


        # Get the selected label layer
        if self._input_sample_combo.value is None:
            show_info("Input sample is not set.")
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Run Analysis"
            self._run_analysis_button.enabled = True
            return

        # Check if the layers exists
        self._cells_layer_name = (
            self._input_sample_combo.value
            + settings.get("file_extensions.cells_file_extension")[0]
        )
        if self._cells_layer_name not in self._viewer.layers:
            show_info(
                f"Layer {self._cells_layer_name} not found in the viewer. Please load the sample first or disable cells processing."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Run Analysis"
            self._run_analysis_button.enabled = True
            return

        self._rings_layer_name = (
            self._input_sample_combo.value
            + settings.get("file_extensions.rings_file_extension")[0]
        )
        if self._rings_layer_name not in self._viewer.layers:
            show_info(
                f"Layer {self._rings_layer_name} not found in the viewer. Please load the sample first or disable rings processing."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Run Analysis"
            self._run_analysis_button.enabled = True
            return

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

        sample_type = None
        if self._measure_cells_checkbox.value:
            sample_type = self._cells_input_layer.metadata.get("sample_type")
        elif self._measure_rings_checkbox.value:
            sample_type = self._rings_input_layer.metadata.get("sample_type")

        config = {
            "pixels_per_um": scale,
            "cluster_dbl_cwt_threshold": self._cluster_dbl_cwt_threshold.value,
            "smoothing_kernel_size": self._smoothing_kernel_size.value,
            "relwidth_cwt_integration": self._relwidth_cwt_integration.value,
            "tangential_angle": settings.get(
                "measurements.cells_tangential_angle"
            ),
            "lower_limit_cwt_iqr_multiplier": settings.get(
                "measurements.lower_limit_cwt_iqr_multiplier"
            ),
            "upper_limit_cwt_iqr_multiplier": settings.get(
                "measurements.upper_limit_cwt_iqr_multiplier"
            ),
            "opposite_cwt_ratio_limit": settings.get(
                "measurements.opposite_cwt_ratio_limit"
            ),
            "adjacent_cwt_ratio_limit": settings.get(
                "measurements.adjacent_cwt_ratio_limit"
            ),
            "sample_type": sample_type,
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

        self.worker_thread.finished.connect(self._spinner_timer.stop)
        self.worker_thread.finished.connect(
            lambda: setattr(self._run_analysis_button, "text", "Run Analysis")
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._run_analysis_button, "enabled", True)
        )

        self.worker_thread.start()


    def _add_result_layers(self, cells_table, rings_table):

        project_dir = Path(settings.get("project_directory"))
        project_dir.mkdir(parents=True, exist_ok=True)

        # One timestamp per measurement run, so that cells and rings written by
        # the same run carry the identical value.
        meas_created_at = datetime.now().isoformat()
        sw_version = get_software_version()

        # ---------------------------
        # Export Cells
        # ---------------------------
        if not cells_table.empty:
            print("[Cells] Updating layer features...")
            self._cells_input_layer.features = cells_table
            self._cells_input_layer.metadata["meas_created_at"] = (
                meas_created_at
            )
            self._cells_input_layer.metadata["sw_version"] = sw_version

            cells_path = self._cells_input_layer.metadata.get("file_path")


            if cells_path is None:
                sample_name = self._cells_input_layer.metadata.get(
                    "sample_name", self._cells_input_layer.name
                )
                cells_path = str(project_dir / f"{sample_name}.cells.png")
                self._cells_input_layer.metadata["file_path"] = cells_path

            print(f"[Cells] Writing results to: {cells_path}")
            write_single_layer(
                path=cells_path,
                data=self._cells_input_layer.data,
                meta={
                    "name": self._cells_input_layer.name,
                    "metadata": self._cells_input_layer.metadata,
                    "features": cells_table,
                },
            )
            show_info(f"Cells exported to: {cells_path}")

        # ---------------------------
        # Export Rings
        # ---------------------------
        if not rings_table.empty:
            print("[Rings] Updating layer features...")
            self._rings_input_layer.features = rings_table
            self._rings_input_layer.metadata["meas_created_at"] = (
                meas_created_at
            )
            self._rings_input_layer.metadata["sw_version"] = sw_version

            rings_path = self._rings_input_layer.metadata.get("file_path")

            if rings_path is None:
                sample_name = self._rings_input_layer.metadata.get(
                    "sample_name", self._rings_input_layer.name
                )
                rings_path = str(project_dir / f"{sample_name}.rings.png")
                self._rings_input_layer.metadata["file_path"] = rings_path

            print(f"[Rings] Writing results to: {rings_path}")

            # move RBXY at the end of the table
            if "RBXY" in rings_table.columns:
                cols = [c for c in rings_table.columns if c != "RBXY"]
                cols.append("RBXY")
                rings_table = rings_table[cols]

            write_single_layer(
                path=rings_path,
                data=self._rings_input_layer.data,
                meta={
                    "name": self._rings_input_layer.name,
                    "metadata": self._rings_input_layer.metadata,
                    "features": rings_table,
                },
            )
            show_info(f"Rings exported to: {rings_path}")

        show_info("Measurements completed and saved.")


