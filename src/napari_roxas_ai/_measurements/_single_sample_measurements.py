from typing import TYPE_CHECKING, Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import napari.layers
from qtpy.QtCore import QTimer
from magicgui.widgets import (
    CheckBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QThread, Signal
from napari_roxas_ai._settings import SettingsManager
from ._sample_measurer import SampleAnalyzer
from .._utils._metadata_keys import MEASUREMENT_PARAMETER_KEYS
from .._utils._version_utils import (
    get_measurement_operator,
    get_software_version,
)
from napari_roxas_ai._writer import write_single_layer
from napari_roxas_ai._writer._writer import save_annotated_scan_image


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

        # Cells measurement checkbox
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )

        # Rings measurement checkbox
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rings"
        )

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Analysis")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._measure_cells_checkbox,
                self._measure_rings_checkbox,
                self._run_analysis_button,
            ]
        )

    @property
    def _cells_layer(self) -> Optional["napari.layers.Labels"]:
        cells_extension = settings.get("file_extensions.cells_file_extension")[
            0
        ]
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name.endswith(
                cells_extension
            ):
                return layer
        return None

    @property
    def _rings_layer(self) -> Optional["napari.layers.Labels"]:
        rings_extension = settings.get("file_extensions.rings_file_extension")[
            0
        ]
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name.endswith(
                rings_extension
            ):
                return layer
        return None

    @property
    def _scan_layer(self) -> Optional["napari.layers.Image"]:
        scan_extension = settings.get("file_extensions.scan_file_extension")[0]
        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name.endswith(
                scan_extension
            ):
                return layer
        return None

    def _update_spinner(self):
        frame = self._spinner_frames[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self._run_analysis_button.text = f"{self._current_status} {frame}"

    def _run_analysis(self):
        self._run_analysis_button.enabled = False
        self._current_status = "Processing"
        self._spinner_index = 0
        self._spinner_timer.start()

        if (
            not self._measure_cells_checkbox.value
            and not self._measure_rings_checkbox.value
        ):
            show_info("Please select at least one measurement (cells or rings).")
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Run Analysis"
            self._run_analysis_button.enabled = True
            return

        cells_layer = self._cells_layer
        if self._measure_cells_checkbox.value and cells_layer is None:
            show_info(
                "Cells layer not found in the viewer. Please load the sample first or disable cells processing."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Run Analysis"
            self._run_analysis_button.enabled = True
            return

        rings_layer = self._rings_layer
        if self._measure_rings_checkbox.value and rings_layer is None:
            show_info(
                "Rings layer not found in the viewer. Please load the sample first or disable rings processing."
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
            self._cells_input_layer = cells_layer
            self._rings_input_layer = rings_layer
            self._cells_layer_name = cells_layer.name
            self._rings_layer_name = rings_layer.name
            scale = self._cells_input_layer.metadata["spatial_resolution"]
            cells_array = self._cells_input_layer.data
            rings_table = self._rings_input_layer.features
            cells_table = self._cells_input_layer.features

        elif (
            self._measure_cells_checkbox.value
            and not self._measure_rings_checkbox.value
        ):
            measurement = "cells"
            self._cells_input_layer = cells_layer
            self._cells_layer_name = cells_layer.name
            scale = self._cells_input_layer.metadata["spatial_resolution"]
            cells_array = self._cells_input_layer.data
            if rings_layer is not None and hasattr(rings_layer, "features") and not rings_layer.features.empty:
                self._rings_input_layer = rings_layer
                self._rings_layer_name = rings_layer.name
                rings_table = self._rings_input_layer.features
            else:
                self._rings_input_layer = None
                self._rings_layer_name = None
                rings_table = pd.DataFrame()
            cells_table = pd.DataFrame()

        elif (
            self._measure_rings_checkbox.value
            and not self._measure_cells_checkbox.value
        ):
            measurement = "rings"
            self._rings_input_layer = rings_layer
            self._rings_layer_name = rings_layer.name
            scale = self._rings_input_layer.metadata["spatial_resolution"]
            cells_array = np.zeros_like(self._rings_input_layer.data)
            rings_table = self._rings_input_layer.features
            cells_table = pd.DataFrame()

        else:
            raise ValueError("Choose a measurement to compute.")

        sample_type = None
        if self._measure_cells_checkbox.value:
            sample_type = self._cells_input_layer.metadata["sample_type"]
        elif self._measure_rings_checkbox.value:
            sample_type = self._rings_input_layer.metadata["sample_type"]

        config = {
            "pixels_per_um": scale,
            "cluster_dbl_cwt_threshold": settings.get(
                "measurements.cluster_dbl_cwt_threshold"
            ),
            "smoothing_kernel_size": settings.get(
                "measurements.cells_smoothing_kernel_size"
            ),
            "relwidth_cwt_integration": settings.get(
                "measurements.relwidth_cwt_integration"
            ),
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

        # Keep the config of this run so that _add_result_layers records the
        # values actually used, even if a widget is changed while it runs.
        self._run_config = config

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
        meas_by = get_measurement_operator()

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
            self._cells_input_layer.metadata["meas_by"] = meas_by
            # Cells-only parameters, recorded so the run can be reproduced
            for key in MEASUREMENT_PARAMETER_KEYS:
                self._cells_input_layer.metadata[key] = self._run_config[key]

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
            self._rings_input_layer.metadata["meas_by"] = meas_by

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

        # ---------------------------
        # Export Annotated Scan Image
        # ---------------------------
        scan_layer = self._scan_layer
        rings_layer = getattr(self, "_rings_input_layer", None) or self._rings_layer
        rings_features = (
            rings_table
            if not rings_table.empty
            else (
                rings_layer.features
                if rings_layer is not None and hasattr(rings_layer, "features")
                else None
            )
        )

        if scan_layer is not None and rings_features is not None and not rings_features.empty:
            scan_path = scan_layer.metadata.get("file_path")
            sample_name = (
                scan_layer.metadata.get("sample_name")
                or (rings_layer.metadata.get("sample_name") if rings_layer and rings_layer.metadata else None)
                or scan_layer.name.split(".")[0]
            )
            annotated_path = str(project_dir / f"{sample_name}_annotated.jpg")

            if scan_path and Path(scan_path).exists():
                try:
                    save_annotated_scan_image(
                        scan_path=scan_path,
                        annotated_path=annotated_path,
                        rings_features=rings_features,
                    )
                    show_info(f"Annotated image exported to: {annotated_path}")
                except Exception as e:
                    print(f"Annotated image export failed: {e}")

        show_info("Measurements completed and saved.")


