from typing import TYPE_CHECKING, Any, Dict, Optional
from datetime import datetime
from pathlib import Path
import napari.layers
from magicgui.widgets import (
    CheckBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QPoint, QThread, QTimer, Signal
from qtpy.QtGui import QTextDocument
from qtpy.QtWidgets import QInputDialog, QToolTip
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present
from ._sample_measurer import SampleAnalyzer, save_cwt_visualization_json
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


def format_cwt_tooltip(circle: Dict[str, Any]) -> str:
    cell_id = circle.get("cell_id", "")
    wall = str(circle.get("wall", "")).capitalize()
    radius_um = circle.get("radius_um")
    if radius_um is not None:
        cwt_str = f"{float(radius_um):.2f} µm"
    else:
        cwt_str = "N/A"
    status = circle.get("status", "included")
    reason = circle.get("reason", "")

    status_color = "#28a745" if status == "included" else "#d9534f"
    status_label = status.capitalize()

    lines = [
        f"<b>Cell {cell_id}</b> ({wall} wall)",
        f"<b>CWT:</b> {cwt_str}",
        f"<b>Status:</b> <span style='color: {status_color};'><b>{status_label}</b></span>",
    ]
    if status != "included" and reason:
        lines.append(f"<b>Reason:</b> {reason}")

    return "<br>".join(lines)


def calculate_cwt_tooltip_pos(
    circle_center: QPoint,
    radius_screen: float,
    wall: str,
    tooltip_html: str,
    margin: int = 12,
) -> QPoint:
    """Calculate target screen coordinate for QToolTip.showText so that the entire
    tooltip box appears outside the cell wall circle with a constant visual margin
    (above for pith, below for bark, left for left, right for right).
    """
    doc = QTextDocument()
    doc.setDefaultFont(QToolTip.font())
    doc.setHtml(tooltip_html)
    size = doc.size()
    box_w = int(size.width()) + 16
    box_h = int(size.height()) + 12

    wall_lower = str(wall).lower()
    r = int(round(radius_screen))
    cx, cy = circle_center.x(), circle_center.y()

    if wall_lower == "pith":
        # Entire box ABOVE the circle: bottom edge placed margin pixels above top of circle
        return QPoint(
            cx - box_w // 2,
            cy - r - box_h - margin,
        )
    elif wall_lower == "bark":
        # Entire box BELOW the circle: top edge placed margin pixels below bottom of circle
        return QPoint(
            cx - box_w // 2,
            cy + r + margin,
        )
    elif wall_lower == "left":
        # Entire box LEFT of the circle: right edge placed margin pixels left of circle
        return QPoint(
            cx - r - box_w - margin,
            cy - box_h // 2,
        )
    elif wall_lower == "right":
        # Entire box RIGHT of the circle: left edge placed margin pixels right of circle
        return QPoint(
            cx + r + margin,
            cy - box_h // 2,
        )
    else:
        return QPoint(cx + r + margin, cy - box_h // 2)


_last_hovered_cwt_key: Optional[Any] = None


def on_cwt_mouse_move(viewer: "napari.viewer.Viewer", event: Any):
    global _last_hovered_cwt_key
    if not hasattr(viewer, "layers"):
        return

    try:
        cwt_layers = [
            layer
            for layer in viewer.layers
            if (
                getattr(layer, "name", "").endswith(".cwt_circles")
                and getattr(layer, "visible", True)
            )
        ]
    except Exception:
        return

    if not cwt_layers:
        if _last_hovered_cwt_key is not None:
            _last_hovered_cwt_key = None
            try:
                QToolTip.hideText()
            except Exception:
                pass
        return

    pos = getattr(event, "position", None)
    if pos is None:
        if _last_hovered_cwt_key is not None:
            _last_hovered_cwt_key = None
            try:
                QToolTip.hideText()
            except Exception:
                pass
        return

    hovered_circle = None
    hovered_layer = None
    hovered_layer_name = ""
    min_dist = float("inf")

    for layer in cwt_layers:
        try:
            circles = layer.metadata.get("cwt_data", {}).get("circles", [])
            if not circles:
                continue

            data_pos = layer.world_to_data(pos)
            y, x = float(data_pos[-2]), float(data_pos[-1])

            for circle in circles:
                center = circle.get("center_yx")
                if not center or len(center) < 2:
                    continue
                cy, cx = float(center[0]), float(center[1])
                r = float(circle.get("radius_px", 0.0))
                dist = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5
                if dist <= (r + 2.0):
                    if dist < min_dist:
                        min_dist = dist
                        hovered_circle = circle
                        hovered_layer = layer
                        hovered_layer_name = getattr(layer, "name", "")
        except Exception:
            continue

    if hovered_circle is not None and hovered_layer is not None:
        circle_key = (
            hovered_layer_name,
            hovered_circle.get("cell_id"),
            str(hovered_circle.get("wall", "")).lower(),
        )
        if circle_key != _last_hovered_cwt_key:
            _last_hovered_cwt_key = circle_key
            tooltip_html = format_cwt_tooltip(hovered_circle)
            try:
                center = hovered_circle.get("center_yx", [0.0, 0.0])
                cy_data, cx_data = float(center[0]), float(center[1])
                r_data = float(hovered_circle.get("radius_px", 0.0))

                qv = getattr(viewer.window, "_qt_viewer", None)
                if qv is not None and hasattr(qv, "canvas"):
                    canvas = qv.canvas
                    transform = (
                        canvas.view.transform * canvas.view.scene.transform
                    )

                    # Transform circle center: layer data coords -> world -> canvas -> screen
                    world_center = hovered_layer.data_to_world([cy_data, cx_data])
                    canvas_center = transform.map(
                        [world_center[-1], world_center[-2]]
                    )[:2]
                    circle_center_screen = canvas.native.mapToGlobal(
                        QPoint(
                            int(round(canvas_center[0])),
                            int(round(canvas_center[1])),
                        )
                    )

                    # Transform individual circle perimeter point to determine screen radius
                    world_edge = hovered_layer.data_to_world(
                        [cy_data, cx_data + r_data]
                    )
                    canvas_edge = transform.map(
                        [world_edge[-1], world_edge[-2]]
                    )[:2]
                    radius_screen = max(
                        1.0,
                        float(np.linalg.norm(canvas_edge - canvas_center)),
                    )

                    target_pos = calculate_cwt_tooltip_pos(
                        circle_center_screen,
                        radius_screen,
                        hovered_circle.get("wall", ""),
                        tooltip_html,
                    )
                    QToolTip.showText(target_pos, tooltip_html)
            except Exception:
                pass
    else:
        if _last_hovered_cwt_key is not None:
            _last_hovered_cwt_key = None
            try:
                QToolTip.hideText()
            except Exception:
                pass


def create_or_update_cwt_circles_layer(
    viewer: "napari.viewer.Viewer",
    sample_name: str,
    cwt_vis_data: Dict[str, Any],
    scale: Any = None,
    file_path: Optional[str] = None,
) -> Optional["napari.layers.Shapes"]:
    """Create or update a napari.layers.Shapes layer representing CWT measurement circles.

    Parameters
    ----------
    viewer : napari.viewer.Viewer
        Active napari viewer.
    sample_name : str
        Name of the sample.
    cwt_vis_data : dict
        CWT visualization dictionary containing circles list.
    scale : tuple or array, optional
        Spatial scale from the cells/scan layer.
    file_path : str, optional
        Path to the .cwt.json file on disk.

    Returns
    -------
    napari.layers.Shapes or None
        The created or updated Shapes layer, or None if disabled / no circles.
    """
    cwt_enabled = settings.get("cwt_visualization.enabled")
    if cwt_enabled is None:
        cwt_enabled = True

    layer_name = f"{sample_name}.cwt_circles"
    existing_cwt_layer = next(
        (layer for layer in viewer.layers if layer.name == layer_name),
        None,
    )

    if not cwt_enabled or not cwt_vis_data or not cwt_vis_data.get("circles"):
        if existing_cwt_layer is not None:
            viewer.layers.remove(existing_cwt_layer)
        return None

    inc_color = str(
        settings.get("cwt_visualization.included_color") or "lime"
    ).strip()
    exc_color = str(
        settings.get("cwt_visualization.excluded_color") or "red"
    ).strip()
    edge_width = float(
        settings.get("cwt_visualization.edge_width") or 3.0
    )

    shapes_data = []
    edge_colors = []
    features_list = []
    for circle in cwt_vis_data["circles"]:
        center = circle.get("center_yx")
        if not center or len(center) < 2:
            continue
        cy, cx = float(center[0]), float(center[1])
        r = float(circle.get("radius_px", 0.0))
        bbox = np.array(
            [
                [cy - r, cx - r],
                [cy - r, cx + r],
                [cy + r, cx + r],
                [cy + r, cx - r],
            ]
        )
        shapes_data.append(bbox)
        status = circle.get("status", "included")
        edge_colors.append(
            inc_color if status == "included" else exc_color
        )
        features_list.append(
            {
                "cell_id": circle.get("cell_id", ""),
                "wall": str(circle.get("wall", "")).capitalize(),
                "cwt_um": circle.get("radius_um"),
                "status": status,
                "reason": circle.get("reason", ""),
            }
        )

    if not shapes_data:
        if existing_cwt_layer is not None:
            viewer.layers.remove(existing_cwt_layer)
        return None

    features_df = pd.DataFrame(features_list)

    if existing_cwt_layer is not None and isinstance(
        existing_cwt_layer, napari.layers.Shapes
    ):
        existing_cwt_layer.data = shapes_data
        existing_cwt_layer.edge_color = edge_colors
        existing_cwt_layer.edge_width = edge_width
        existing_cwt_layer.features = features_df
        if scale is not None:
            existing_cwt_layer.scale = scale
        existing_cwt_layer.metadata["sample_name"] = sample_name
        existing_cwt_layer.metadata["cwt_data"] = cwt_vis_data
        if file_path is not None:
            existing_cwt_layer.metadata["file_path"] = str(file_path)
        cwt_layer = existing_cwt_layer
    else:
        if existing_cwt_layer is not None:
            viewer.layers.remove(existing_cwt_layer)
        metadata = {
            "sample_name": sample_name,
            "cwt_data": cwt_vis_data,
        }
        if file_path is not None:
            metadata["file_path"] = str(file_path)
        cwt_layer = viewer.add_shapes(
            shapes_data,
            shape_type="ellipse",
            name=layer_name,
            edge_width=edge_width,
            edge_color=edge_colors,
            face_color=[0, 0, 0, 0],
            opacity=1.0,
            scale=scale,
            features=features_df,
            metadata=metadata,
        )

    if hasattr(viewer, "mouse_move_callbacks"):
        if on_cwt_mouse_move not in viewer.mouse_move_callbacks:
            viewer.mouse_move_callbacks.append(on_cwt_mouse_move)

    return cwt_layer


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object, object, object, object)

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
        cleaned_cells_array = None
        cwt_vis_data = None
        if self.measurement == "both":
            cells_table, rings_table = self.analyzer.analyze_sample()
            cleaned_cells_array = self.analyzer.cells_array
            cwt_vis_data = self.analyzer.get_cwt_visualization_data()
        elif self.measurement == "cells":
            cells_table = self.analyzer.analyze_cells()
            cleaned_cells_array = self.analyzer.cells_array
            rings_table = pd.DataFrame()
            cwt_vis_data = self.analyzer.get_cwt_visualization_data()
        elif self.measurement == "rings":
            cells_table = pd.DataFrame()
            rings_table = self.analyzer.analyze_rings()
        # Emit results to be added as layers
        self.result_ready.emit(
            cells_table, rings_table, cleaned_cells_array, cwt_vis_data
        )

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
        prevent_duplicate_dock_if_already_present(
            self, viewer, "9A - Measure cells & rings (individual image)"
        )
        self._selected_sample = None

        # Cells measurement checkbox
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )

        # Rings measurement checkbox
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rings"
        )

        # Button to open settings focused on measurements
        self._check_settings_button = PushButton(text="Check Settings")
        self._check_settings_button.changed.connect(self._open_settings)

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Measure")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._measure_cells_checkbox,
                self._measure_rings_checkbox,
                self._check_settings_button,
                self._run_analysis_button,
            ]
        )

    def _open_settings(self):
        """Open SettingsWidget as a floating window centered on the active screen with measurements expanded."""
        SettingsWidget.open_floating(self._viewer, section_name="measurements")

    def _find_layer(
        self, layer_type: type, extension_key: str
    ) -> Optional["napari.layers.Layer"]:
        """Return the layer of the selected sample (or the first match if none is selected)."""
        extension = settings.get(f"file_extensions.{extension_key}")[0]
        sample_name = self._selected_sample
        for layer in self._viewer.layers:
            if not isinstance(layer, layer_type):
                continue
            if sample_name is not None:
                if layer.name == f"{sample_name}{extension}":
                    return layer
            elif layer.name.endswith(extension):
                return layer
        return None

    @property
    def _cells_layer(self) -> Optional["napari.layers.Labels"]:
        return self._find_layer(napari.layers.Labels, "cells_file_extension")

    @property
    def _rings_layer(self) -> Optional["napari.layers.Labels"]:
        return self._find_layer(napari.layers.Labels, "rings_file_extension")

    @property
    def _scan_layer(self) -> Optional["napari.layers.Image"]:
        return self._find_layer(napari.layers.Image, "scan_file_extension")

    def _loaded_sample_names(self) -> list:
        """Names of the samples that have a cells or rings layer loaded."""
        extensions = [
            settings.get("file_extensions.cells_file_extension")[0],
            settings.get("file_extensions.rings_file_extension")[0],
        ]
        names = set()
        for layer in self._viewer.layers:
            if not isinstance(layer, napari.layers.Labels):
                continue
            for extension in extensions:
                if layer.name.endswith(extension):
                    names.add(layer.name[: -len(extension)])
        return sorted(names)

    def _choose_sample(self) -> bool:
        """
        Select the sample to measure, asking the user if several are loaded.

        Returns False if the user cancelled the choice.
        """
        names = self._loaded_sample_names()
        if len(names) <= 1:
            self._selected_sample = names[0] if names else None
            return True

        name, ok = QInputDialog.getItem(
            self.native,
            "Select sample",
            "Several samples are loaded. Which one should be measured?",
            names,
            0,
            False,
        )
        if not ok:
            return False
        self._selected_sample = name
        return True

    def _update_spinner(self):
        frame = self._spinner_frames[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
        self._run_analysis_button.text = f"{self._current_status} {frame}"

    def _run_analysis(self):
        if not self._choose_sample():
            return

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
            self._run_analysis_button.text = "Measure"
            self._run_analysis_button.enabled = True
            return

        # Ensure a target directory can be determined if any layer lacks a file_path
        proj = settings.get("project_directory")
        has_file_path = any(
            layer is not None and layer.metadata.get("file_path")
            for layer in (self._cells_layer, self._rings_layer, self._scan_layer)
        )
        if not proj and not has_file_path:
            show_info(
                "Project directory is not configured. Please define a project directory before running measurements."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Measure"
            self._run_analysis_button.enabled = True
            return

        cells_layer = self._cells_layer
        if self._measure_cells_checkbox.value and cells_layer is None:
            show_info(
                "Cells layer not found in the viewer. Please load the cells layer first or uncheck 'Measure Cells'."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Measure"
            self._run_analysis_button.enabled = True
            return

        rings_layer = self._rings_layer
        if self._measure_rings_checkbox.value and rings_layer is None:
            show_info(
                "Rings layer not found in the viewer. Please load the rings layer first or uncheck 'Measure Rings'."
            )
            self._spinner_timer.stop()
            self._run_analysis_button.text = "Measure"
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
            "opposite_cwt_percentile_exception": settings.get(
                "measurements.opposite_cwt_percentile_exception"
            ),
            "adjacent_cwt_ratio_limit": settings.get(
                "measurements.adjacent_cwt_ratio_limit"
            ),
            "sample_interval": settings.get(
                "cwt_visualization.sample_interval", 25
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
            lambda: setattr(self._run_analysis_button, "text", "Measure")
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._run_analysis_button, "enabled", True)
        )

        self.worker_thread.start()


    def _add_result_layers(
        self,
        cells_table,
        rings_table,
        cleaned_cells_array=None,
        cwt_vis_data=None,
    ):

        proj_setting = settings.get("project_directory")
        fallback_dir = None
        for layer in (
            getattr(self, "_cells_input_layer", None),
            getattr(self, "_rings_input_layer", None),
            getattr(self, "_scan_layer", None),
        ):
            if layer is not None and layer.metadata.get("file_path"):
                fallback_dir = Path(layer.metadata["file_path"]).parent
                break

        if proj_setting:
            project_dir = Path(proj_setting)
            project_dir.mkdir(parents=True, exist_ok=True)
        elif fallback_dir is not None:
            project_dir = fallback_dir
        else:
            project_dir = None

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
            if cleaned_cells_array is not None and hasattr(self._cells_input_layer, "data"):
                if self._cells_input_layer.data.dtype == bool:
                    self._cells_input_layer.data = cleaned_cells_array > 0
                elif np.issubdtype(self._cells_input_layer.data.dtype, np.integer):
                    if self._cells_input_layer.data.max() <= 1:
                        self._cells_input_layer.data = (cleaned_cells_array > 0).astype(self._cells_input_layer.data.dtype)
                    else:
                        self._cells_input_layer.data = cleaned_cells_array.astype(self._cells_input_layer.data.dtype)
                else:
                    self._cells_input_layer.data = (cleaned_cells_array > 0).astype(self._cells_input_layer.data.dtype)
            self._cells_input_layer.metadata["meas_created_at"] = (
                meas_created_at
            )
            self._cells_input_layer.metadata["sw_version"] = sw_version
            self._cells_input_layer.metadata["meas_by"] = meas_by
            # Cells-only parameters, recorded so the run can be reproduced
            if getattr(self, "_run_config", None):
                for key in MEASUREMENT_PARAMETER_KEYS:
                    if key in self._run_config:
                        self._cells_input_layer.metadata[key] = self._run_config[key]

            cells_path = self._cells_input_layer.metadata.get("file_path")


            if cells_path is None:
                if project_dir is None:
                    show_info(
                        "Project directory is not configured. Please define a project directory to save results."
                    )
                    return
                sample_name = self._cells_input_layer.metadata.get(
                    "sample_name", self._cells_input_layer.name
                )
                cells_path = str(project_dir / f"{sample_name}.cells.png")
                self._cells_input_layer.metadata["file_path"] = cells_path
            else:
                sample_name = self._cells_input_layer.metadata.get(
                    "sample_name"
                ) or (
                    Path(cells_path).stem.replace(".cells", "")
                    if cells_path
                    else self._cells_input_layer.name
                )

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
            # Written with the measurements: nothing left to save
            self._cells_input_layer.metadata["_is_dirty"] = False
            show_info(f"Cells exported to: {cells_path}")

            # ---------------------------
            # Export CWT Circle Visualization (Shapes Layer & JSON)
            # ---------------------------
            if cwt_vis_data is not None:
                cwt_path = Path(cells_path).parent / f"{sample_name}.cwt.json"
                cwt_vis_data["sample_name"] = sample_name
                save_cwt_visualization_json(cwt_path, cwt_vis_data)
                scale = getattr(self._cells_input_layer, "scale", None)
                create_or_update_cwt_circles_layer(
                    viewer=self._viewer,
                    sample_name=sample_name,
                    cwt_vis_data=cwt_vis_data,
                    scale=scale,
                    file_path=str(cwt_path),
                )

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
                if project_dir is None:
                    show_info(
                        "Project directory is not configured. Please define a project directory to save results."
                    )
                    return
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
            # Written with the measurements: nothing left to save
            self._rings_input_layer.metadata["_is_dirty"] = False
            show_info(f"Rings exported to: {rings_path}")

        # ---------------------------
        # Export Annotated Scan Image
        # ---------------------------
        scan_layer = self._scan_layer
        rings_layer = getattr(self, "_rings_input_layer", None) or self._rings_layer
        cells_layer = getattr(self, "_cells_input_layer", None) or getattr(self, "_cells_layer", None)
        cells_data = cells_layer.data if cells_layer is not None and hasattr(cells_layer, "data") else None
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
            annotated_dir = project_dir or (Path(scan_path).parent if scan_path else None)

            if annotated_dir and scan_path and Path(scan_path).exists():
                annotated_path = str(annotated_dir / f"{sample_name}_annotated.jpg")
                try:
                    save_annotated_scan_image(
                        scan_path=scan_path,
                        annotated_path=annotated_path,
                        rings_features=rings_features,
                        cells_data=cells_data,
                    )
                    show_info(f"Annotated image exported to: {annotated_path}")
                except Exception as e:
                    print(f"Annotated image export failed: {e}")

        show_info("Measurements completed and saved.")


