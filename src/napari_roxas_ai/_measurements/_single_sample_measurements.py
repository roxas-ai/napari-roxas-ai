from typing import TYPE_CHECKING, Any, Dict, Optional
from pathlib import Path
import napari.layers
from magicgui.widgets import (
    CheckBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import QPoint
from qtpy.QtGui import QTextDocument
from qtpy.QtWidgets import QInputDialog, QToolTip
from napari_roxas_ai._job_runner import JobButton, JobHandle
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present
from ._actions import start_measuring
from ._operations import MeasurementResult

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


class SingleSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

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

        # Create a button to launch the analysis (shows its progress; the
        # measured layers are written at the end, so it cannot be cancelled)
        self._run_analysis_button = JobButton(
            text="Measure", start=self._run_analysis, cancellable=False
        )

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

    def _run_analysis(self) -> JobHandle | None:
        """Measure the selected sample; None when nothing was started."""
        if not self._choose_sample():
            return None

        measure_cells = self._measure_cells_checkbox.value
        measure_rings = self._measure_rings_checkbox.value
        if not measure_cells and not measure_rings:
            show_info("Please select at least one measurement (cells or rings).")
            return None

        cells_layer = self._cells_layer
        if measure_cells and cells_layer is None:
            show_info(
                "Cells layer not found in the viewer. Please load the cells layer first or uncheck 'Measure Cells'."
            )
            return None

        rings_layer = self._rings_layer
        if measure_rings and rings_layer is None:
            show_info(
                "Rings layer not found in the viewer. Please load the rings layer first or uncheck 'Measure Rings'."
            )
            return None

        # Ensure a target directory can be determined if any layer lacks a
        # file_path
        project_dir = settings.get("project_directory")
        has_file_path = any(
            layer is not None and layer.metadata.get("file_path")
            for layer in (cells_layer, rings_layer, self._scan_layer)
        )
        if not project_dir and not has_file_path:
            show_info(
                "Project directory is not configured. Please define a project directory before running measurements."
            )
            return None
        if not project_dir:
            # Next to the measured layers' files
            used_rings = measure_rings or (
                rings_layer is not None and not rings_layer.features.empty
            )
            project_dir = next(
                str(Path(layer.metadata["file_path"]).parent)
                for layer in (
                    cells_layer if measure_cells else None,
                    rings_layer if used_rings else None,
                    self._scan_layer,
                )
                if layer is not None and layer.metadata.get("file_path")
            )

        if measure_cells and measure_rings:
            measurement = "both"
        elif measure_cells:
            measurement = "cells"
        else:
            measurement = "rings"

        # Scale and sample type: from the cells layer when measuring cells
        reference_layer = cells_layer if measure_cells else rings_layer
        config = {
            "pixels_per_um": reference_layer.metadata["spatial_resolution"],
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
            "sample_type": reference_layer.metadata["sample_type"],
        }

        handle = start_measuring(
            measurement,
            config,
            cells_layer,
            rings_layer,
            project_dir,
        )

        def show_cwt_circles(result: MeasurementResult) -> None:
            if result.cwt is None:
                return
            sample_name, cwt_vis_data, cwt_path = result.cwt
            create_or_update_cwt_circles_layer(
                viewer=self._viewer,
                sample_name=sample_name,
                cwt_vis_data=cwt_vis_data,
                scale=getattr(cells_layer, "scale", None),
                file_path=cwt_path,
            )

        handle.connect(on_result=show_cwt_circles)
        return handle
