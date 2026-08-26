from pathlib import Path
from typing import TYPE_CHECKING, Optional, ContextManager
from contextlib import contextmanager

import cv2
import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    Label,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._utils import make_rings_colormap
from napari_roxas_ai._utils._callback_manager import (
    register_layer_callback,
    unregister_layer_callback,
)

if TYPE_CHECKING:
    import napari

settings = SettingsManager()

SEGMENTATION_MODULE_PATH = (
    Path(__file__).parent.parent.absolute() / "_segmentation"
)
RINGS_MODELS_PATH = SEGMENTATION_MODULE_PATH / "_models" / "_rings"


def rearrange_coordinates(coords: list) -> list:
    """
    Rearrange coordinates from left to right if needed.
    Args:
        coords (list): List of coordinates.
    Returns:
        list: Rearranged coordinates.
    """

    if coords[0][1] > coords[-1][1]:
        coords = coords[::-1]
    return coords


def outside_rings_deletion(
    rings_table: pd.DataFrame, shape: tuple
) -> pd.DataFrame:
    """
    Delete rings that are outside the rings raster.
    Args:
        rings_table (pd.DataFrame): DataFrame containing the rings data.
        shape (tuple): Shape of the image (height, width).
    Returns:
        pd.DataFrame: Updated rings table with only valid rings.
    """

    # Check if any point of the ring is inside the mask
    for i, coords in rings_table["RBXY"].items():
        coords = np.array(coords)
        is_any_valid = np.any(
            (coords[:, 0] >= 0)
            & (coords[:, 0] <= shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] <= shape[1])
        )
        if not is_any_valid:
            rings_table.drop(i, inplace=True)
    return rings_table


def horizontal_rings_completion(coords: list, width: int) -> list:
    """
    This function takes a list of coordinates and a width, and ensures that the
    coordinates start at (x, 0) and end at (x, width).
    This allows to have complete rings across the entire width of the image.
    Straight lines to the edges is not the best approximation, but it allows to avoid crossing rings resulting from other interpolations.
    Args:
        coords (list): List of coordinates.
        width (int): Width of the image.
    Returns:
        list: Completed coordinates.
    """

    if coords[0][1] > 0:
        coords = [[coords[0][0], 0]] + coords
    if coords[-1][1] < width:
        coords = coords + [[coords[-1][0], width]]
    return coords


def horizontal_rings_clippping(coords: list, width: int) -> list:
    """
    Clip the coordinates of the rings to a specified width.
    This ensures that all ring boundary vertices stay within the image boundaries [0, width].

    Args:
        coords (list): List of coordinates (y, x).
        width (int): Width of the image.
    Returns:
        list: Clipped coordinates.
    """
    if not coords:
        return []

    coords = np.array(coords)

    # Find the indices of points that are at or beyond the left (x<=0) and right (x>=width) boundaries
    left_points = np.where(coords[:, 1] <= 0)[0]
    right_points = np.where(coords[:, 1] >= width)[0]

    # Safety check: if no points are at or beyond boundaries, use the full range.
    # This can happen after a lasso-delete operation removes vertices near the image edges,
    # leaving only internal points. We must avoid IndexError by checking if boundary indices exist.
    start_idx = left_points[-1] if len(left_points) > 0 else 0
    end_idx = right_points[0] + 1 if len(right_points) > 0 else len(coords)

    # Slice the coordinates to only include the relevant range within the boundaries
    coords = coords[start_idx:end_idx]

    # Final safeguard: Ensure we don't operate on empty arrays and clip everything precisely to [0, width]
    if len(coords) > 0:
        coords[:, 1] = np.clip(coords[:, 1], 0, width)

    return coords.tolist()


def calculate_polygon_area(coords: list, width: int) -> int:
    """
    This function calculates the number of pixels that would be drawn
    for a polygon defined by the given coordinates using the Shoelace formula.
    Args:
        coords (list): List of coordinates.
        width (int): Width of the canvas.
    Returns:
        int: Number of pixels that would be drawn for the polygon.
    """

    coords = np.array([[0, 0]] + coords + [[0, width]])
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return int(area)


def rasterize_rings(
    rings_table: pd.DataFrame, image_shape: tuple
) -> np.ndarray:
    """
    Rasterize the rings from the rings table into a 2D array.
    Args:
        rings_table (pd.DataFrame): DataFrame containing the rings data.
        image_shape (tuple): Shape of the image (height, width).
    Returns:
        np.ndarray: Rasterized rings as a 2D array.
    """

    rings_raster = np.ones(image_shape) * settings.get(
        "rasterization.uncomplete_ring_value"
    )
    previous_boundary = np.flip(
        np.array([[0, 0], [0, image_shape[1]]]).round().astype("int32"), axis=1
    )[::-1]

    for _i, row in rings_table.iterrows():
        coords = np.flip(
            np.array(row["RBXY"]).round().astype("int32"),
            axis=1,
        )
        value = (
            row["YEAR"]
            if row["enabled"]
            else settings.get("rasterization.uncomplete_ring_value")
        )
        cv2.fillPoly(
            rings_raster, [np.vstack([previous_boundary, coords])], value
        )
        previous_boundary = coords[::-1]
    return rings_raster.astype("int32")


def update_rings_geometries(
    rings_table: pd.DataFrame, last_year: int, image_shape: tuple
) -> tuple:
    """
    Update the rings table with the new geometries and rasterize the rings.
    Args:
        rings_table (pd.DataFrame): DataFrame containing the rings data.
        last_year (int): Year of the last complete ring.
        image_shape (tuple): Shape of the image (height, width).
    Returns:
        tuple: Updated rings table, rasterized rings, and colormap.
    """

    # Rearrange coordinates from left to right if needed
    rings_table["RBXY"] = rings_table["RBXY"].apply(rearrange_coordinates)

    # Ensure rings are inside the image
    rings_table = outside_rings_deletion(rings_table, image_shape)

    # Ensure complete rings
    rings_table["RBXY"] = rings_table["RBXY"].apply(
        lambda x: horizontal_rings_completion(x, image_shape[1])
    )

    # Clip rings to the image width
    rings_table["RBXY"] = rings_table["RBXY"].apply(
        lambda x: horizontal_rings_clippping(x, image_shape[1])
    )

    # Sort new rings chronologically by using the area of the polygon formed with the ring and the image top edge
    rings_table["cells_above"] = rings_table["RBXY"].apply(
        lambda x: calculate_polygon_area(x, image_shape[1])
    )
    rings_table = (
        rings_table.sort_values(by="cells_above", ascending=True)
        .reset_index(drop=True)
        .rename_axis("id")
    )

    rings_table["YEAR"] = [
        a + 1 for a in range(last_year - len(rings_table), last_year)
    ]

    # Disable rings (by default, the first ring is considered uncomplete and is disabled)
    if "enabled" not in rings_table.columns:
        rings_table["enabled"] = True
        # By default, mark the first ring as uncomplete only for legacy inputs
        rings_table.loc[0, "enabled"] = False

    # Rings_rasterization
    rings_raster = rasterize_rings(rings_table, image_shape)

    # Build a new colormap
    unique_rings_raster_values = np.unique(rings_raster)
    colormap = make_rings_colormap(unique_rings_raster_values)

    return rings_table, rings_raster, colormap


def interpolate_row_at_col(coords_rc: list, col_query: float) -> float:
    # used to place the YEAR label vertically centered *inside* a ring at the left margin:
    # take each ring boundary polyline, compute its row-position at a fixed column (near the left edge),
    # then the ring center is halfway between boundary i and i+1 at that same column.
    pts = np.asarray(coords_rc, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return float("nan")

    rows = pts[:, 0]
    cols = pts[:, 1]

    order = np.argsort(cols)
    cols = cols[order]
    rows = rows[order]

    if col_query <= cols[0]:
        return float(rows[0])
    if col_query >= cols[-1]:
        return float(rows[-1])

    j = int(np.searchsorted(cols, col_query) - 1)
    c0, c1 = float(cols[j]), float(cols[j + 1])
    r0, r1 = float(rows[j]), float(rows[j + 1])

    if c1 == c0:
        return float(r0)

    t = (col_query - c0) / (c1 - c0)
    return float(r0 + t * (r1 - r0))


class RingsLayerEditorWidget(Container):
    @property
    def _input_layer(self) -> Optional["napari.layers.Labels"]:
        """Get the single valid ring layer currently in the viewer."""
        valid_layers = self._get_valid_layers()
        return valid_layers[0] if valid_layers else None

    def _get_rings_data(self, source_layer: "napari.layers.Layer") -> pd.DataFrame:
        """
        Extract RBXY, YEAR, and enabled status from a layer.
        Handles both Labels (via features) and Shapes (via data/features).
        """
        if source_layer is None:
            return pd.DataFrame()

        # Extract features/data based on layer type
        if hasattr(source_layer, "features") and source_layer.features is not None:
            df = source_layer.features.copy()
        else:
            df = pd.DataFrame()

        # Handle RBXY extraction
        if "RBXY" not in df.columns:
            if isinstance(source_layer, napari.layers.Shapes):
                df["RBXY"] = [coords.tolist() for coords in source_layer.data]
            else:
                return pd.DataFrame()

        # Handle YEAR assignment if missing
        if "YEAR" not in df.columns:
            n = len(df)
            last_year = 9999
            if source_layer.metadata and "rings_outmost_complete_year" in source_layer.metadata:
                last_year = int(source_layer.metadata["rings_outmost_complete_year"])
            df["YEAR"] = list(range(last_year - n + 1, last_year + 1))

        # Handle enabled status
        if "enabled" not in df.columns:
            df["enabled"] = True

        return df

    def _set_ui_editing_mode(self, editing: bool) -> None:
        """Toggle UI visibility for editing/idle states."""
        self._is_editing = editing
        self._edit_rings_geometries_button.visible = not editing
        self._last_year_row.visible = not editing
        
        self._cancel_rings_geometries_button.visible = editing
        self._apply_rings_geometries_button.visible = editing
        
        # Explicitly toggle visibility of lasso and rerun model rows
        self._lasso_container.visible = editing
        self._rerun_params_row.visible = editing
        self._rerun_model_button.visible = editing
        
        # Ensure lasso selection elements are properly handled
        if editing:
            # Force inner widgets to be visible as well to overcome potential propagation issues
            self._lasso_selection_checkbox.visible = True
            # Show the deletion button only if the checkbox is checked
            self._delete_lasso_vertices_button.visible = self._lasso_selection_checkbox.value
        else:
            # Reset checkbox and hide inner widgets when exiting edit mode
            self._lasso_selection_checkbox.value = False
            self._lasso_selection_checkbox.visible = False
            self._delete_lasso_vertices_button.visible = False

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(labels=False)
        self._viewer = viewer
        self.settings = SettingsManager()

        self._layer_callback = None
        self._is_editing = False
        self._is_updating = False  # Global lock for scenegraph modifications
        
        # Debounce timer for Rings Years updates to prevent scenegraph collisions during editing
        self._years_update_timer = QTimer()
        self._years_update_timer.setSingleShot(True)
        self._years_update_timer.timeout.connect(self._update_rings_years_layer)

        # --- SELECTION & VISIBILITY STATE ---
        # Stores indices of vertices selected via lasso: shape_index -> set of vertex_indices
        self._selected_vertices = {}
        # Stores coordinates of all vertices deleted during the current edit session
        self._all_deleted_points = []
        # Stores visibility of .cells and .rings layers before editing to restore them later
        self._layer_visibility_states = {}

        # --- UI WIDGETS ---
        # Explicit label for Rerun Model section to ensure consistent appearance
        self._rerun_model_header = Label(value="Rerun Model from year:", visible=False)
        self._rerun_model_header.native.setFixedWidth(140)  # Ensure consistent width for alignment

        # Create spinbox for the last year
        year_value = (
            self._input_layer.metadata[
                "rings_outmost_complete_year"
            ]
            if self._input_layer
            else 9999
        )
        self._last_year_spinbox = SpinBox(
            value=year_value,
            label="Last Complete Ring Year",
            min=-999999,
            max=9999,
            step=1,
        )

        self._last_year_update_button = PushButton(
            text="Update Year",
            visible=True,
        )
        self._last_year_update_button.changed.connect(self._update_layer_year)

        # Create a button to create the rings working layer
        self._edit_rings_geometries_button = PushButton(
            text="Edit Ring Boundaries"
        )
        self._edit_rings_geometries_button.changed.connect(
            self._edit_rings_geometries
        )

        # Create a button to cancel the changes
        self._cancel_rings_geometries_button = PushButton(
            text="Cancel Ring Changes", visible=False
        )
        self._cancel_rings_geometries_button.changed.connect(
            self._cancel_rings_geometries
        )

        # Create a button to apply the changes
        self._apply_rings_geometries_button = PushButton(
            text="Apply Ring Changes", visible=False
        )
        self._apply_rings_geometries_button.changed.connect(
            self._apply_rings_geometries
        )

        # --- RERUN MODEL UI ---
        # Line 1: Rerun Model from year: --- spinbox --- Rings Model --- combo
        self._rerun_model_year_spinbox = SpinBox(
            value=9999,
            label="Rerun from Year",  # Use a simple label
            min=-100000,
            max=10000,
            step=1,
        )

        self._rings_model_weights_file = ComboBox(
            choices=tuple(
                path.name for path in Path(RINGS_MODELS_PATH).iterdir()
            ),
            label="Model",
        )

        # Horizontal row for the rerun model parameters
        self._rerun_params_row = Container(
            widgets=[
                self._rerun_model_header,
                self._rerun_model_year_spinbox,
                self._rings_model_weights_file,
            ],
            layout="horizontal",
            labels=True,  # Enable labels for Year and Model
            visible=False,
        )
        # Ensure the header doesn't have a double label effect
        self._rerun_model_header.label = ""

        # Line 2: Run
        self._rerun_model_button = PushButton(
            text="Run",
            visible=False,
        )
        self._rerun_model_button.changed.connect(self._rerun_model_from_year)

        # --- LASSO SELECTION UI ---
        self._lasso_selection_checkbox = CheckBox(
            label="Lasso Selection Mode",
            value=False,
            visible=False,
        )
        self._lasso_selection_checkbox.changed.connect(
            lambda val: self._toggle_lasso_selection_mode(val)
        )

        self._delete_lasso_vertices_button = PushButton(
            text="Delete Vertices in Lasso",
            visible=False,
        )
        self._delete_lasso_vertices_button.changed.connect(
            self._execute_lasso_deletion
        )

        # Horizontal row for lasso selection controls
        self._lasso_container = Container(
            widgets=[
                self._lasso_selection_checkbox,
                self._delete_lasso_vertices_button,
            ],
            layout="horizontal",
            labels=False,
            visible=False,
        )
        # Ensure the container is explicitly hidden at start
        self._lasso_container.visible = False
        
        # Force alignment to the left margin and consistent spacing
        if hasattr(self._lasso_container.native, "layout"):
            layout = self._lasso_container.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(10)

        self._last_year_row = Container(
            widgets=[
                self._last_year_spinbox,
                self._last_year_update_button,
            ],
            layout="horizontal",
        )

        self.extend(
            [
                self._edit_rings_geometries_button,
                self._cancel_rings_geometries_button,
                self._apply_rings_geometries_button,
                self._lasso_container,
                self._rerun_params_row,
                self._rerun_model_button,
                self._last_year_row,
            ]
        )

        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)

        # --- INITIALIZATION ---
        self._connect_layer_callback()
        # Ensure 'Rings Years' layer is initialized if a rings layer already exists.
        # A small delay ensures that napari has finished its initial layer setup.
        # Use a slightly longer delay and immediate follow-up to ensure visibility.
        self._years_update_timer.start(500)
        QTimer.singleShot(1000, lambda: self._years_update_timer.start(10))

    @contextmanager
    def _pause_rendering(self) -> ContextManager[None]:
        """
        Context manager to pause Vispy rendering.
        
        This is a CRITICAL stability component. Direct layer modifications in napari
        (adding/removing/moving layers) trigger immediate redraws in Vispy. If these
        modifications happen while Vispy is already in its paint cycle, it causes
        'RuntimeError: Error drawing visual' or 'OSError' during scenegraph updates.
        
        By setting canvas._pause_scene_graph = True, we prevent Vispy from attempting
        to draw until we've finished all our synchronous layer operations.
        """
        canvas = None
        try:
            # Safely access the qt_viewer and its canvas using getattr to avoid deprecation warnings
            # and to handle different napari versions.
            qt_viewer = getattr(self._viewer.window, "qt_viewer", None)
            if qt_viewer is None:
                qt_viewer = getattr(self._viewer.window, "_qt_viewer", None)

            if qt_viewer is not None and hasattr(qt_viewer, "canvas"):
                canvas = qt_viewer.canvas
                if canvas is not None:
                    canvas._pause_scene_graph = True
            yield
        finally:
            if canvas is not None:
                canvas._pause_scene_graph = False
                # Force a single clean redraw once modifications are complete.
                if hasattr(canvas, "update"):
                    canvas.update()
                elif hasattr(canvas, "native") and hasattr(canvas.native, "update"):
                    canvas.native.update()

    def _on_layer_change(self, event=None):
        """Called when layers are added/removed in the viewer.
        Ensures persistent 'Rings Years' labels stay in sync with the current layers.
        """
        # Guard against recursive updates from "Rings Years" itself
        if event is not None and hasattr(event, "value") and event.value is not None:
            if getattr(event.value, "name", "") == "Rings Years":
                return

        # Always trigger years update when layer list changes
        self._years_update_timer.start(50)

        if self._is_editing:
            return

        if not self._get_valid_layers():
            self._disconnect_layer_callback()
            self._last_year_spinbox.value = 9999
            return

        self._connect_layer_callback()

    def _deferred_remove_layer(self, name: str) -> None:
        """
        Safely remove a layer by name in the next event loop iteration.

        Direct removal of layers during complex event emissions (like inside a button 
        click handler that also modifies layer data) can lead to Tracebacks in 
        Napari's evented list or Vispy's scenegraph manager.
        
        Using QTimer.singleShot(50, ...) ensures the removal happens when the current
        processing cycle is complete and the GUI event loop is idle.
        """
        def _rm():
            if name in self._viewer.layers:
                try:
                    # Before removing, we hide the layer to stop Vispy from 
                    # trying to draw it while it's being purged from the list.
                    self._viewer.layers[name].visible = False
                    self._viewer.layers.remove(name)
                except Exception:
                    # Catch all to prevent crashes if the layer was already 
                    # removed by another concurrent event.
                    pass

        QTimer.singleShot(50, _rm)

    def _get_valid_layers(self, widget=None) -> list:
        """Get layers that are both Labels type and match the rings file extension."""
        rings_extension = settings.get("file_extensions.rings_file_extension")[
            0
        ]
        valid_layers = []

        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name.endswith(
                rings_extension
            ):
                valid_layers.append(layer)

        return valid_layers

    def _connect_layer_callback(self):
        """Connect callback to the currently selected layer."""
        # Clean up any previous callback first
        self._disconnect_layer_callback()

        if self._input_layer is not None:
            # Connect to the layer's events using the shared callback manager
            self._layer_callback = register_layer_callback(
                self._input_layer, self, self._on_layer_data_change
            )
            self._update_year_spinbox()
            self._years_update_timer.start(50)

    def _disconnect_layer_callback(self):
        """Disconnect callback from the previously selected layer."""
        if (
            self._input_layer is not None
            and self._layer_callback is not None
        ):
            unregister_layer_callback(self._input_layer, self)
            self._layer_callback = None

    def _on_layer_data_change(self, event=None):
        """Called when the data in the selected layer changes."""
        # Only respond to data, metadata, or features changes
        if (
            event.type == "data"
            or event.type == "metadata"
            or event.type == "features"
        ):

            # Update the spinbox value
            self._update_year_spinbox()
            # Defer updating years layer to avoid scenegraph issues during event emission
            self._years_update_timer.start(50)


    def _update_layer_year(self) -> None:
        """Update the last year value in the layer metadata."""
        if self._input_layer:
            layer = self._input_layer
            layer.metadata["rings_outmost_complete_year"] = (
                self._last_year_spinbox.value
            )
            new_rings_table, new_rings_raster, new_colormap = (
                update_rings_geometries(
                    rings_table=layer.features,
                    last_year=self._last_year_spinbox.value,
                    image_shape=layer.data.shape,
                )
            )
            layer.data = new_rings_raster
            layer.features = new_rings_table
            layer.colormap = new_colormap
            layer.events.metadata()
            layer.events.data()
            layer.events.features()

    def _edit_rings_geometries(self) -> None:
        """Run the segmentation analysis in a separate thread."""
        # Get the selected input layer
        input_layer = self._input_layer
        if not input_layer:
            QMessageBox.warning(None, "Error", "No valid rings layer found")
            return

        # --- LAYER VISIBILITY MANAGEMENT ---
        # Hide .cells and .rings layers to reduce clutter during ring boundary editing.
        # Store their original visibility state to restore it when editing is finished.
        self._layer_visibility_states = {}
        for layer in self._viewer.layers:
            if layer.name.endswith(".cells") or layer.name.endswith(".rings"):
                self._layer_visibility_states[layer.name] = layer.visible
                layer.visible = False

        # --- UI UPDATE ---
        self._set_ui_editing_mode(True)

        self._current_input_layer = input_layer

        # If there is already an edit session open, remove old helper layers first
        if "Rings Modification" in self._viewer.layers:
            self._deferred_remove_layer("Rings Modification")

        # Build a DF in the same logical order as the annotated export
        df = input_layer.features.copy()

        # Prefer YEAR ordering; otherwise fall back to cells_above if present
        if "YEAR" in df.columns:
            df = df.sort_values("YEAR").reset_index(drop=True)
            # set value here to run model from the first year in the table by default
            self._rerun_model_year_spinbox.value = df["YEAR"].iloc[0]
        elif "cells_above" in df.columns:
            df = df.sort_values("cells_above").reset_index(drop=True)

        # Do NOT drop disabled rings; the top "uncomplete" boundary is required
        # to preserve the red/uncomplete region after rasterization.
        if "enabled" in df.columns:
            df["enabled"] = df["enabled"].fillna(True)

        if df.empty or "RBXY" not in df.columns or "YEAR" not in df.columns:
            show_info("No valid rings to edit")
            return

        # Simplify boundary coordinates using cv2.approxPolyDP if tolerance is > 0
        simplified_boundary_lines = []
        keep_rows = []
        tolerance = settings.get("vectorization.rings_tolerance")
        for i, coords in enumerate(df["RBXY"].tolist()):
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue

            if tolerance > 0:
                approx = cv2.approxPolyDP(
                    np.array(coords, dtype=np.float32),
                    epsilon=tolerance,
                    closed=False,
                )
                approx = np.squeeze(approx)
            else:
                approx = np.array(coords, dtype=np.float32)

            if approx.ndim != 2 or approx.shape[0] < 2:
                continue

            simplified_boundary_lines.append(approx.tolist())
            keep_rows.append(i)

        # Ensure features rows match the number of shapes
        df = df.iloc[keep_rows].reset_index(drop=True)
        if df.empty:
            show_info("No valid rings to edit")
            return

        # Create the editable Shapes layer
        shapes_layer = self._viewer.add_shapes(
            simplified_boundary_lines,
            shape_type="path",
            edge_color=settings.get("vectorization.rings_edge_color"),
            edge_width=settings.get("vectorization.rings_edge_width"),
            opacity=1,
            name="Rings Modification",
            scale=input_layer.scale,
            features={
                "YEAR": df["YEAR"].tolist(),
                "enabled": (
                    df["enabled"].fillna(True).tolist()
                    if "enabled" in df.columns
                    else [True] * len(df)
                ),
            },
        )

        self._update_rings_years_layer()

        self._viewer.layers.selection.active = shapes_layer

        self._selected_vertices = {}
        self._all_deleted_points = []

        self._update_year_spinbox()

    def _cancel_rings_geometries(self) -> None:
        """Cancel the changes made to the input layer and cleanup the edit session."""
        if self._is_updating:
            return
        self._is_updating = True

        try:
            with self._pause_rendering():
                # Cleanup temporary modification and selection layers
                # We hide them first to stop Vispy rendering before they are removed.
                if "Rings Modification" in self._viewer.layers:
                    self._viewer.layers["Rings Modification"].visible = False
                    self._viewer.layers.remove("Rings Modification")
                
                if "Lasso Selection" in self._viewer.layers:
                    self._viewer.layers["Lasso Selection"].visible = False
                    self._viewer.layers.remove("Lasso Selection")
                if "Selected Vertices" in self._viewer.layers:
                    self._viewer.layers["Selected Vertices"].visible = False
                    self._viewer.layers.remove("Selected Vertices")
                self._selected_vertices = {}
                self._all_deleted_points = []

            # Reset UI widgets visibility and state (outside pause)
            self._set_ui_editing_mode(False)
            self._lasso_selection_checkbox.value = False
            self._delete_lasso_vertices_button.visible = False

            # Restore original visibility of .cells and .rings layers
            def restore_visibility():
                try:
                    for layer_name, visible in self._layer_visibility_states.items():
                        if layer_name in self._viewer.layers:
                            self._viewer.layers[layer_name].visible = visible
                    self._layer_visibility_states = {}
                    
                    # Force a canvas update to ensure the restored layers are redrawn
                    qt_viewer = getattr(self._viewer.window, "qt_viewer", None)
                    if qt_viewer is None:
                        qt_viewer = getattr(self._viewer.window, "_qt_viewer", None)
                    
                    if qt_viewer is not None and hasattr(qt_viewer, "canvas"):
                        canvas = qt_viewer.canvas
                        if canvas is not None:
                            if hasattr(canvas, "update"):
                                canvas.update()
                            elif hasattr(canvas, "native") and hasattr(canvas.native, "update"):
                                canvas.native.update()
                except Exception:
                    pass

            # Defer visibility restoration slightly to let scenegraph settle from removals
            QTimer.singleShot(100, restore_visibility)

            show_info("Rings geometries modification cancelled")
            self._years_update_timer.start(200)

        except Exception as e:
            show_info(f"Error cancelling ring geometries: {str(e)}")
        finally:
            self._is_updating = False

    def _apply_rings_geometries(self) -> None:
        """Apply the changes to the input layer and finish the edit session."""
        if self._is_updating:
            return
        self._is_updating = True

        try:
            with self._pause_rendering():
                if "Rings Modification" not in self._viewer.layers:
                    show_info("No editing layer found")
                    return

                layer = self._viewer.layers["Rings Modification"]
                layer.visible = False # Hide immediately to stop rendering

                # Prepare the new geometries for rasterization
                rings_table = self._get_rings_data(layer)

                # Cleanup temporary edit layers.
                # Explicit hiding before removal prevents Vispy from drawing partially purged states.
                self._viewer.layers.remove("Rings Modification")
                
                if "Lasso Selection" in self._viewer.layers:
                    self._viewer.layers["Lasso Selection"].visible = False
                    self._viewer.layers.remove("Lasso Selection")
                if "Selected Vertices" in self._viewer.layers:
                    self._viewer.layers["Selected Vertices"].visible = False
                    self._viewer.layers.remove("Selected Vertices")
                self._selected_vertices = {}
                self._all_deleted_points = []

                input_layer = self._input_layer
                if input_layer is None:
                    show_info("No valid rings layer found to apply geometries")
                    return

                # Update the rings layer with the new geometries
                new_rings_table, new_rings_raster, new_colormap = (
                    update_rings_geometries(
                        rings_table=rings_table,
                        last_year=input_layer.metadata[
                            "rings_outmost_complete_year"
                        ],
                        image_shape=input_layer.data.shape,
                    )
                )

                # Update the rings layer with the new geometries
                input_layer.data = new_rings_raster
                input_layer.features = new_rings_table
                input_layer.colormap = new_colormap

            # UI Update (outside pause)
            self._set_ui_editing_mode(False)
            self._lasso_selection_checkbox.value = False
            self._delete_lasso_vertices_button.visible = False

            # Restore original visibility of .cells and .rings layers
            def restore_visibility():
                try:
                    for layer_name, visible in self._layer_visibility_states.items():
                        if layer_name in self._viewer.layers:
                            self._viewer.layers[layer_name].visible = visible
                    self._layer_visibility_states = {}
                    
                    # Force a canvas update to ensure the restored layers are redrawn
                    qt_viewer = getattr(self._viewer.window, "qt_viewer", None)
                    if qt_viewer is None:
                        qt_viewer = getattr(self._viewer.window, "_qt_viewer", None)
                    
                    if qt_viewer is not None and hasattr(qt_viewer, "canvas"):
                        canvas = qt_viewer.canvas
                        if canvas is not None:
                            if hasattr(canvas, "update"):
                                canvas.update()
                            elif hasattr(canvas, "native") and hasattr(canvas.native, "update"):
                                canvas.native.update()
                except Exception:
                    pass

            # Defer visibility restoration slightly to let scenegraph settle from removals
            QTimer.singleShot(100, restore_visibility)

            # Defer updating years layer to ensure all other layer removals/modifications are settled
            self._years_update_timer.start(400)
            show_info("Ring boundaries successfully updated")

        except Exception as e:
            show_info(f"Error applying ring geometries: {str(e)}")
        finally:
            self._is_updating = False

    def _toggle_lasso_selection_mode(self, enabled: bool) -> None:
        """
        Toggle the lasso selection mode.
        When enabled, a temporary yellow shapes layer is created to allow users
        to draw polygons defining areas for vertex deletion.
        """
        # Toggle visibility of the deletion button based on checkbox state
        self._delete_lasso_vertices_button.visible = enabled

        if enabled:
            # Use the current layer's scale to ensure the lasso matches image coordinates accurately.
            # Default to [1.0, 1.0] if the layer is not available or scale is not set.
            scale = [1.0, 1.0]
            try:
                if hasattr(self, "_current_input_layer") and self._current_input_layer is not None:
                    scale = self._current_input_layer.scale
                elif self._input_layer is not None:
                    scale = self._input_layer.scale
            except Exception:
                pass

            if "Lasso Selection" not in self._viewer.layers:
                self._viewer.add_shapes(
                    name="Lasso Selection",
                    shape_type="polygon",
                    edge_color="yellow",
                    face_color=[1, 1, 0, 0.3],
                    edge_width=2,
                    scale=scale,
                )
            
            if "Lasso Selection" in self._viewer.layers:
                lasso_layer = self._viewer.layers["Lasso Selection"]
                # Ensure the tool is selected
                self._viewer.layers.selection.active = lasso_layer
                lasso_layer.mode = "add_polygon_lasso"
                
            self._delete_lasso_vertices_button.visible = True
            show_info("Lasso Mode: Draw polygons and click 'Delete Vertices in Lasso' (or press 'Delete')")
        else:
            # Revert to standard direct selection mode on the rings modification layer.
            # We defer the lasso layer removal to ensure the viewer state is stable.
            if "Lasso Selection" in self._viewer.layers:
                self._deferred_remove_layer("Lasso Selection")
            
            # Explicitly set "Rings Modification" as active layer after lasso mode is disabled
            if "Rings Modification" in self._viewer.layers:
                try:
                    edit_layer = self._viewer.layers["Rings Modification"]
                    self._viewer.layers.selection.active = edit_layer
                    edit_layer.mode = "direct"
                except (ValueError, KeyError, IndexError):
                    pass
            self._delete_lasso_vertices_button.visible = False

        # Bind the Delete key to our custom handler to support lasso-delete via keyboard.
        # We deselect lasso polygons before execution to prevent napari from deleting the lasso shape itself.
        @self._viewer.bind_key("Delete", overwrite=True)
        def _delete_selected(viewer):
            if not self._is_editing:
                return

            if self._lasso_selection_checkbox.value:
                if "Lasso Selection" in self._viewer.layers:
                    self._viewer.layers["Lasso Selection"].selected_data = set()
                
                self._execute_lasso_deletion()
                return

    def _execute_lasso_deletion(self) -> None:
        """
        Main entry point for lasso-based deletion.
        Identifies vertices within the lasso area and removes them from the ring boundaries.
        """
        if not self._is_editing or not self._lasso_selection_checkbox.value:
            return

        try:
            self._select_vertices_in_lasso()
            
            if self._selected_vertices:
                self._delete_selected_vertices()
            else:
                # Clear the lasso polygons even if no vertices were found to maintain UI flow
                if "Lasso Selection" in self._viewer.layers:
                    self._viewer.layers["Lasso Selection"].data = []
                show_info("No vertices found inside the lasso area")
        except Exception as e:
            show_info(f"Error during lasso deletion: {str(e)}")
        finally:
            # Reset the lasso selection mode checkbox as requested by the user.
            # This will also trigger the cleanup via _toggle_lasso_selection_mode.
            self._lasso_selection_checkbox.value = False
            # Ensure the horizontal container stays visible since we're still in editing mode.
            self._lasso_container.visible = True
            
            # Re-activate the Rings Modification layer to ensure it stays selected after deletion markers are shown
            if "Rings Modification" in self._viewer.layers:
                def _restore_active():
                    if "Rings Modification" in self._viewer.layers:
                        try:
                            self._viewer.layers.selection.active = self._viewer.layers["Rings Modification"]
                        except Exception:
                            pass
                QTimer.singleShot(100, _restore_active)

    def _select_vertices_in_lasso(self) -> None:
        """
        Iterates through all drawn lasso polygons and identifies all ring boundary vertices
        that fall within their boundaries using point-in-polygon tests.
        """
        if "Lasso Selection" not in self._viewer.layers:
            return
        
        lasso_layer = self._viewer.layers["Lasso Selection"]
        if len(lasso_layer.data) == 0:
            return
        
        if "Rings Modification" not in self._viewer.layers:
            return
            
        edit_layer = self._viewer.layers["Rings Modification"]
        
        for lasso_poly in lasso_layer.data:
            for i, shape_data in enumerate(edit_layer.data):
                if i not in self._selected_vertices:
                    self._selected_vertices[i] = set()
                
                for j, vertex in enumerate(shape_data):
                    try:
                        if self._is_point_in_polygon(vertex, lasso_poly):
                            self._selected_vertices[i].add(j)
                    except Exception:
                        continue
        
        self._highlight_selected_vertices()

    def _is_point_in_polygon(self, point, polygon) -> bool:
        """
        Helper method using OpenCV's pointPolygonTest to determine if a (y, x) coordinate
        is inside a polygon defined by a set of (y, x) vertices.
        """
        poly_pts = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(poly_pts, (float(point[0]), float(point[1])), False) >= 0

    def _highlight_selected_vertices(self) -> None:
        """
        Creates a temporary red 'Points' layer to provide visual feedback for vertices
        identified by the lasso tool before they are actually deleted.
        """
        if "Rings Modification" not in self._viewer.layers:
            return
            
        points = []
        # Add points that are currently marked for deletion in the current lasso selection
        edit_layer = self._viewer.layers["Rings Modification"]
        for shape_idx, vertex_indices in self._selected_vertices.items():
            if shape_idx >= len(edit_layer.data):
                continue
            shape_data = edit_layer.data[shape_idx]
            for v_idx in vertex_indices:
                if v_idx < len(shape_data):
                    points.append(shape_data[v_idx])
        
        # Add points that were deleted in previous lasso operations within the same edit session
        if self._all_deleted_points:
            points.extend(self._all_deleted_points)
        
        if points:
            # We defer the addition of feedback points to ensure it doesn't collide
            # with the ongoing lasso drawing event processing.
            def _add_selected_points():
                if not self._is_editing:
                    return
                try:
                    # Update data if layer exists, otherwise create it
                    if "Selected Vertices" in self._viewer.layers:
                        points_layer = self._viewer.layers["Selected Vertices"]
                        points_layer.data = points
                        points_layer.refresh()
                    elif points:
                        points_layer = self._viewer.add_points(
                            points,
                            name="Selected Vertices",
                            size=75,
                            face_color="yellow",
                            border_color="black",
                            scale=edit_layer.scale,
                        )
                        # Ensure feedback points are on top of other layers
                        try:
                            p_idx = self._viewer.layers.index(points_layer)
                            self._viewer.layers.move(p_idx, -1)
                        except (ValueError, KeyError, IndexError):
                            pass

                    # Maintain active selection on the appropriate tool layer.
                    # This prevents the feedback layer from stealing focus.
                    if self._lasso_selection_checkbox.value:
                        if "Lasso Selection" in self._viewer.layers:
                            self._viewer.layers.selection.active = self._viewer.layers["Lasso Selection"]
                    else:
                        self._viewer.layers.selection.active = edit_layer
                except Exception:
                    pass

            QTimer.singleShot(50, _add_selected_points)
        else:
            # If no points are left to highlight, remove the layer if it exists
            if "Selected Vertices" in self._viewer.layers:
                self._deferred_remove_layer("Selected Vertices")

    def _delete_selected_vertices(self) -> None:
        """
        Performs the actual removal of vertices from the 'Rings Modification' Shapes layer.
        Reconstructs the boundary paths and their associated metadata (YEAR, enabled).
        Shapes that drop below the 2-vertex minimum are automatically removed.
        """
        if not self._selected_vertices:
            return

        if "Rings Modification" not in self._viewer.layers:
            return
            
        edit_layer = self._viewer.layers["Rings Modification"]
        new_data = []
        new_features_dict = {k: [] for k in edit_layer.features.keys()}
        
        changed = False
        for i, shape_data in enumerate(edit_layer.data):
            indices_to_delete = self._selected_vertices.get(i, set())
            
            if indices_to_delete:
                changed = True
                shape_list = shape_data.tolist() if hasattr(shape_data, "tolist") else list(shape_data)
                
                # Capture coordinates of vertices being deleted to maintain visual feedback
                for v_idx in indices_to_delete:
                    if v_idx < len(shape_list):
                        self._all_deleted_points.append(shape_list[v_idx])

                updated_shape = [p for j, p in enumerate(shape_list) if j not in indices_to_delete]
                
                # Keep only paths with at least 2 vertices
                if len(updated_shape) >= 2:
                    new_data.append(np.array(updated_shape))
                    for k in new_features_dict:
                        new_features_dict[k].append(edit_layer.features[k][i])
            else:
                new_data.append(shape_data)
                for k in new_features_dict:
                    new_features_dict[k].append(edit_layer.features[k][i])

        if changed:
            try:
                # Update layer data and features atomically to maintain synchronization
                edit_layer.data = new_data
                edit_layer.features = pd.DataFrame(new_features_dict)
                edit_layer.refresh()
                show_info("Selected vertices deleted")
            except Exception as e:
                show_info(f"Failed to update layers after deletion: {str(e)}")
        
        # Cleanup selection feedback
        self._selected_vertices = {}
        
        # Refresh highlighting to show accumulated deleted points
        self._highlight_selected_vertices()
        
        # Defer updating years layer to ensure stability after data removal
        self._years_update_timer.start(200)
        
        # Clear the lasso polygons after deletion is complete
        if "Lasso Selection" in self._viewer.layers:
            try:
                self._viewer.layers["Lasso Selection"].data = []
            except Exception:
                pass

    def _update_year_spinbox(self) -> None:
        """Update the year spinbox value based on the selected layer."""
        if self._input_layer:
            layer = self._input_layer
            if "rings_outmost_complete_year" in layer.metadata:
                self._last_year_spinbox.value = layer.metadata[
                    "rings_outmost_complete_year"
                ]
            # Update the rerun model year spinbox with the first year in the table
            if hasattr(layer, "features") and "YEAR" in layer.features.columns:
                first_year = layer.features["YEAR"].min()
                self._rerun_model_year_spinbox.value = int(first_year)

    def _update_rings_years_layer(self) -> None:
        """
        Update or recreate the persistent 'Rings Years' points layer.
        This layer displays the year labels for each ring boundary at the left margin.
        It automatically tracks whether the viewer is in standard mode (Labels layer)
        or editing mode (Shapes layer).
        """
        # Global lock: prevents background updates from colliding with 
        # UI-driven layer modifications (like Apply/Cancel).
        if self._is_updating:
            return
            
        # Do not update year labels while the user is drawing a lasso selection polygon
        # to avoid scenegraph race conditions during high-frequency mouse events.
        if hasattr(self, "_lasso_selection_checkbox") and self._lasso_selection_checkbox.value:
            return

        try:
            self._is_updating = True
            # Wrap in a rendering pause context to ensure the Vispy scenegraph 
            # is stable while we modify the 'Rings Years' layer.
            with self._pause_rendering():
                self._do_update_rings_years_layer()
        except Exception:
            # Silent failure for visual feedback layer to avoid crashing the main UI
            # if a race condition still occurs during the update.
            pass
        finally:
            self._is_updating = False

    def _do_update_rings_years_layer(self) -> None:
        """Actual implementation of Rings Years update."""
        if not hasattr(self, "_viewer") or self._viewer is None:
            return

        # Determine the source of geometry data
        source_layer = None
        if "Rings Modification" in self._viewer.layers:
            source_layer = self._viewer.layers["Rings Modification"]
        else:
            source_layer = self._input_layer

        if source_layer is None:
            valid_layers = self._get_valid_layers()
            if valid_layers:
                source_layer = valid_layers[0]

        if source_layer is None:
            if "Rings Years" in self._viewer.layers:
                self._deferred_remove_layer("Rings Years")
            return

        try:
            df = self._get_rings_data(source_layer)
            if df.empty:
                if "Rings Years" in self._viewer.layers:
                    self._deferred_remove_layer("Rings Years")
                return

            if "RBXY" not in df.columns:
                if "Rings Years" in self._viewer.layers:
                    self._deferred_remove_layer("Rings Years")
                return

            # Placement logic for labels
            scale = source_layer.scale if hasattr(source_layer, "scale") else [1.0, 1.0]
            sx = float(scale[1])
            x_left = 10.0 / sx

            y_on_left = []
            for coords in df["RBXY"].tolist():
                try:
                    y = interpolate_row_at_col(coords, x_left)
                    y_on_left.append(y)
                except Exception:
                    y_on_left.append(0.0)

            centers_r = []
            for i in range(len(y_on_left)):
                upper = y_on_left[i - 1] if i > 0 else 0.0
                centers_r.append(0.5 * (upper + y_on_left[i]))

            years = [str(int(y)) for y in df["YEAR"].tolist()]
            h, w = source_layer.data.shape[:2]
            centers_r = [max(0, min(h - 1, r)) for r in centers_r]

            points_rc = np.column_stack([
                np.array(centers_r, dtype=float),
                np.full(len(df), x_left, dtype=float),
            ])
        except Exception:
            self._years_update_timer.start(500)
            return

        # Check if we can update in-place to avoid flickering
        if "Rings Years" in self._viewer.layers:
            years_layer = self._viewer.layers["Rings Years"]
            # Only update in-place if the number of points is the same
            if len(years_layer.data) == len(points_rc):
                years_layer.data = points_rc
                years_layer.features = {"YEAR": years}
                years_layer.scale = scale
                
                # Refresh text properties
                if hasattr(years_layer, "text"):
                    years_layer.text.visible = True
                    try:
                        years_layer.text.refresh(years_layer.features)
                    except TypeError:
                        years_layer.text.refresh()
                years_layer.refresh()
                return

        # Fallback: remove and recreate the layer if in-place update is not possible
        if "Rings Years" in self._viewer.layers:
            self._viewer.layers.remove("Rings Years")

        # Defer the addition to ensure it happens in a clean event cycle
        def _add_years_layer():
            try:
                if "Rings Years" in self._viewer.layers:
                    return

                self._viewer.add_points(
                    points_rc,
                    name="Rings Years",
                    size=1,
                    opacity=1.0,
                    border_width=0,
                    face_color=[0, 0, 0, 0],
                    border_color=[0, 0, 0, 0],
                    scale=scale,
                    features={"YEAR": years},
                    text={
                        "string": "{YEAR}",
                        "anchor": "upper_left",
                        "translation": [0, 0],
                        "size": 8,
                        "color": "black",
                        "blending": "translucent",
                        "visible": True,
                    },
                )

                years_layer = self._viewer.layers["Rings Years"]
                years_layer.editable = False
                years_layer.visible = True

                # Position years layer: top-most, but below editing layers if they exist
                layers = self._viewer.layers
                try:
                    years_index = layers.index(years_layer)
                    target_idx = len(layers) - 1

                    editing_layers = ["Rings Modification", "Selected Vertices", "Lasso Selection"]
                    for layer_name in editing_layers:
                        if layer_name in layers:
                            # If an editing layer exists, position years just below it
                            target_idx = layers.index(layer_name) - 1
                            break

                    if years_index != target_idx and target_idx >= 0:
                        layers.move(years_index, target_idx)
                except (ValueError, IndexError):
                    pass

                years_layer.refresh()
            except Exception:
                pass

        QTimer.singleShot(50, _add_years_layer)

    def _rerun_model_from_year(self) -> None:
        """
        Reruns the ring detection model starting from a specific boundary identified by its year.
        This allows for local adjustments and refinements without reprocessing the entire image.
        Uses the 'Rings Modification' Shapes layer as the geometry source.
        """
        selected_year = self._rerun_model_year_spinbox.value

        if not self._input_layer:
            show_info("No layer selected")
            return

        # Read from the Shapes editing layer, not the original Labels layer
        if "Rings Modification" not in self._viewer.layers:
            show_info(
                "No editing session active — click 'Edit Ring Boundaries' first"
            )
            return

        edit_layer = self._viewer.layers["Rings Modification"]

        # Build rings_table from the current Shapes layer data using consolidated helper
        rings_table = self._get_rings_data(edit_layer)
        rings_table = rings_table.rename_axis("id")
        rings_table = rings_table.dropna(axis=1, how="all")

        if (
            "YEAR" not in rings_table.columns
            or "RBXY" not in rings_table.columns
        ):
            show_info("No valid rings data found")
            return

        # Sort by YEAR to ensure correct ordering
        rings_table = rings_table.sort_values("YEAR").reset_index(drop=True)

        # Find the boundary index for the selected year
        matching = rings_table[rings_table["YEAR"] == selected_year]
        if matching.empty:
            show_info(
                f"Year {selected_year-1} not found in rings table or does not have a valid end boundary"
            )
            return

        boundary_idx = matching.index[0]

        start_boundary = np.array(rings_table.loc[boundary_idx, "RBXY"])

        # start_boundary is an array of shape (N, 2) with [row, col] coordinates
        cols = start_boundary[:, 1]
        rows = start_boundary[:, 0]

        # Sort by column to ensure monotonic x for np.interp
        sort_idx = np.argsort(cols)
        cols_sorted = cols[sort_idx]
        rows_sorted = rows[sort_idx]

        # Interpolate at every pixel column from 0 to width-1
        w = self.input_layer.data.shape[1]
        downsample_factor = 4.0
        all_cols = np.arange(0, w, downsample_factor, dtype=float)

        interpolated_rows = np.ceil(
            np.interp(all_cols, cols_sorted, rows_sorted) / downsample_factor
        )

        # Result: shape (width, 2) with [row, col] at every pixel
        start_boundary = np.column_stack(
            [interpolated_rows, all_cols // downsample_factor]
        )

        show_info(
            f"Rerunning model from year {selected_year} "
            f"(boundary with {len(start_boundary)} vertices)"
        )

        # Heavy ML imports are deferred to run time: importing torch at module
        # level would drag the whole ML stack into every widget that imports
        # this module (e.g. update_rings_geometries is used by the segmentation
        # widgets), slowing widget opening. Import only when actually running.
        import torch
        from torch.package import PackageImporter

        # Set up rings model
        rings_model = PackageImporter(
            f"{RINGS_MODELS_PATH}/{self._rings_model_weights_file.value}"
        ).load_pickle("LinearRingModel", "model.pkl")
        rings_model.available_device = (
            "cuda"
            if torch.cuda.is_available()
            and self.settings.get("processing.try_to_use_gpu")
            else (
                "mps"
                if torch.mps.is_available()
                and self.settings.get("processing.try_to_use_gpu")
                else "cpu"
            )
        )
        rings_model.to(device=rings_model.available_device)
        # Force synchronization of the internal device attribute
        # We use multiple methods because some models have read-only properties
        try:
            rings_model.device = rings_model.available_device
        except Exception:
            try:
                setattr(rings_model, "device", rings_model.available_device)
            except Exception:
                # Last resort for read-only properties in some model wrappers
                if hasattr(rings_model, "__dict__"):
                    rings_model.__dict__["device"] = rings_model.available_device
        # Use local device variable for autocast checks
        device_obj = torch.device(rings_model.available_device)
        rings_model.use_autocast = bool(
            torch.amp.autocast_mode.is_autocast_available(device_obj.type)
            and self.settings.get("processing.try_to_use_gpu")
            and (rings_model.available_device == "cuda" or rings_model.available_device == "mps")
        )

        # Perform inference
        scan_extension = self.settings.get(
            "file_extensions.scan_file_extension"
        )[0]
        rings_extension = self.settings.get(
            "file_extensions.rings_file_extension"
        )[0]

        image_layer_name = self._input_layer.name.replace(
            rings_extension, scan_extension
        )
        image = (
            self._viewer.layers[image_layer_name].data
            if image_layer_name in self._viewer.layers
            else None
        )
        if image is None:
            show_info(
                f"Corresponding image layer '{image_layer_name}' not found for rings layer '{self._input_layer.name}'"
            )
            return
        import inspect

        # Check if the model's infer method accepts start_boundary
        if "start_boundary" in inspect.signature(rings_model.infer).parameters:
            _, rings_boundaries = rings_model.infer(
                image, start_boundary=start_boundary
            )
        else:
            show_info(
                "Model does not support starting boundary selection; running model on the whole image"
            )
            _, rings_boundaries = rings_model.infer(image)

        # approximate the boundary to reduce number of vertices
        boundary_approx = []
        for _i, boundary in enumerate(rings_boundaries):
            # Convert to numpy or list, whichever is more appropriate
            if isinstance(boundary, torch.Tensor):
                coords = boundary.cpu().numpy().tolist()
            else:
                coords = boundary

            # simplify the new boundaries using cv2.approxPolyDP
            if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                continue

            approx = cv2.approxPolyDP(
                np.array(coords, dtype=np.float32),
                epsilon=settings.get("vectorization.rings_tolerance"),
                closed=False,
            )

            approx = np.squeeze(approx)
            if approx.ndim != 2 or approx.shape[0] < 2:
                continue

            boundary_approx.append(approx)

        layer = self._viewer.layers["Rings Modification"]
        years = layer.features["YEAR"].astype(int)

        # Find indices of shapes with YEAR > selected_year
        indices_to_remove = set(years[years > selected_year].index.tolist())

        # Select and remove them
        layer.selected_data = indices_to_remove
        layer.remove_selected()

        # Set all boundaries to the standard edge color
        layer.edge_color = [
            settings.get("vectorization.rings_edge_color")
        ] * len(layer.data)

        # Add the new boundaries as new shapes
        layer.add(
            boundary_approx,
            shape_type="path",
            edge_color=settings.get(
                "vectorization.rerun_interactive_edge_color"
            ),
            edge_width=settings.get("vectorization.rings_edge_width"),
        )

        # Reassign YEAR values for all shapes based on spatial ordering
        # Keep the same approach as update_rings_geometries: sort by cells_above
        # area, then assign years as (last_year - n + 1) .. last_year
        last_year = self.input_layer.metadata["rings_outmost_complete_year"]
        all_coords = [
            coords.tolist() if hasattr(coords, "tolist") else coords
            for coords in layer.data
        ]
        n_shapes = len(all_coords)

        # Compute cells_above for spatial ordering
        image_width = self.input_layer.data.shape[1]
        areas = [calculate_polygon_area(c, image_width) for c in all_coords]
        order = np.argsort(areas)

        # Assign years: range is (last_year - n_shapes + 1) .. last_year
        # rank 0 = smallest area = topmost boundary = oldest year
        assigned_years = np.zeros(n_shapes, dtype=int)
        for rank, idx in enumerate(order):
            assigned_years[idx] = (last_year - n_shapes) + rank + 1

        # Preserve enabled flags for kept shapes, mark new ones as enabled
        n_kept = n_shapes - len(boundary_approx)
        existing_features = layer.features.copy()
        enabled = []
        for i in range(n_shapes):
            if i < n_kept and "enabled" in existing_features.columns:
                enabled.append(existing_features["enabled"].iloc[i])
            else:
                enabled.append(True)

        layer.features = pd.DataFrame(
            {
                "YEAR": assigned_years.tolist(),
                "enabled": enabled,
            }
        )

        # --- Redraw the "Rings Years" label layer ---
        self._years_update_timer.start(100)

        show_info(
            f"Model rerun complete: {len(boundary_approx)} new boundaries added"
        )
