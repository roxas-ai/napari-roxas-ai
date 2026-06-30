from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_info
from PIL import Image
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

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

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
    Args:
        coords (list): List of coordinates.
        width (int): Width of the image.
    Returns:
        list: Clipped coordinates.
    """
    coords = np.array(coords)

    left_points = np.where(coords[:, 1] <= 0)[0]
    right_points = np.where(coords[:, 1] >= width)[0]

    coords = coords[left_points[-1] : right_points[0] + 1]

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

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.settings = SettingsManager()

        self._layer_callback = None
        self._is_editing = False

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
            text="Edit Rings Geometries"
        )
        self._edit_rings_geometries_button.changed.connect(
            self._edit_rings_geometries
        )

        # Create a button to cancel the changes
        self._cancel_rings_geometries_button = PushButton(
            text="Cancel Geometries Changes", visible=False
        )
        self._cancel_rings_geometries_button.changed.connect(
            self._cancel_rings_geometries
        )

        # Create a button to apply the changes
        self._apply_rings_geometries_button = PushButton(
            text="Apply Geometries Changes", visible=False
        )
        self._apply_rings_geometries_button.changed.connect(
            self._apply_rings_geometries
        )

        # Create a horizontal container for the "Rerun Model" button and year selector
        self._rerun_model_year_spinbox = SpinBox(
            value=9999,
            label="Year",
            min=-10000,
            max=10000,
            step=1,
        )

        self._rings_model_weights_file = ComboBox(
            choices=tuple(
                path.name for path in Path(RINGS_MODELS_PATH).iterdir()
            ),
            label="Rings Model",
        )
        self._rerun_model_button = PushButton(
            text="Run",
        )

        self._rerun_model_button.changed.connect(self._rerun_model_from_year)
        self._rerun_model_container = Container(
            widgets=[
                # self.label_rerun_model,
                self._rerun_model_year_spinbox,
                self._rings_model_weights_file,
                self._rerun_model_button,
            ],
            layout="Vertical",
            visible=False,
            labels=True,
            label="Rerun Model from year:",
        )

        # Append the widgets to the container
        self.extend(
            [
                self._edit_rings_geometries_button,
                self._cancel_rings_geometries_button,
                self._apply_rings_geometries_button,
                self._rerun_model_container,
                self._last_year_spinbox,
                self._last_year_update_button,
            ]
        )

        self._viewer.layers.events.inserted.connect(self._on_layer_change)
        self._viewer.layers.events.removed.connect(self._on_layer_change)

        self._connect_layer_callback()

    def _on_layer_change(self, event=None):
        """Called when layers are added/removed in the viewer."""
        if self._is_editing:
            return

        if not self._get_valid_layers():
            self._disconnect_layer_callback()
            self._last_year_spinbox.value = 9999
            return

        self._connect_layer_callback()

    def _deferred_remove_layer(self, name: str) -> None:
        def _rm():
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        QTimer.singleShot(0, _rm)

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
        if not self._input_layer:
            QMessageBox.warning(None, "Error", "No valid rings layer found")
            return

        self._is_editing = True

        # Update button visibility
        self._edit_rings_geometries_button.visible = False
        self._last_year_spinbox.visible = False
        self._last_year_update_button.visible = False
        self._cancel_rings_geometries_button.visible = True
        self._apply_rings_geometries_button.visible = True
        self._rerun_model_container.visible = True

        input_layer = self._input_layer
        self.input_layer = input_layer

        # If there is already an edit session open, remove old helper layers first
        if "Rings Years" in self._viewer.layers:
            self._viewer.layers.remove("Rings Years")
        if "Rings Modification" in self._viewer.layers:
            self._viewer.layers.remove("Rings Modification")

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

        # Simplify boundary coordinates using cv2.approxPolyDP
        simplified_boundary_lines = []
        keep_rows = []
        for i, coords in enumerate(df["RBXY"].tolist()):
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

            simplified_boundary_lines.append(approx.tolist())
            keep_rows.append(i)

        # Ensure features rows match the number of shapes
        df = df.iloc[keep_rows].reset_index(drop=True)
        if df.empty:
            show_info("No valid rings to edit")
            return

        # Create the editable Shapes layer
        self._viewer.add_shapes(
            simplified_boundary_lines,
            shape_type="path",
            edge_color=settings.get("vectorization.rings_edge_color"),
            edge_width=settings.get("vectorization.rings_edge_width"),
            opacity=1,
            name="Rings Modification",
            scale=self.input_layer.scale,
            features={
                "YEAR": df["YEAR"].tolist(),
                "enabled": (
                    df["enabled"].fillna(True).tolist()
                    if "enabled" in df.columns
                    else [True] * len(df)
                ),
            },
        )

        # left margin in data coords
        sx = (
            float(self.input_layer.scale[1])
            if hasattr(self.input_layer, "scale")
            else 1.0
        )
        x_left = 10.0 / sx

        # y-value of each boundary line at x_left
        y_on_left = [
            interpolate_row_at_col(coords, x_left)
            for coords in df["RBXY"].tolist()
        ]

        # Each boundary's YEAR labels the ring *above* it (between the previous
        # boundary and this one).  So the label for boundary[i] should sit
        # between boundary[i-1] and boundary[i].  For boundary[0] the upper
        # edge is the top of the image (row 0).
        centers_r = []
        for i in range(len(y_on_left)):
            upper = y_on_left[i - 1] if i > 0 else 0.0
            centers_r.append(0.5 * (upper + y_on_left[i]))

        years = [str(int(y)) for y in df["YEAR"].tolist()]

        # left margin in *data coords*
        sx = (
            float(self.input_layer.scale[1])
            if hasattr(self.input_layer, "scale")
            else 1.0
        )
        x_left = 10.0 / sx

        points_rc = np.column_stack(
            [
                np.array(centers_r, dtype=float),
                np.full(len(df), x_left, dtype=float),
            ]
        )

        self._viewer.add_points(
            points_rc,
            name="Rings Years",
            size=1,
            opacity=1.0,
            border_width=0,
            face_color=[0, 0, 0, 0],
            border_color=[0, 0, 0, 0],
            scale=self.input_layer.scale,
            features={"YEAR": years},
            text={
                "string": "{YEAR}",
                "anchor": "upper_left",
                "translation": [0, 0],
                "size": 8,
                "color": "black",
                "blending": "translucent",
            },
        )

        years_layer = self._viewer.layers["Rings Years"]
        shapes_layer = self._viewer.layers["Rings Modification"]

        layers = self._viewer.layers
        years_index = layers.index(years_layer)
        shapes_index = layers.index(shapes_layer)

        # We want years directly below shapes => years should end up at index == shapes_index - 1.
        dest_index = shapes_index
        if years_index < shapes_index:
            dest_index -= 1

        layers.move(years_index, dest_index)

        layers.selection.active = shapes_layer
        years_layer.editable = False

        self._update_year_spinbox()

    def _cancel_rings_geometries(self) -> None:
        """Cancel the changes made to the input layer."""
        self._is_editing = False

        if "Rings Modification" in self._viewer.layers:
            self._viewer.layers.remove("Rings Modification")
        if "Rings Years" in self._viewer.layers:
            self._viewer.layers.remove("Rings Years")

        # Reset the button visibility
        self._edit_rings_geometries_button.visible = True
        self._last_year_spinbox.visible = True
        self._last_year_update_button.visible = True
        self._cancel_rings_geometries_button.visible = False
        self._apply_rings_geometries_button.visible = False
        self._rerun_model_container.visible = False

        show_info("Rings geometries modification cancelled")

    def _apply_rings_geometries(self) -> None:
        """Apply the changes to the input layer."""
        self._is_editing = False

        layer = self._viewer.layers["Rings Modification"]

        rings_table = pd.DataFrame(
            {
                "RBXY": [coords.tolist() for coords in layer.data],
                "YEAR": (
                    layer.features["YEAR"].astype(int).tolist()
                    if "YEAR" in layer.features
                    else None
                ),
                "enabled": (
                    layer.features["enabled"].tolist()
                    if "enabled" in layer.features
                    else None
                ),
            }
        ).rename_axis("id")
        rings_table = rings_table.dropna(axis=1, how="all")

        if "Rings Modification" in self._viewer.layers:
            self._viewer.layers.remove("Rings Modification")
        if "Rings Years" in self._viewer.layers:
            self._viewer.layers.remove("Rings Years")

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

        input_layer.data = new_rings_raster
        input_layer.features = new_rings_table
        input_layer.colormap = new_colormap

        # Reset the button visibility
        self._edit_rings_geometries_button.visible = True
        self._last_year_spinbox.visible = True
        self._last_year_update_button.visible = True
        self._cancel_rings_geometries_button.visible = False
        self._apply_rings_geometries_button.visible = False
        self._rerun_model_container.visible = False

        # Show confirmation message
        show_info("Ring geometries successfully updated")

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

    def _rerun_model_from_year(self) -> None:
        """Rerun the ring detection model starting from the selected year."""
        selected_year = self._rerun_model_year_spinbox.value

        if not self._input_layer:
            show_info("No layer selected")
            return

        # Read from the Shapes editing layer, not the original Labels layer
        if "Rings Modification" not in self._viewer.layers:
            show_info(
                "No editing session active — click 'Edit Rings Geometries' first"
            )
            return

        edit_layer = self._viewer.layers["Rings Modification"]

        # Build rings_table from the current Shapes layer data
        rings_table = pd.DataFrame(
            {
                "RBXY": [
                    coords.tolist() if hasattr(coords, "tolist") else coords
                    for coords in edit_layer.data
                ],
                "YEAR": (
                    edit_layer.features["YEAR"].astype(int).tolist()
                    if "YEAR" in edit_layer.features
                    else None
                ),
                "enabled": (
                    edit_layer.features["enabled"].tolist()
                    if "enabled" in edit_layer.features
                    else None
                ),
            }
        ).rename_axis("id")
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
        # Fix for problem with model object; device attribute is not updated with to()
        rings_model.device = rings_model.available_device
        rings_model.use_autocast = bool(
            torch.amp.autocast_mode.is_autocast_available(rings_model.device)
            and self.settings.get("processing.try_to_use_gpu")
            and (rings_model.device == "cuda" or rings_model.device == "mps")
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
        if "Rings Years" in self._viewer.layers:
            self._viewer.layers.remove("Rings Years")

        # Reorder coords by YEAR for label placement
        df_labels = (
            pd.DataFrame(
                {
                    "YEAR": assigned_years.tolist(),
                    "RBXY": all_coords,
                }
            )
            .sort_values("YEAR")
            .reset_index(drop=True)
        )

        sx = (
            float(self.input_layer.scale[1])
            if hasattr(self.input_layer, "scale")
            else 1.0
        )
        x_left = 10.0 / sx

        y_on_left = [
            interpolate_row_at_col(c, x_left)
            for c in df_labels["RBXY"].tolist()
        ]
        centers_r = []
        for i in range(len(y_on_left)):
            upper = y_on_left[i - 1] if i > 0 else 0.0
            centers_r.append(0.5 * (upper + y_on_left[i]))

        year_strings = [str(int(y)) for y in df_labels["YEAR"].tolist()]
        points_rc = np.column_stack(
            [
                np.array(centers_r, dtype=float),
                np.full(len(df_labels), x_left, dtype=float),
            ]
        )

        self._viewer.add_points(
            points_rc,
            name="Rings Years",
            size=1,
            opacity=1.0,
            border_width=0,
            face_color=[0, 0, 0, 0],
            border_color=[0, 0, 0, 0],
            scale=self.input_layer.scale,
            features={"YEAR": year_strings},
            text={
                "string": "{YEAR}",
                "anchor": "upper_left",
                "translation": [0, 0],
                "size": 8,
                "color": "black",
                "blending": "translucent",
            },
        )

        # Position years layer just below shapes layer
        years_layer = self._viewer.layers["Rings Years"]
        shapes_layer = self._viewer.layers["Rings Modification"]
        layers = self._viewer.layers
        years_index = layers.index(years_layer)
        shapes_index = layers.index(shapes_layer)
        dest_index = shapes_index
        if years_index < shapes_index:
            dest_index -= 1
        layers.move(years_index, dest_index)
        layers.selection.active = shapes_layer
        years_layer.editable = False

        show_info(
            f"Model rerun complete: {len(boundary_approx)} new boundaries added"
        )
