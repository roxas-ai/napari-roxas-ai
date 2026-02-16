from typing import TYPE_CHECKING

import cv2
import napari.layers
import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_info
from PIL import Image
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

    # Check if any point of the ring is outside the mask
    for i, coords in rings_table["RBXY"].items():
        coords = np.array(coords)
        is_any_valid = np.any(
            (coords[:, 0] > 0)
            & (coords[:, 0] <= shape[0])
            & (coords[:, 1] > 0)
            & (coords[:, 1] < shape[1])
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
        canvas_width (int): Width of the canvas.
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
    rings_table["RBXY"] = rings_table[
        "RBXY"
    ].apply(rearrange_coordinates)

    # Ensure rings are inside the image
    rings_table = outside_rings_deletion(rings_table, image_shape)

    # Ensure complete rings
    rings_table["RBXY"] = rings_table[
        "RBXY"
    ].apply(lambda x: horizontal_rings_completion(x, image_shape[1]))

    # Clip rings to the image width
    rings_table["RBXY"] = rings_table[
        "RBXY"
    ].apply(lambda x: horizontal_rings_clippping(x, image_shape[1]))

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
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.settings = SettingsManager()

        self._layer_callback = None

        # Create a layer selection widget filtered by scan extension
        self._input_layer_combo = ComboBox(
            label="Rings Layer",
            annotation="napari.layers.Labels",
            choices=self._get_valid_layers,
        )
        # Connect the layer selection to update the year spinbox
        self._input_layer_combo.changed.connect(self._connect_layer_callback)

        # Create spinbox for the last year
        year_value = (
            self._input_layer_combo.value.metadata[
                "rings_outmost_complete_year"
            ]
            if self._input_layer_combo.value
            else 9999
        )
        self._last_year_spinbox = SpinBox(
            value=year_value,
            label="Last Complete Ring Year",
            min=-999999,
            max=9999,
            step=1,
        )

        self._refresh_button = PushButton(text="Refresh widget", visible=True)
        self._refresh_button.changed.connect(self._refresh_widget)

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

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._edit_rings_geometries_button,
                self._cancel_rings_geometries_button,
                self._apply_rings_geometries_button,
                self._last_year_spinbox,
                self._last_year_update_button,
                self._refresh_button,
            ]
        )

        # Update choices when layers change

    def _deferred_remove_layer(self, name: str) -> None:
        def _rm():
            if name in self._viewer.layers:
                self._viewer.layers.remove(name)

        QTimer.singleShot(0, _rm)

    def _refresh_widget(self) -> None:
        """Refresh widget state by re-reading layers/metadata and re-binding callbacks."""
        # Remove helper layers if present
        for name in ("Rings Years", "Rings Modification"):
            self._deferred_remove_layer(name)

        current = self._input_layer_combo.value

        new_choices = self._get_valid_layers()
        self._input_layer_combo.choices = new_choices

        if not new_choices:
            self._disconnect_layer_callback()
            # Optional: reset UI to a safe default
            self._last_year_spinbox.value = 9999
            show_info("No rings layers found to refresh.")
            return

        # Keep current selection if still valid, otherwise fall back to first choice
        if current in new_choices:
            self._input_layer_combo.value = current
        else:
            self._input_layer_combo.value = new_choices[0]

        self._connect_layer_callback()
        self._update_year_spinbox()

        show_info("Refreshed rings editor")

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

        if self._input_layer_combo.value is not None:
            # Connect to the layer's events using the shared callback manager
            self._layer_callback = register_layer_callback(
                self._input_layer_combo.value, self, self._on_layer_data_change
            )
            self._update_year_spinbox()

    def _disconnect_layer_callback(self):
        """Disconnect callback from the previously selected layer."""
        if (
            self._input_layer_combo.value is not None
            and self._layer_callback is not None
        ):
            unregister_layer_callback(self._input_layer_combo.value, self)
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

    def _update_year_spinbox(self) -> None:
        """Update the year spinbox value based on the selected layer."""
        if self._input_layer_combo.value:
            layer = self._input_layer_combo.value
            if "rings_outmost_complete_year" in layer.metadata:
                self._last_year_spinbox.value = layer.metadata[
                    "rings_outmost_complete_year"
                ]

    def _update_layer_year(self) -> None:
        """Update the last year value in the layer metadata."""
        if self._input_layer_combo.value:

            self.input_layer = self._input_layer_combo.value
            self.input_layer.metadata["rings_outmost_complete_year"] = (
                self._last_year_spinbox.value
            )
            new_rings_table, new_rings_raster, new_colormap = (
                update_rings_geometries(
                    rings_table=self.input_layer.features,
                    last_year=self._last_year_spinbox.value,
                    image_shape=self.input_layer.data.shape,
                )
            )
            self.input_layer.data = new_rings_raster
            self.input_layer.features = new_rings_table
            self.input_layer.colormap = new_colormap

    def _edit_rings_geometries(self) -> None:
        """Run the segmentation analysis in a separate thread."""
        # Get the selected input layer
        if not self._input_layer_combo.value:
            QMessageBox.warning(None, "Error", "Please select an input layer")
            return

        # Update button visibility
        self._edit_rings_geometries_button.visible = False
        self._last_year_spinbox.visible = False
        self._last_year_update_button.visible = False
        self._cancel_rings_geometries_button.visible = True
        self._apply_rings_geometries_button.visible = True

        self.input_layer = self._input_layer_combo.value

        # If there is already an edit session open, remove old helper layers first
        if "Rings Years" in self._viewer.layers:
            self._deferred_remove_layer("Rings Years")
        if "Rings Modification" in self._viewer.layers:
            self._deferred_remove_layer("Rings Modification")

        # Build a DF in the same logical order as the annotated export
        df = self.input_layer.features.copy()

        # Prefer YEAR ordering; otherwise fall back to cells_above if present
        if "YEAR" in df.columns:
            df = df.sort_values("YEAR").reset_index(drop=True)
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
                "enabled": df["enabled"].fillna(True).tolist() if "enabled" in df.columns else [True] * len(df),
            },
        )

        # left margin in data coords
        sx = float(self.input_layer.scale[1]) if hasattr(self.input_layer, "scale") else 1.0
        x_left = 10.0 / sx

        # y-value of each boundary line at x_left
        y_on_left = [interpolate_row_at_col(coords, x_left) for coords in df["RBXY"].tolist()]

        # center of ring i is between boundary i and boundary i+1 at that same x
        # for the last ring, use the image height as the lower boundary
        image_height = self.input_layer.data.shape[0]
        centers_r = []
        for i in range(len(y_on_left)):
            if i + 1 < len(y_on_left):
                centers_r.append(0.5 * (y_on_left[i] + y_on_left[i + 1]))
            else:
                centers_r.append(0.5 * (y_on_left[i] + image_height))

        # left margin in *data coords*
        sx = float(self.input_layer.scale[1]) if hasattr(self.input_layer, "scale") else 1.0
        x_left = 10.0 / sx

        years = [str(int(y)) for y in df["YEAR"].tolist()]

        points_rc = np.column_stack([
            np.array(centers_r, dtype=float),
            np.full(len(df), x_left, dtype=float),
        ])

        self._viewer.add_points(
            points_rc,
            name="Rings Years",
            size=1,
            opacity=1.0,
            edge_width=0,
            face_color=[0, 0, 0, 0],
            edge_color=[0, 0, 0, 0],
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

    def _cancel_rings_geometries(self) -> None:
        """Cancel the changes made to the input layer."""
        # Remove the working layer
        self._deferred_remove_layer("Rings Modification")
        if "Rings Years" in self._viewer.layers:
            self._deferred_remove_layer("Rings Years")

        # Reset the button visibility
        self._edit_rings_geometries_button.visible = True
        self._last_year_spinbox.visible = True
        self._last_year_update_button.visible = True
        self._cancel_rings_geometries_button.visible = False
        self._apply_rings_geometries_button.visible = False

        # Show confirmation message
        show_info("Rings geometries modification cancelled")

    def _apply_rings_geometries(self) -> None:
        """Apply the changes to the input layer."""

        layer = self._viewer.layers["Rings Modification"]

        rings_table = pd.DataFrame(
            {
                "RBXY": [coords.tolist() for coords in layer.data],
                "YEAR": layer.features["YEAR"].astype(int).tolist() if "YEAR" in layer.features else None,
                "enabled": layer.features["enabled"].tolist() if "enabled" in layer.features else None,
            }
        ).rename_axis("id")
        rings_table = rings_table.dropna(axis=1, how="all")

        # Remove helper layers
        if "Rings Modification" in self._viewer.layers:
            self._deferred_remove_layer("Rings Modification")
        if "Rings Years" in self._viewer.layers:
            self._deferred_remove_layer("Rings Years")

        # Update the rings layer with the new geometries
        new_rings_table, new_rings_raster, new_colormap = (
            update_rings_geometries(
                rings_table=rings_table,
                last_year=self.input_layer.metadata[
                    "rings_outmost_complete_year"
                ],
                image_shape=self.input_layer.data.shape,
            )
        )

        self.input_layer.data = new_rings_raster
        self.input_layer.features = new_rings_table
        self.input_layer.colormap = new_colormap

        # Reset the button visibility
        self._edit_rings_geometries_button.visible = True
        self._last_year_spinbox.visible = True
        self._last_year_update_button.visible = True
        self._cancel_rings_geometries_button.visible = False
        self._apply_rings_geometries_button.visible = False

        # Show confirmation message
        show_info("Ring geometries successfully updated")
