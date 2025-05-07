from collections import defaultdict
from typing import TYPE_CHECKING

import cv2
import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._settings import SettingsManager

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

settings = SettingsManager()


class CellsLayerEditorWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.settings = SettingsManager()

        # Create a layer selection widget filtered by scan extension
        self._input_layer_combo = ComboBox(
            label="Cells Layer",
            annotation="napari.layers.Labels",
            choices=self._get_valid_layers,
        )

        # Create an edition mode combo box
        self._edition_mode_combo = ComboBox(
            label="Edition Mode",
            choices=["Edit As Raster", "Edit As Vector"],
            value="Edit As Raster",
        )

        # Create a button to create the cells working layer
        self._edit_cells_geometries_button = PushButton(
            text="Edit Cells Geometries"
        )
        self._edit_cells_geometries_button.changed.connect(
            self._edit_cells_geometries
        )

        # Create a button to cancel the changes
        self._cancel_cells_geometries_button = PushButton(
            text="Cancel Geometries Changes", visible=False
        )
        self._cancel_cells_geometries_button.changed.connect(
            self._cancel_cells_geometries
        )

        # Create a button to apply the changes
        self._apply_cells_geometries_button = PushButton(
            text="Apply Geometries Changes", visible=False
        )
        self._apply_cells_geometries_button.changed.connect(
            self._apply_cells_geometries
        )

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._edition_mode_combo,
                self._edit_cells_geometries_button,
                self._cancel_cells_geometries_button,
                self._apply_cells_geometries_button,
            ]
        )

    def _get_valid_layers(self, widget=None):
        """Get layers that are both Labels type and match the cells file extension."""
        cells_extension = settings.get("file_extensions.cells_file_extension")[
            0
        ]
        valid_layers = []

        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Labels) and layer.name.endswith(
                cells_extension
            ):
                valid_layers.append(layer)

        return valid_layers

    def _edit_cells_geometries(self) -> None:
        """Run the segmentation analysis in a separate thread."""
        # Get the selected input layer
        if not self._input_layer_combo.value:
            QMessageBox.warning(None, "Error", "Please select an input layer")
            return

        # Update button visibility
        self._edition_mode_combo.visible = False
        self._edit_cells_geometries_button.visible = False
        self._cancel_cells_geometries_button.visible = True
        self._apply_cells_geometries_button.visible = True

        self.input_layer = self._input_layer_combo.value

        if self._edition_mode_combo.value == "Edit As Raster":
            # Create a new working layer with the same data as the input layer

            colormap = defaultdict(lambda: [0, 0, 0, 0])
            colormap[1] = self.settings.get("vectorization.cells_face_color")
            self._viewer.add_labels(
                self.input_layer.data,
                name="Cells Modification",
                scale=self.input_layer.scale,
                colormap=colormap,
            )

        elif self._edition_mode_combo.value == "Edit As Vector":
            # Simplify boundary coordinates using cv2.approxPolyDP
            cells_contours = [
                np.maximum(contour - 1, 0)
                for contour in cv2.findContours(
                    self.input_layer.data.astype("uint8"),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )[0]
            ]
            cells_polygons = [
                cv2.approxPolyDP(
                    contour,
                    epsilon=settings.get("vectorization.cells_tolerance"),
                    closed=True,
                )
                for contour in cells_contours
            ]
            cells_polygons = [
                polygon.squeeze(axis=1)[:, ::-1]
                for polygon in cells_polygons
                if polygon.shape[0] > 2
            ]

            # Create a new Shapes layer with the simplified boundary lines
            self._viewer.add_shapes(
                cells_polygons,
                shape_type="polygon",
                face_color=settings.get("vectorization.cells_face_color"),
                edge_color=settings.get("vectorization.cells_edge_color"),
                edge_width=settings.get("vectorization.cells_edge_width"),
                opacity=1,
                name="Cells Modification",
                scale=self.input_layer.scale,
            )

    def _cancel_cells_geometries(self) -> None:
        """Cancel the changes made to the input layer."""
        # Remove the working layer
        self._viewer.layers.remove("Cells Modification")

        # Reset the button visibility
        self._edition_mode_combo.visible = True
        self._edit_cells_geometries_button.visible = True
        self._cancel_cells_geometries_button.visible = False
        self._apply_cells_geometries_button.visible = False

        # Show confirmation message
        show_info("Cells geometries modification cancelled")

    def _apply_cells_geometries(self) -> None:
        """Apply the changes to the input layer."""

        if self._edition_mode_combo.value == "Edit As Raster":
            new_cells_raster = self._viewer.layers["Cells Modification"].data
            self._viewer.layers.remove("Cells Modification")

        elif self._edition_mode_combo.value == "Edit As Vector":
            # Recover new shapes data from the viewer
            new_cells_shapes = self._viewer.layers["Cells Modification"].data
            new_cells_shapes = [
                shape[:, ::-1].round().astype("int")
                for shape in new_cells_shapes
            ]
            self._viewer.layers.remove("Cells Modification")

            # Rasterize the new shapes
            new_cells_raster = np.zeros_like(self.input_layer.data).astype(
                "uint8"
            )
            cv2.drawContours(new_cells_raster, new_cells_shapes, -1, 1, -1)

        # Update the cells layer with the new geometries
        self.input_layer.data = new_cells_raster
        self.input_layer.features = pd.DataFrame()

        # Reset the button visibility
        self._edition_mode_combo.visible = True
        self._edit_cells_geometries_button.visible = True
        self._cancel_cells_geometries_button.visible = False
        self._apply_cells_geometries_button.visible = False

        # Show confirmation message
        show_info("Ring geometries successfully updated")
