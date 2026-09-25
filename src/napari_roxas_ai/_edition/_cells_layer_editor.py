from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import cv2
import napari.layers
import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._settings import SettingsManager

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

settings = SettingsManager()


class CellsLayerEditorWidget(Container):
    @property
    def _input_layer(self) -> Optional["napari.layers.Labels"]:
        """Get the single valid cells layer currently in the viewer."""
        valid_layers = self._get_valid_layers()
        return valid_layers[0] if valid_layers else None

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__(labels=False)
        self._viewer = viewer
        self.settings = SettingsManager()

        # Stores visibility of .rings and Rings Year layers before editing to restore them later
        self._layer_visibility_states = {}

        # --- LASSO SELECTION UI ---
        self._lasso_selection_checkbox = CheckBox(
            label="Lasso Select",
            value=False,
            visible=False,
        )
        self._lasso_selection_checkbox.changed.connect(
            lambda val: self._toggle_lasso_selection_mode(val)
        )

        self._delete_lasso_cells_button = PushButton(
            text="Delete Selected Cells",
            visible=False,
        )
        self._delete_lasso_cells_button.changed.connect(
            self._execute_lasso_deletion
        )

        # Horizontal row for lasso selection controls
        self._lasso_container = Container(
            widgets=[
                self._lasso_selection_checkbox,
                self._delete_lasso_cells_button,
            ],
            layout="horizontal",
            labels=False,
            visible=False,
        )
        self._lasso_container.visible = False

        if hasattr(self._lasso_container.native, "layout"):
            layout = self._lasso_container.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(10)

        # Create a button to create the cells working layer
        self._edit_cells_geometries_button = PushButton(
            text="Edit Cells"
        )
        self._edit_cells_geometries_button.changed.connect(
            self._edit_cells_geometries
        )

        # Create an edition mode combo box
        self._edition_mode_combo = ComboBox(
            label="Edition Mode",
            choices=["Edit As Vector", "Edit As Raster"],
            value="Edit As Vector",
        )

        # Create a button to cancel the changes
        self._cancel_cells_geometries_button = PushButton(
            text="Cancel Cell Changes", visible=False
        )
        self._cancel_cells_geometries_button.changed.connect(
            self._cancel_cells_geometries
        )

        # Create a button to apply the changes
        self._apply_cells_geometries_button = PushButton(
            text="Apply Cell Changes", visible=False
        )
        self._apply_cells_geometries_button.changed.connect(
            self._apply_cells_geometries
        )

        # Append the widgets to the container
        self.extend(
            [
                self._lasso_container,
                self._edit_cells_geometries_button,
                self._edition_mode_combo,
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

    def _deferred_remove_layer(self, name: str) -> None:
        """Safely remove a layer by name in the next event loop iteration."""
        def _rm():
            if name in self._viewer.layers:
                try:
                    self._viewer.layers[name].visible = False
                    self._viewer.layers.remove(name)
                except Exception:
                    pass

        QTimer.singleShot(50, _rm)

    def _toggle_lasso_selection_mode(self, enabled: bool) -> None:
        """
        Toggle the lasso selection mode.
        When enabled, a temporary yellow shapes layer is created to allow users
        to draw polygons defining areas for cell deletion.
        """
        self._delete_lasso_cells_button.visible = enabled
        self._delete_lasso_cells_button.enabled = True

        if enabled:
            scale = [1.0, 1.0]
            try:
                if "Cells Modification" in self._viewer.layers:
                    scale = self._viewer.layers["Cells Modification"].scale
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
                    edge_width=5,
                    scale=scale,
                )

            if "Lasso Selection" in self._viewer.layers:
                lasso_layer = self._viewer.layers["Lasso Selection"]
                self._viewer.layers.selection.active = lasso_layer
                lasso_layer.mode = "add_polygon_lasso"

            self._delete_lasso_cells_button.visible = True
            show_info("Lasso Mode: Draw polygons and click 'Delete Selected Cells' (or press 'Delete')")
        else:
            if "Lasso Selection" in self._viewer.layers:
                self._deferred_remove_layer("Lasso Selection")

            if "Cells Modification" in self._viewer.layers:
                try:
                    edit_layer = self._viewer.layers["Cells Modification"]
                    self._viewer.layers.selection.active = edit_layer
                    if isinstance(edit_layer, napari.layers.Shapes):
                        edit_layer.mode = "direct"
                except (ValueError, KeyError, IndexError):
                    pass
            self._delete_lasso_cells_button.visible = False

        # Bind the Delete key to our custom handler
        @self._viewer.bind_key("Delete", overwrite=True)
        def _delete_selected(viewer):
            if not self._cancel_cells_geometries_button.visible:
                return

            if self._lasso_selection_checkbox.value:
                if "Lasso Selection" in self._viewer.layers:
                    self._viewer.layers["Lasso Selection"].selected_data = set()
                self._execute_lasso_deletion()
                return

    def _execute_lasso_deletion(self) -> None:
        """
        Deletes any cells in 'Cells Modification' that have at least one vertex
        inside or on the boundary of any lasso polygon.
        """
        if not self._lasso_selection_checkbox.value:
            return

        self._delete_lasso_cells_button.enabled = False

        if "Lasso Selection" not in self._viewer.layers:
            return

        lasso_layer = self._viewer.layers["Lasso Selection"]
        if len(lasso_layer.data) == 0:
            show_info("No lasso polygon drawn")
            self._lasso_selection_checkbox.value = False
            return

        if "Cells Modification" not in self._viewer.layers:
            return

        edit_layer = self._viewer.layers["Cells Modification"]
        if not isinstance(edit_layer, napari.layers.Shapes):
            return

        try:
            lasso_shapes = [
                shape for shape in lasso_layer.data if len(shape) >= 3
            ]

            if not lasso_shapes:
                show_info("No valid lasso polygons found")
                return

            cell_shapes = edit_layer.data
            if not cell_shapes:
                return

            # Compute bounding box of all lasso polygons (coordinates in (row, col) / (y, x))
            all_lasso_pts = np.concatenate(lasso_shapes, axis=0)
            min_r = int(np.floor(np.min(all_lasso_pts[:, 0]))) - 1
            max_r = int(np.ceil(np.max(all_lasso_pts[:, 0]))) + 1
            min_c = int(np.floor(np.min(all_lasso_pts[:, 1]))) - 1
            max_c = int(np.ceil(np.max(all_lasso_pts[:, 1]))) + 1

            height = max_r - min_r + 1
            width = max_c - min_c + 1

            # Rasterize lasso polygons once into a local bounding box binary mask.
            # cv2.drawContours/fillPoly expects (x, y) = (col, row).
            mask = np.zeros((height, width), dtype=np.uint8)
            lasso_cv_contours = [
                np.round(
                    np.column_stack((poly[:, 1] - min_c, poly[:, 0] - min_r))
                ).astype(np.int32)
                for poly in lasso_shapes
            ]
            cv2.fillPoly(mask, lasso_cv_contours, 1)
            cv2.drawContours(mask, lasso_cv_contours, -1, 1, 1)

            remaining_shapes = []
            deleted_count = 0

            for cell_shape in cell_shapes:
                # cell_shape is an Nx2 array of (row, col) vertices
                v_rows = np.round(cell_shape[:, 0]).astype(int) - min_r
                v_cols = np.round(cell_shape[:, 1]).astype(int) - min_c

                # Check which vertices fall within the mask bounding box
                in_bounds = (
                    (v_rows >= 0)
                    & (v_rows < height)
                    & (v_cols >= 0)
                    & (v_cols < width)
                )

                if np.any(in_bounds):
                    valid_rows = v_rows[in_bounds]
                    valid_cols = v_cols[in_bounds]
                    if np.any(mask[valid_rows, valid_cols] > 0):
                        deleted_count += 1
                        continue

                remaining_shapes.append(cell_shape)

            edit_layer.data = remaining_shapes
            if deleted_count > 0:
                show_info(f"Deleted {deleted_count} cell(s)")
            else:
                show_info("No cells touched or included in the lasso area")

            # Clear the lasso polygons
            lasso_layer.data = []
        except Exception as e:
            show_info(f"Error during cell deletion: {str(e)}")
        finally:
            self._lasso_selection_checkbox.value = False
            if self._edition_mode == "Edit As Vector":
                self._lasso_container.visible = True

    def _restore_original_visibility(self) -> None:
        """
        Restore the original visibility state of layers that were hidden during editing.
        """
        try:
            for layer_name, visible in self._layer_visibility_states.items():
                if layer_name in self._viewer.layers:
                    self._viewer.layers[layer_name].visible = visible
            self._layer_visibility_states = {}
        except Exception:
            pass

    def _edit_cells_geometries(self) -> None:
        """Run the segmentation analysis in a separate thread."""
        # Get the selected input layer
        input_layer = self._input_layer
        if not input_layer:
            QMessageBox.warning(None, "Error", "No cells layer found in the viewer")
            return

        # --- LAYER VISIBILITY MANAGEMENT ---
        # Hide .rings and Rings Year layers to reduce clutter during cell editing.
        # Store their original visibility state to restore it when editing is finished.
        self._layer_visibility_states = {}
        for layer in self._viewer.layers:
            if layer.name.endswith(".rings") or layer.name.startswith("Rings Year"):
                self._layer_visibility_states[layer.name] = layer.visible
                layer.visible = False

        self._edition_mode = self._edition_mode_combo.value

        # Update button visibility
        self._edition_mode_combo.visible = False
        self._edit_cells_geometries_button.visible = False
        self._cancel_cells_geometries_button.visible = True
        self._apply_cells_geometries_button.visible = True

        if self._edition_mode == "Edit As Vector":
            self._lasso_container.visible = True
            self._lasso_selection_checkbox.visible = True
            self._lasso_selection_checkbox.value = False
            self._delete_lasso_cells_button.visible = False
        else:
            self._lasso_container.visible = False

        self.input_layer = input_layer

        sample_name = self.input_layer.metadata.get("sample_name")
        sample_stem_path = self.input_layer.metadata.get("sample_stem_path")

        if self._edition_mode == "Edit As Raster":
            colormap = defaultdict(lambda: [0, 0, 0, 0])
            colormap[1] = self.settings.get("vectorization.cells_face_color")

            # Create working layer with a copy - original layer stays untouched
            work_layer = self._viewer.add_labels(
                self.input_layer.data.copy(),
                name="Cells Modification",
                scale=self.input_layer.scale,
                colormap=colormap,
                metadata={
                    "sample_name": sample_name,
                    "sample_stem_path": sample_stem_path,
                },
            )

        elif self._edition_mode == "Edit As Vector":
            contours, _ = cv2.findContours(
                self.input_layer.data.astype("uint8"),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            tolerance = float(settings.get("vectorization.cells_tolerance") or 0)
            cells_polygons = []
            for contour in contours:
                if len(contour) < 3:
                    continue
                if tolerance > 0:
                    poly = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
                else:
                    poly = contour
                if poly.shape[0] > 2:
                    cells_polygons.append(poly.squeeze(axis=1)[:, ::-1])

            work_layer = self._viewer.add_shapes(
                cells_polygons,
                shape_type="polygon",
                face_color=settings.get("vectorization.cells_face_color"),
                edge_color=settings.get("vectorization.cells_edge_color"),
                edge_width=settings.get("vectorization.cells_edge_width"),
                opacity=1,
                name="Cells Modification",
                scale=self.input_layer.scale,
                metadata={
                    "sample_name": sample_name,
                    "sample_stem_path": sample_stem_path,
                },
            )

        else:
            QMessageBox.warning(None, "Error", "Unknown edition mode selected")
            return

        # Attach required metadata so other widgets don't crash when iterating layers
        if sample_name is not None:
            work_layer.metadata["sample_name"] = sample_name
        if sample_stem_path is not None:
            work_layer.metadata["sample_stem_path"] = sample_stem_path

    def _cancel_cells_geometries(self) -> None:
        """Cancel the changes made to the input layer."""
        if self._lasso_selection_checkbox.value:
            self._lasso_selection_checkbox.value = False
        if "Lasso Selection" in self._viewer.layers:
            self._viewer.layers.remove("Lasso Selection")

        self._lasso_container.visible = False
        self._delete_lasso_cells_button.visible = False

        # Remove the working layer - original layer was never modified
        if "Cells Modification" in self._viewer.layers:
            self._viewer.layers.remove("Cells Modification")

        # Reset the button visibility
        self._edition_mode_combo.visible = True
        self._edit_cells_geometries_button.visible = True
        self._cancel_cells_geometries_button.visible = False
        self._apply_cells_geometries_button.visible = False

        self._restore_original_visibility()

        # Show confirmation message
        show_info("Cells geometries modification cancelled")

    def _apply_cells_geometries(self) -> None:
        """Apply the changes to the input layer."""
        if self._lasso_selection_checkbox.value:
            self._lasso_selection_checkbox.value = False
        if "Lasso Selection" in self._viewer.layers:
            self._viewer.layers.remove("Lasso Selection")

        self._lasso_container.visible = False
        self._delete_lasso_cells_button.visible = False

        if getattr(self, "_edition_mode", None) == "Edit As Raster":
            new_cells_raster = self._viewer.layers["Cells Modification"].data
            self._viewer.layers.remove("Cells Modification")

        elif getattr(self, "_edition_mode", None) == "Edit As Vector":
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

        self._restore_original_visibility()

        # Show confirmation message
        show_info("Cells geometries successfully updated")
