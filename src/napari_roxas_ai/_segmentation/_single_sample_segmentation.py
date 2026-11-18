from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import napari.layers
from scipy import ndimage as ndi
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
)
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._assets_files import check_assets_and_download
from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._job_runner import JobButton, JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag
from napari_roxas_ai._reader import get_metadata_from_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._utils import make_binary_labels_colormap
from napari_roxas_ai._writer._writer import save_layers_annotated_image
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present

from .._utils._fix_sample_stem_path import fix_sample_stem_paths_in_project
from .._utils._metadata_keys import SAMPLE_METADATA_PREFIXES
from ._operations import segment_sample

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

MODULE_PATH = Path(__file__).parent.absolute()
CELLS_MODELS_PATH = MODULE_PATH / "_models" / "_cells"
RINGS_MODELS_PATH = MODULE_PATH / "_models" / "_rings"
DEFAULT_RINGS_MODEL = "linear_rings_model.pth"

# Gap between a row's label/checkbox and its combo box
LABEL_SPACING = 10

def apply_segmentation_results_to_viewer(
    viewer,
    *,
    results: Dict[str, Any],
    settings,
    input_scale,
    sample_metadata: Dict[str, Any],
    sample_stem_path: Optional[str] = None,
    cells_model_file: Optional[str] = None,
    rings_model_file: Optional[str] = None,
) -> None:
    """Apply segmentation outputs to the napari viewer.

    Fix: ensure layers have sample metadata at *insert time* (viewer.add_labels),
    so any UI callbacks triggered by layer insertion never see missing keys.
    """

    sample_metadata = sample_metadata or {}

    # --- CELLS ---
    if "cells" in results:
        cells = results["cells"]
        cells_name = cells["name"]

        cells_metadata = {
            "cells_segmentation_model": cells_model_file,
            "cells_segmentation_datetime": datetime.now().isoformat(),
        }

        cells_md = {}
        cells_md.update(sample_metadata)
        cells_md.update(cells_metadata)

        try:
            cells_layer = viewer.layers[cells_name]
            cells_layer.data = cells["data"]
            cells_layer.colormap = make_binary_labels_colormap()
            # Update metadata for existing layer as well
            cells_layer.metadata.update(cells_md)
        except KeyError:
            # Important: pass metadata at creation time
            cells_layer = viewer.add_labels(
                cells["data"],
                name=cells_name,
                colormap=make_binary_labels_colormap(),
                metadata=cells_md,
            )

        cells_layer.scale = input_scale
        # New segmentation, not yet written to disk
        cells_layer.metadata["_is_dirty"] = True

    # --- RINGS ---
    if "rings" in results:
        rings = results["rings"]
        rings_name = rings["name"]

        metadata_file_contents = None
        if sample_stem_path:
            p = Path(sample_stem_path)
            if not p.is_absolute():
                proj = settings.get("project_directory")
                if isinstance(proj, str) and proj:
                    p = Path(proj) / p
            metadata_file_contents = get_metadata_from_file(
                path=str(p),
                path_is_stem=True,
            )
        default_rings_year_value = [
            field["default"]
            for field in settings.get("samples_metadata.fields")
            if field["id"] == "rings_outmost_complete_year"
        ][0]

        rings_metadata = {
            "rings_outmost_complete_year": (
                metadata_file_contents.get("rings_outmost_complete_year")
                if metadata_file_contents
                else default_rings_year_value
            ),
            "rings_segmentation_model": rings_model_file,
            "rings_segmentation_datetime": datetime.now().isoformat(),
        }

        rings_md = {}
        rings_md.update(sample_metadata)
        rings_md.update(rings_metadata)

        try:
            rings_layer = viewer.layers[rings_name]
            rings_layer.data = rings["data"]
            rings_layer.metadata.update(rings_md)
        except KeyError:
            # pass metadata at creation time
            rings_layer = viewer.add_labels(
                rings["data"],
                name=rings_name,
                metadata=rings_md,
            )

        rings_layer.scale = input_scale
        # New segmentation, not yet written to disk
        rings_layer.metadata["_is_dirty"] = True

        # Apply / merge features
        new_df = rings.get("features", None)

        with rings_layer.events.blocker():
            if new_df is not None and not new_df.empty:
                df = new_df.copy()

                # Normalize YEAR
                if "YEAR" not in df.columns:
                    if "ring_year" in df.columns:
                        df = df.rename(columns={"ring_year": "YEAR"})
                    else:
                        # Pre-fill YEAR based on last_year and number of boundaries
                        n = len(df)
                        last_year = int(rings_layer.metadata["rings_outmost_complete_year"])
                        df["YEAR"] = list(range(last_year - n + 1, last_year + 1))

                # Ensure enabled exists
                if "enabled" not in df.columns:
                    df["enabled"] = True
                    if len(df) > 0:
                        df.loc[df.index[0], "enabled"] = False

                # Replace existing features: the new boundaries (with the dummy
                # top boundary) don't map row by row onto the old table, and
                # YEAR / enabled are recomputed by update_rings_geometries
                rings_layer.features = df

            # Ensure rings_layer.features exists before update_rings_geometries
            if getattr(rings_layer, "features", None) is None or rings_layer.features.empty:
                rings_layer.features = pd.DataFrame({"RBXY": [], "YEAR": [], "enabled": []})

            new_rings_table, new_rings_raster, new_colormap = update_rings_geometries(
                rings_table=rings_layer.features,
                last_year=rings_layer.metadata["rings_outmost_complete_year"],
                image_shape=rings_layer.data.shape,
            )

            rings_layer.features = new_rings_table
            rings_layer.data = new_rings_raster
            rings_layer.colormap = new_colormap


class SingleSampleSegmentationWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        # The rows bring their own labels, a label column would only indent them
        super().__init__(labels=False)
        self._viewer = viewer
        prevent_duplicate_dock_if_already_present(
            self, viewer, "3A - Detect cells & rings (individual image)"
        )
        self.settings = SettingsManager()

        # Ensure model weights are present before listing them below.
        # (Moved out of import time; only downloads if the dirs are missing.)
        check_assets_and_download(str(CELLS_MODELS_PATH), "cells_models.zip")
        check_assets_and_download(str(RINGS_MODELS_PATH), "rings_models.zip")

        # Cells segmentation checkbox and model selection
        self._segment_cells_checkbox = CheckBox(
            value=True, label="Cell Segmentation:"
        )
        self._segment_cells_checkbox.changed.connect(
            self._update_cells_model_visibility
        )

        # Cells model selection
        self._cells_model_weights_file = ComboBox(
            choices=self._get_model_files(where=CELLS_MODELS_PATH),
            label="",
        )

        self._cells_row = Container(
            widgets=[
                self._segment_cells_checkbox,
                self._cells_model_weights_file,
            ],
            layout="horizontal",
            labels=True,
        )
        self._cells_row.margins = (0, 0, 0, 0)
        if hasattr(self._cells_row.native, "layout"):
            layout = self._cells_row.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)

        # Rings segmentation checkbox and model selection
        self._segment_rings_checkbox = CheckBox(
            value=True, label="Ring Segmentation:"
        )
        self._segment_rings_checkbox.changed.connect(
            self._update_rings_model_visibility
        )

        # Rings model selection
        self._rings_model_weights_file = ComboBox(
            choices=self._get_model_files(where=RINGS_MODELS_PATH),
            label="",
        )
        # The rings folder also holds the model for rerunning the editor's
        # segmentation from a given year; keep the standard model preselected
        if DEFAULT_RINGS_MODEL in self._rings_model_weights_file.choices:
            self._rings_model_weights_file.value = DEFAULT_RINGS_MODEL

        self._rings_row = Container(
            widgets=[
                self._segment_rings_checkbox,
                self._rings_model_weights_file,
            ],
            layout="horizontal",
            labels=True,
        )
        self._rings_row.margins = (0, 0, 0, 0)
        if hasattr(self._rings_row.native, "layout"):
            layout = self._rings_row.native.layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)

        # Equalize checkbox widths so the model selection dropdowns align
        # and take equal width, with sufficient padding to prevent truncation
        checkbox_width = max(
            self._segment_cells_checkbox.native.sizeHint().width(),
            self._segment_rings_checkbox.native.sizeHint().width(),
        ) + 25
        self._segment_cells_checkbox.native.setFixedWidth(checkbox_width)
        self._segment_rings_checkbox.native.setFixedWidth(checkbox_width)

        # Equalize model selection dropdown widths based on all available choices
        # across cell and ring models
        all_model_choices = list(self._cells_model_weights_file.choices) + list(
            self._rings_model_weights_file.choices
        )
        font_metrics = self._cells_model_weights_file.native.fontMetrics()
        max_text_width = max(
            (
                font_metrics.horizontalAdvance(str(choice))
                for choice in all_model_choices
            ),
            default=0,
        )
        model_dropdown_width = max_text_width + 70
        self._cells_model_weights_file.native.setFixedWidth(model_dropdown_width)
        self._rings_model_weights_file.native.setFixedWidth(model_dropdown_width)

        if hasattr(self._cells_row.native, "layout") and self._cells_row.native.layout() is not None:
            cells_layout = self._cells_row.native.layout()
            cells_layout.setSpacing(LABEL_SPACING)
            cells_layout.addStretch()
        if hasattr(self._rings_row.native, "layout") and self._rings_row.native.layout() is not None:
            rings_layout = self._rings_row.native.layout()
            rings_layout.setSpacing(LABEL_SPACING)
            rings_layout.addStretch()

        # Create a button to launch the analysis, showing its progress
        self._run_segmentation_button = JobButton(
            text="Run Segmentation", start=self._run_segmentation
        )

        # Append the widgets to the container
        self.extend(
            [
                self._cells_row,
                self._rings_row,
                self._run_segmentation_button,
            ]
        )

        # Initialize visibility of model selection widgets and button state
        self._update_cells_model_visibility()
        self._update_rings_model_visibility()
        self._update_run_button_state()

    def _get_valid_layers(self, widget=None) -> list:
        """Get layers that are both Labels type and match the scan file extension."""
        scan_extension = self.settings.get(
            "file_extensions.scan_file_extension"
        )[0]
        valid_layers = []

        for layer in self._viewer.layers:
            if isinstance(layer, napari.layers.Image) and layer.name.endswith(
                scan_extension
            ):
                valid_layers.append(layer)

        return valid_layers

    def _get_model_files(self, where: str) -> tuple:
        """Get available model weight files from the weights directory."""
        return tuple(
            path.name
            for path in Path(where).iterdir()
            if not path.name.startswith(".")
        )

    def _update_cells_model_visibility(self) -> None:
        """Update visibility of cells model selection based on checkbox."""
        self._cells_model_weights_file.visible = (
            self._segment_cells_checkbox.value
        )
        self._update_run_button_state()

    def _update_rings_model_visibility(self) -> None:
        """Update visibility of rings model selection based on checkbox."""
        self._rings_model_weights_file.visible = (
            self._segment_rings_checkbox.value
        )
        self._update_run_button_state()

    def _update_run_button_state(self) -> None:
        """Update enabled state of run button based on checkboxes."""
        self._run_segmentation_button.enabled = (
            self._segment_cells_checkbox.value
            or self._segment_rings_checkbox.value
        )

    def _extract_base_name(self, layer_name: str) -> str:
        """Extract base name from layer name by removing the extension."""
        scan_extension = self.settings.get(
            "file_extensions.scan_file_extension"
        )[0]
        if layer_name.endswith(scan_extension):
            return layer_name[: -len(scan_extension)]
        return layer_name

    def _run_segmentation(self) -> JobHandle | None:
        """Run the segmentation in the background."""

        # --- FIX OUTDATED METADATA STEMS (silent, filesystem-truth based) ---
        proj = self.settings.get("project_directory")
        if isinstance(proj, str) and proj:
            fix_sample_stem_paths_in_project(proj)

        # Get the input layer
        valid_layers = self._get_valid_layers()
        if not valid_layers:
            QMessageBox.warning(
                None, "Error", "Please open a thin section image first"
            )
            return

        self.input_layer = valid_layers[0]

        # Check if at least one segmentation method is selected
        if not (
            self._segment_cells_checkbox.value
            or self._segment_rings_checkbox.value
        ):
            QMessageBox.warning(
                None, "Error", "Please select at least one segmentation method"
            )
            return

        # Validate cells model if cells segmentation is selected
        self.cells_model_file = None
        if self._segment_cells_checkbox.value:
            self.cells_model_file = self._cells_model_weights_file.value
            if not self.cells_model_file:
                QMessageBox.warning(
                    None, "Error", "Please select a cells model file"
                )
                return

        # Validate rings model if rings segmentation is selected
        self.rings_model_file = None
        if self._segment_rings_checkbox.value:
            self.rings_model_file = self._rings_model_weights_file.value
            if not self.rings_model_file:
                QMessageBox.warning(
                    None, "Error", "Please select a rings model file"
                )
                return

        # Extract base name from input layer
        base_name = self._extract_base_name(self.input_layer.name)
        cells_extension = self.settings.get(
            "file_extensions.cells_file_extension"
        )[0]
        rings_extension = self.settings.get(
            "file_extensions.rings_file_extension"
        )[0]

        def on_segmented(results: Dict[str, Any]) -> None:
            self._add_result_layers(results)
            show_info("Segmentation complete")

        show_info("Segmentation in progress...")
        return get_runner().submit(
            Job(
                "Segmentation",
                segment_sample,
                args=(
                    self.input_layer.data,
                    (
                        CELLS_MODELS_PATH / self.cells_model_file
                        if self.cells_model_file
                        else None
                    ),
                    (
                        RINGS_MODELS_PATH / self.rings_model_file
                        if self.rings_model_file
                        else None
                    ),
                    self.settings.get("processing.try_to_use_gpu"),
                    f"{base_name}{cells_extension}",
                    f"{base_name}{rings_extension}",
                ),
                tags={JobTag.SEGMENTATION},
            ),
            on_result=on_segmented,
        )

    def _add_result_layers(self, results: Dict[str, Any]) -> None:
        input_scale = self.input_layer.scale
        sample_metadata = {
            k: v
            for k, v in self.input_layer.metadata.items()
            if isinstance(k, str)
            and k.startswith(SAMPLE_METADATA_PREFIXES)
        }

        # Ensure stem is propagated to derived layers
        stem = self.input_layer.metadata.get("sample_stem_path")
        if isinstance(stem, str) and stem.strip():
            sample_metadata["sample_stem_path"] = stem

        apply_segmentation_results_to_viewer(
            self._viewer,
            results=results,
            settings=self.settings,
            input_scale=input_scale,
            sample_metadata=sample_metadata,
            sample_stem_path=stem,
            cells_model_file=self.cells_model_file,
            rings_model_file=self.rings_model_file,
        )

        # Write the annotated image for an immediate check of the segmentation,
        # drawn from the scan and masks in memory
        save_layers_annotated_image(self._viewer.layers, stem)

