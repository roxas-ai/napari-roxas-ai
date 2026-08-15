from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import napari.layers
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from PIL import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._assets_files import check_assets_and_download
from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._reader import get_metadata_from_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._utils import make_binary_labels_colormap

from .._utils._fix_sample_stem_path import fix_sample_stem_paths_in_project
from .._utils._segmentation_postprocess import remove_border_touching_components

# NOTE: torch, torch.package.PackageImporter and ._cells_model.CellsSegmentationModel
# are intentionally imported lazily inside Worker.run() instead of at module level.
# They pull in the heavy ML stack (torch / pytorch-lightning /
# segmentation-models-pytorch) which is the dominant import cost, so deferring it
# keeps opening this widget fast; the cost is paid only when segmentation runs.

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

MODULE_PATH = Path(__file__).parent.absolute()
CELLS_MODELS_PATH = MODULE_PATH / "_models" / "_cells"
RINGS_MODELS_PATH = MODULE_PATH / "_models" / "_rings"

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

                # If existing features already contain YEAR etc., preserve them and only update RBXY
                if (
                        getattr(rings_layer, "features", None) is not None
                        and not rings_layer.features.empty
                        and "YEAR" in rings_layer.features.columns
                        and "RBXY" in df.columns
                ):
                    old_df = rings_layer.features.copy()
                    n = min(len(old_df), len(df))
                    old_df = old_df.iloc[:n].copy()
                    old_df.loc[old_df.index[:n], "RBXY"] = df["RBXY"].iloc[:n].values
                    rings_layer.features = old_df
                else:
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


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(dict)  # Dictionary with results

    def __init__(
        self,
        input_array: np.ndarray,
        segment_cells: bool = True,
        cells_model_weights_file: Optional[str] = None,
        segment_rings: bool = True,
        rings_model_weights_file: Optional[str] = None,
        settings: SettingsManager = None,
        base_name: str = "",
    ):
        super().__init__()
        self.input_array = input_array
        self.segment_cells = segment_cells
        self.segment_rings = segment_rings
        self.cells_model_weights_file = None
        if cells_model_weights_file:
            self.cells_model_weights_file = (
                CELLS_MODELS_PATH / cells_model_weights_file
            )
        self.rings_model_weights_file = None
        if rings_model_weights_file:
            self.rings_model_weights_file = (
                RINGS_MODELS_PATH / rings_model_weights_file
            )
        self.settings = settings
        self.base_name = base_name

    def run(self):
        # Heavy ML imports are deferred to run time (see module-level note).
        import torch
        from torch.package import PackageImporter

        from ._cells_model import CellsSegmentationModel

        # Prevent OpenMP crash when running inside a QThread with PyQt6 on macOS
        torch.set_num_threads(1)

        results = {}

        # Process cells if requested
        if self.segment_cells:

            # Set up cells model
            cells_model = CellsSegmentationModel()
            cells_model.load_weights(self.cells_model_weights_file)
            cells_model.available_device = (
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
            cells_model.to(device=cells_model.available_device)
            cells_model.use_autocast = bool(
                torch.amp.autocast_mode.is_autocast_available(
                    cells_model.device.type
                )
                and self.settings.get("processing.try_to_use_gpu")
                and (
                    cells_model.device == "cuda" or cells_model.device == "mps"
                )
            )

            # Perform inference
            cells_labels = cells_model.infer(self.input_array)

            # Add to results (compute layer name first)
            suffix = self.settings.get("file_extensions.cells_file_extension")[0]
            name = f"{self.base_name}{suffix}"

            # Normalize to 0/1 and remove border-touching components
            cells_binary = (cells_labels > 0).astype("uint8")
            cells_binary = remove_border_touching_components(cells_binary)

            results["cells"] = {
                "data": cells_binary,
                "name": name,
            }

        # Process rings if requested
        if self.segment_rings:

            # NUMPY 2.0 COMPATIBILITY MONKEYPATCH:
            # The rings model package (packaged with torch.package) uses np.Inf,
            # which was deprecated and finally removed in NumPy 2.0.
            # We monkeypatch it back to np.inf to ensure legacy model code remains functional.
            import numpy as np
            if not hasattr(np, "Inf"):
                np.Inf = np.inf

            # Set up rings model
            rings_model = PackageImporter(
                self.rings_model_weights_file
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
            rings_model.device = torch.device(rings_model.available_device)
            rings_model.use_autocast = bool(
                torch.amp.autocast_mode.is_autocast_available(
                    rings_model.device.type
                )
                and self.settings.get("processing.try_to_use_gpu")
                and (
                    rings_model.device.type == "cuda" or rings_model.device.type == "mps"
                )
            )

            # Perform inference
            rings_labels, rings_boundaries = rings_model.infer(
                self.input_array
            )

            # Create a DataFrame from boundaries
            boundary_data = []
            for _i, boundary in enumerate(rings_boundaries):
                # Convert to numpy or list, whichever is more appropriate
                if isinstance(boundary, torch.Tensor):
                    coords = boundary.cpu().numpy().tolist()
                else:
                    coords = boundary
                boundary_data.append({"RBXY": coords})

            boundaries_df = pd.DataFrame(boundary_data)

            # Add to results
            suffix = self.settings.get("file_extensions.rings_file_extension")[
                0
            ]
            name = f"{self.base_name}{suffix}"
            results["rings"] = {
                "data": rings_labels.astype("int32"),
                "name": name,
                "features": boundaries_df,
            }

        # Emit results
        self.result_ready.emit(results)
        self.finished.emit()


class SingleSampleSegmentationWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self.settings = SettingsManager()

        # Ensure model weights are present before listing them below.
        # (Moved out of import time; only downloads if the dirs are missing.)
        check_assets_and_download(str(CELLS_MODELS_PATH), "cells_models.zip")
        check_assets_and_download(str(RINGS_MODELS_PATH), "rings_models.zip")

        # Create a layer selection widget filtered by scan extension
        self._input_layer_combo = ComboBox(
            label="Thin Section",
            annotation="napari.layers.Image",
            choices=self._get_valid_layers,
        )

        # Cells segmentation checkbox and model selection
        self._segment_cells_checkbox = CheckBox(
            value=True, label="Segment Cells"
        )
        self._segment_cells_checkbox.changed.connect(
            self._update_cells_model_visibility
        )

        # Cells model selection
        self._cells_model_weights_file = ComboBox(
            choices=self._get_model_files(where=CELLS_MODELS_PATH),
            label="Cells Model",
        )

        # Rings segmentation checkbox and model selection
        self._segment_rings_checkbox = CheckBox(
            value=True, label="Segment Rings"
        )
        self._segment_rings_checkbox.changed.connect(
            self._update_rings_model_visibility
        )

        # Rings model selection
        self._rings_model_weights_file = ComboBox(
            choices=self._get_model_files(where=RINGS_MODELS_PATH),
            label="Rings Model",
        )

        # Create a button to launch the analysis
        self._run_segmentation_button = PushButton(text="Run Segmentation")
        self._run_segmentation_button.changed.connect(self._run_segmentation)

        # Append the widgets to the container
        self.extend(
            [
                self._input_layer_combo,
                self._segment_cells_checkbox,
                self._cells_model_weights_file,
                self._segment_rings_checkbox,
                self._rings_model_weights_file,
                self._run_segmentation_button,
            ]
        )

        # Initialize visibility of model selection widgets
        self._update_cells_model_visibility()
        self._update_rings_model_visibility()

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
        return tuple(path.name for path in Path(where).iterdir())

    def _update_cells_model_visibility(self) -> None:
        """Update visibility of cells model selection based on checkbox."""
        self._cells_model_weights_file.visible = (
            self._segment_cells_checkbox.value
        )

    def _update_rings_model_visibility(self) -> None:
        """Update visibility of rings model selection based on checkbox."""
        self._rings_model_weights_file.visible = (
            self._segment_rings_checkbox.value
        )

    def _extract_base_name(self, layer_name: str) -> str:
        """Extract base name from layer name by removing the extension."""
        scan_extension = self.settings.get(
            "file_extensions.scan_file_extension"
        )[0]
        if layer_name.endswith(scan_extension):
            return layer_name[: -len(scan_extension)]
        return layer_name

    def _run_segmentation(self) -> None:
        """Run the segmentation analysis in a separate thread."""

        # --- FIX OUTDATED METADATA STEMS (silent, filesystem-truth based) ---
        proj = self.settings.get("project_directory")
        if isinstance(proj, str) and proj:
            fix_sample_stem_paths_in_project(proj)

        # Get the selected input layer
        if not self._input_layer_combo.value:
            QMessageBox.warning(None, "Error", "Please select an input layer")
            return

        self.input_layer = self._input_layer_combo.value

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

        # Prepare worker to run the analysis in a separate thread
        self.worker_thread = QThread()
        self.worker = Worker(
            self.input_layer.data,
            self._segment_cells_checkbox.value,
            self.cells_model_file,
            self._segment_rings_checkbox.value,
            self.rings_model_file,
            self.settings,
            base_name,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.result_ready.connect(self._add_result_layers)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Disable the run button while processing
        self._run_segmentation_button.enabled = False

        # Restore run segmentation button and show message when finished
        def on_finished():
            self._run_segmentation_button.enabled = True
            show_info("Segmentation complete")

        self.worker_thread.finished.connect(on_finished)

        show_info("Segmentation in progress...")
        # Run the analysis in a separate thread
        self.worker_thread.start()

    def _add_result_layers(self, results: Dict[str, Any]) -> None:
        input_scale = self.input_layer.scale
        sample_metadata = {
            k: v
            for k, v in self.input_layer.metadata.items()
            if isinstance(k, str) and k.startswith("sample_")
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

