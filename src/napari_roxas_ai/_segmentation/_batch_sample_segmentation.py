import glob
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd
import torch
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    ProgressBar,
    PushButton,
)
from PIL import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox
from torch.package import PackageImporter

from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._reader import get_metadata_from_file, read_scan_file
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._writer import write_single_layer

from ._cells_model import CellsSegmentationModel

if TYPE_CHECKING:
    import napari

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

MODULE_PATH = Path(__file__).parent.absolute()
CELLS_MODELS_PATH = MODULE_PATH / "_models" / "_cells"
RINGS_MODELS_PATH = MODULE_PATH / "_models" / "_rings"

settings = SettingsManager()


class Worker(QObject):
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(
        self,
        input_directory_path: str,
        segment_cells: bool = True,
        cells_model_weights_file: Optional[str] = None,
        segment_rings: bool = True,
        rings_model_weights_file: Optional[str] = None,
    ):
        super().__init__()
        self.input_directory_path = input_directory_path
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

        # Getting input files
        self.scan_content_ext = settings.get(
            "file_extensions.scan_file_extension"
        )[0]
        self.scan_file_paths = sorted(
            glob.glob(
                str(
                    Path(self.input_directory_path)
                    / "**"
                    / f"*{self.scan_content_ext}.*"
                ),
                recursive=True,
            )
        )

        # Setup for cells segmentation
        if self.segment_cells:
            # Set up cells model
            self.cells_model = CellsSegmentationModel()
            self.cells_model.load_weights(self.cells_model_weights_file)
            self.cells_model.available_device = (
                "cuda"
                if torch.cuda.is_available()
                and settings.get("processing.try_to_use_gpu")
                else (
                    "mps"
                    if torch.mps.is_available()
                    and settings.get("processing.try_to_use_gpu")
                    else "cpu"
                )
            )
            self.cells_model.to(device=self.cells_model.available_device)
            self.cells_model.use_autocast = bool(
                torch.amp.autocast_mode.is_autocast_available(
                    self.cells_model.device.type
                )
                and settings.get("processing.try_to_use_gpu")
                and (
                    self.cells_model.device == "cuda"
                    or self.cells_model.device == "mps"
                )
            )

            self.cells_content_ext = settings.get(
                "file_extensions.cells_file_extension"
            )[0]

        # Setup for rings segmentation
        if self.segment_rings:
            # Set up rings model
            self.rings_model = PackageImporter(
                self.rings_model_weights_file
            ).load_pickle("LinearRingModel", "model.pkl")
            self.rings_model.available_device = (
                "cuda"
                if torch.cuda.is_available()
                and settings.get("processing.try_to_use_gpu")
                else (
                    "mps"
                    if torch.mps.is_available()
                    and settings.get("processing.try_to_use_gpu")
                    else "cpu"
                )
            )
            self.rings_model.to(device=self.rings_model.available_device)
            # Fix for problem with model object; device attribute is not updated with to()
            self.rings_model.device = self.rings_model.available_device
            self.rings_model.use_autocast = bool(
                torch.amp.autocast_mode.is_autocast_available(
                    self.rings_model.device
                )
                and settings.get("processing.try_to_use_gpu")
                and (
                    self.rings_model.device == "cuda"
                    or self.rings_model.device == "mps"
                )
            )

            self.rings_content_ext = settings.get(
                "file_extensions.rings_file_extension"
            )[0]

    def run(self):

        factor = int(self.segment_cells) + int(self.segment_rings)
        total = len(self.scan_file_paths) * factor
        i = 0

        for scan_file_path in self.scan_file_paths:

            scan_data, scan_add_kwargs, _ = read_scan_file(scan_file_path)

            sample_metadata = {
                k: v
                for k, v in scan_add_kwargs["metadata"].items()
                if isinstance(k, str) and k.startswith("sample_")
            }

            # Process cells if requested
            if self.segment_cells:

                # Emit progress signal
                self.progress.emit(i, total)
                i += 1

                # Perform inference
                cells_labels = self.cells_model.infer(scan_data)

                # Create outputs
                cells_layer_name = (
                    f"{sample_metadata['sample_name']}{self.cells_content_ext}"
                )
                cells_data = (cells_labels / 255).astype("uint8")
                cells_add_kwargs = {
                    "name": cells_layer_name,
                    "scale": scan_add_kwargs["scale"],
                    "features": pd.DataFrame(),
                    "metadata": {},
                }
                cells_add_kwargs["metadata"].update(sample_metadata)
                cells_add_kwargs["metadata"].update(
                    {
                        "cells_segmentation_model": Path(
                            self.cells_model_weights_file
                        ).name,
                        "cells_segmentation_datetime": datetime.now().isoformat(),
                    }
                )

                # Save to file (the file extension in the path argument is ignored)
                write_single_layer(
                    path=scan_file_path, data=cells_data, meta=cells_add_kwargs
                )

            # Process rings if requested
            if self.segment_rings:

                # Emit progress signal
                self.progress.emit(i, total)
                i += 1

                # Perform inference
                rings_labels, rings_boundaries = self.rings_model.infer(
                    scan_data
                )

                # Create a DataFrame from boundaries
                boundary_data = []
                for _i, boundary in enumerate(rings_boundaries):
                    # Convert to numpy or list, whichever is more appropriate
                    if isinstance(boundary, torch.Tensor):
                        coords = boundary.cpu().numpy().tolist()
                    else:
                        coords = boundary
                    boundary_data.append({"boundary_coordinates": coords})

                boundaries_df = pd.DataFrame(boundary_data)

                # Create outputs
                rings_layer_name = (
                    f"{sample_metadata['sample_name']}{self.rings_content_ext}"
                )
                rings_data = rings_labels.astype("int32")
                rings_add_kwargs = {
                    "name": rings_layer_name,
                    "scale": scan_add_kwargs["scale"],
                    "features": boundaries_df,
                    "metadata": {},
                }
                rings_add_kwargs["metadata"].update(sample_metadata)

                metadata_file_contents = get_metadata_from_file(
                    path=sample_metadata["sample_stem_path"], path_is_stem=True
                )
                default_rings_year_value = [
                    field["default"]
                    for field in settings.get("samples_metadata.fields")
                    if field["id"] == "rings_outmost_complete_year"
                ][0]

                rings_add_kwargs["metadata"].update(
                    {
                        "rings_outmost_complete_year": (
                            metadata_file_contents[
                                "rings_outmost_complete_year"
                            ]
                            if metadata_file_contents
                            else default_rings_year_value
                        ),
                        "rings_segmentation_model": Path(
                            self.rings_model_weights_file
                        ).name,
                        "rings_segmentation_datetime": datetime.now().isoformat(),
                    }
                )

                rings_add_kwargs["features"], rings_data, _ = (
                    update_rings_geometries(
                        rings_table=rings_add_kwargs["features"],
                        last_year=rings_add_kwargs["metadata"][
                            "rings_outmost_complete_year"
                        ],
                        image_shape=rings_data.shape,
                    )
                )

                # Save to file (the file extension in the path argument is ignored)
                write_single_layer(
                    path=scan_file_path, data=rings_data, meta=rings_add_kwargs
                )

        self.progress.emit(total, total)
        self.finished.emit()


class BatchSampleSegmentationWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Get input directory
        self.input_directory_path = settings.get("project_directory")
        self._input_file_dialog_button = PushButton(
            text=f"Input Directory: {self.input_directory_path}",
        )
        self._input_file_dialog_button.changed.connect(
            self._open_input_file_dialog
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

        # Add a progress bar with a description
        self.progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Progress"
        )

        # Append the widgets to the container
        self.extend(
            [
                self._input_file_dialog_button,
                self._segment_cells_checkbox,
                self._cells_model_weights_file,
                self._segment_rings_checkbox,
                self._rings_model_weights_file,
                self._run_segmentation_button,
                self.progress_bar,
            ]
        )

        # Initialize visibility of model selection widgets
        self._update_cells_model_visibility()
        self._update_rings_model_visibility()

    def _open_input_file_dialog(self):
        """Open a file dialog to select the input directory path."""
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Input Directory",
            directory=self.input_directory_path,
        )
        if directory:
            self.input_directory_path = directory
            self._input_file_dialog_button.text = (
                f"Input Directory: {self.input_directory_path}"
            )

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

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self.progress_bar.value = percentage
            self.progress_bar.visible = True

    def _run_segmentation(self) -> None:
        """Run the segmentation analysis in a separate thread."""
        # Check that there is an input directory
        if self.input_directory_path is None:
            QMessageBox.warning(
                None, "Error", "Please select an input directory"
            )
            return

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

        # Prepare worker to run the analysis in a separate thread
        self.worker_thread = QThread()
        self.worker = Worker(
            self.input_directory_path,
            self._segment_cells_checkbox.value,
            self.cells_model_file,
            self._segment_rings_checkbox.value,
            self.rings_model_file,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Connect progress signal
        self.worker.progress.connect(self._update_progress)

        # Disable the run button while processing
        self._run_segmentation_button.enabled = False
        self.worker_thread.finished.connect(
            lambda: setattr(self._run_segmentation_button, "enabled", True)
        )

        # Run the analysis in a separate thread
        self.worker_thread.start()
