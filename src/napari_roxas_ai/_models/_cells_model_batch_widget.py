import glob
import os
import re
from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    ComboBox,
    Container,
    PushButton,
)
from PIL import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog

from ._cells_model import CellsSegmentationModel

if TYPE_CHECKING:
    import napari

# Settings
module_path = os.path.abspath(__file__).rsplit("/", 1)[0]
excluded_path_words = [
    "annotated",
    "Preview",
    "ReferenceSeries",
    "ReferenceSeriesLong",
]


class Worker(QObject):
    finished = Signal()
    result_ready = Signal(object)  # One images to send

    def __init__(
        self,
        input_directory_path,
        output_directory_path: str,
        model_weights_file: str,
    ):
        super().__init__()
        self.input_directory_path = input_directory_path
        self.output_directory_path = output_directory_path
        self.model_weights_file = (
            f"{module_path}/_weights/{model_weights_file}"
        )

    def run(self):
        # Set up model
        model = CellsSegmentationModel()
        model.load_weights(self.model_weights_file)

        # Get all input files paths. Currently, we assume that jpg files are inputs
        files_in = sorted(
            glob.glob(self.input_directory_path + "/**/*.jpg", recursive=True)
        )
        pattern = re.compile(f'({"|".join(excluded_path_words)})')
        files_in = [i for i in files_in if not pattern.search(i)]

        # Create output files paths to mimic the input directory hierarchy
        files_out = [
            f"{self.output_directory_path}/{f[len(self.input_directory_path)+1:][:-4]}"
            for f in files_in
        ]

        # Perform inference on images, and save the outputs as .png files
        for _image_num, (img_path, out_path) in enumerate(
            zip(files_in, files_out)
        ):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with Image.open(img_path) as img:
                img_array = np.array(img)
            prediction = model.infer(img_array)
            Image.fromarray(prediction.astype("uint8")).save(f"{out_path}.png")

        self.finished.emit()


class CellsModelBatchWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Get input directory
        self._input_file_dialog_button = PushButton(
            text="Input Directory: None"
        )
        self._input_file_dialog_button.changed.connect(
            self._open_input_file_dialog
        )

        # Get output directory
        self._output_file_dialog_button = PushButton(
            text="Output Directory: None"
        )
        self._output_file_dialog_button.changed.connect(
            self._open_output_file_dialog
        )

        # Scan weights directory for weight files (currrently no restiction on file names)
        self.model_weights_file = ComboBox(
            choices=tuple(os.listdir(f"{module_path}/_weights/")),
            label="Model Weights",
        )

        # Create a button to launch the analysis
        self._run_analysis_button = PushButton(text="Run Model")
        self._run_analysis_button.changed.connect(self._run_analysis)

        # Append the widgets to the container
        self.extend(
            [
                self._input_file_dialog_button,
                self._output_file_dialog_button,
                self.model_weights_file,
                self._run_analysis_button,
            ]
        )

        # Initialize output file path
        self.input_directory_path = None
        self.output_directory_path = None

    def _open_input_file_dialog(self):
        """Open a file dialog to select the input directory path."""
        self.input_directory_path = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Input Directory",
        )
        self._input_file_dialog_button.text = (
            f"Input Directory: {self.input_directory_path}"
        )

    def _open_output_file_dialog(self):
        """Open a file dialog to select the output directory path."""
        self.output_directory_path = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Output Directory",
        )
        self._output_file_dialog_button.text = (
            f"Output Directory: {self.output_directory_path}"
        )

    def _run_analysis(self):
        """Run the analysis in a separate thread."""

        # Check that there is an input directory
        if self.input_directory_path is None:
            raise ValueError("Input directory is not set.")

        # Check that there is an output directory
        if self.output_directory_path is None:
            raise ValueError("Output directory is not set.")

        # Get the selected weights file
        if self.model_weights_file.value is None:
            raise ValueError("Model weights file is not set.")

        # Run the analysis in a separate thread
        self._run_in_thread(
            self.input_directory_path,
            self.output_directory_path,
            self.model_weights_file.value,
        )

    def _run_in_thread(
        self,
        input_directory_path: str,
        output_directory_path: str,
        model_weights_file: str,
    ):
        """Run the analysis in a separate thread."""
        self.worker_thread = QThread()
        self.worker = Worker(
            input_directory_path, output_directory_path, model_weights_file
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()
