import glob
import json
import os
import shutil
from typing import TYPE_CHECKING, Dict, List, Optional

from magicgui.widgets import (
    Container,
    ProgressBar,
    PushButton,
)
from PIL import ExifTags, Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import QFileDialog, QMessageBox

from .._settings._settings_manager import SettingsManager
from ._metadata_dialog import MetadataDialog

if TYPE_CHECKING:
    import napari

# Common image extensions
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".jp2"]

Image.MAX_IMAGE_PIXELS = (
    None  # disables the decompressionbomb warning for large images
)


class Worker(QObject):
    """Worker for processing files in a separate thread."""

    finished = Signal()
    progress = Signal(int, int)  # current, total
    metadata_request = Signal(str)  # filename
    abort_signal = Signal()

    def __init__(
        self,
        source_directory: str,
        target_directory: str,
        default_metadata: Optional[Dict] = None,
        apply_to_all: bool = False,
        authorized_sample_types: List[str] = None,
        default_scale: float = 2.2675,
        default_angle: float = 0.0,
        same_directory: bool = False,
        scan_file_prefix: str = ".scan",
        metadata_file_extension: str = ".metadata.json",
        roxas_file_extensions: List[str] = None,
    ):
        super().__init__()
        self.source_directory = source_directory
        self.target_directory = target_directory
        self.default_metadata = default_metadata
        self.apply_to_all = apply_to_all
        self.should_stop = False
        self.current_file_index = 0
        self.all_files = []
        self.authorized_sample_types = authorized_sample_types or [
            "conifer",
            "angiosperm",
        ]
        self.default_scale = default_scale
        self.default_angle = default_angle
        self.same_directory = same_directory
        self.scan_file_prefix = scan_file_prefix
        self.metadata_file_extension = metadata_file_extension
        self.roxas_file_extensions = roxas_file_extensions or []

    def run(self):
        """Process all image files from source to target directory."""
        # Get all image files in source directory
        self.all_files = []
        for ext in IMAGE_EXTENSIONS:
            self.all_files.extend(
                glob.glob(f"{self.source_directory}/**/*{ext}", recursive=True)
            )
            self.all_files.extend(
                glob.glob(
                    f"{self.source_directory}/**/*{ext.upper()}",
                    recursive=True,
                )
            )

        # Filter out files that already have ROXAS extensions
        if self.roxas_file_extensions:
            filtered_files = []
            for file_path in self.all_files:
                file_name = os.path.basename(file_path)
                skip_file = False

                # Check if the file name contains any ROXAS extension
                for ext in self.roxas_file_extensions:
                    if ext in file_name:
                        skip_file = True
                        print(f"Skipping already processed file: {file_path}")
                        break

                if not skip_file:
                    filtered_files.append(file_path)

            self.all_files = filtered_files

        # Sort files for consistent processing
        self.all_files = sorted(self.all_files)

        # Create target directory if it doesn't exist
        os.makedirs(self.target_directory, exist_ok=True)

        total = len(self.all_files)
        self.current_file_index = 0

        if total > 0:
            self._process_next_file()
        else:
            self.finished.emit()

    def _process_next_file(self):
        """Process the next file in the queue."""
        if self.should_stop or self.current_file_index >= len(self.all_files):
            # Processing complete or stopped
            if not self.should_stop:
                self.progress.emit(len(self.all_files), len(self.all_files))
            self.finished.emit()
            return

        # Get current file
        file_path = self.all_files[self.current_file_index]

        # Emit progress signal
        self.progress.emit(self.current_file_index, len(self.all_files))

        # Extract relative path from source directory
        rel_path = file_path[len(self.source_directory) + 1 :]

        # Get directory and filename
        dir_path = os.path.dirname(rel_path)
        file_name = os.path.basename(rel_path)
        base_name, file_ext = os.path.splitext(file_name)

        # Create target directory structure
        target_dir = os.path.join(self.target_directory, dir_path)
        os.makedirs(target_dir, exist_ok=True)

        # Create new file paths using settings-defined extensions
        new_image_path = os.path.join(
            target_dir, f"{base_name}{self.scan_file_prefix}{file_ext}"
        )
        metadata_path = os.path.join(
            target_dir, f"{base_name}{self.metadata_file_extension}"
        )

        # Copy the image without conversion
        img_metadata = self._copy_image(file_path, new_image_path)

        # If source and target are the same, remove the original file if different
        if self.same_directory and file_path != new_image_path:
            try:
                os.remove(file_path)
                print(f"Removed original file: {file_path}")
            except (PermissionError, OSError) as e:
                print(f"Error removing original file {file_path}: {e}")

        # Get or create metadata
        if self.apply_to_all and self.default_metadata:
            # Use default metadata for all files but ensure consistent field order
            metadata = {}
            # Start with sample_name to maintain field order
            metadata["sample_name"] = base_name
            # Then add the rest of the fields in the expected order
            metadata["sample_type"] = self.default_metadata.get(
                "sample_type", ""
            )
            metadata["sample_scale"] = self.default_metadata.get(
                "sample_scale", self.default_scale
            )
            metadata["sample_angle"] = self.default_metadata.get(
                "sample_angle", self.default_angle
            )

            # Include image metadata if available
            if img_metadata:
                metadata.update(img_metadata)

            self._save_metadata(metadata, metadata_path)

            # Move to next file
            self.current_file_index += 1
            self._process_next_file()
        else:
            # Request metadata from user
            self.metadata_request.emit(base_name)
            # Processing will continue when metadata is provided via set_metadata

    def set_metadata(self, metadata: Dict, apply_to_all: bool = False):
        """Set metadata for current file and continue processing."""
        # Get the current file path and extract necessary components
        file_path = self.all_files[self.current_file_index]
        rel_path = file_path[len(self.source_directory) + 1 :]
        dir_path = os.path.dirname(rel_path)
        base_name = metadata["sample_name"]
        _, file_ext = os.path.splitext(os.path.basename(rel_path))

        # Create target path for metadata file using settings-defined extension
        target_dir = os.path.join(self.target_directory, dir_path)
        metadata_path = os.path.join(
            target_dir, f"{base_name}{self.metadata_file_extension}"
        )

        # Create new image path to extract metadata from
        new_image_path = os.path.join(
            target_dir, f"{base_name}{self.scan_file_prefix}{file_ext}"
        )

        # Extract image metadata (copy if not already done)
        img_metadata = self._copy_image(file_path, new_image_path)

        # Include image metadata in the user-provided metadata
        if img_metadata:
            metadata.update(img_metadata)

        # Save the metadata file
        self._save_metadata(metadata, metadata_path)

        # Update apply_to_all flag
        self.apply_to_all = apply_to_all
        if apply_to_all:
            # Create a copy of metadata for future files
            self.default_metadata = {}
            # Maintain field order by explicitly copying each field except sample_name
            self.default_metadata["sample_type"] = metadata.get(
                "sample_type", ""
            )
            self.default_metadata["sample_geometry"] = metadata.get(
                "sample_geometry", ""
            )
            self.default_metadata["sample_scale"] = metadata.get(
                "sample_scale", self.default_scale
            )
            self.default_metadata["sample_angle"] = metadata.get(
                "sample_angle", self.default_angle
            )

        # Move to next file
        self.current_file_index += 1
        self._process_next_file()

    def stop(self):
        """Stop the processing."""
        self.should_stop = True
        self.abort_signal.emit()

    def _copy_image(self, source_path: str, target_path: str):
        """
        Copy image file without conversion, maintaining original format.
        Returns a dictionary with image metadata to be included in the JSON file.
        """
        try:
            # Extract metadata from the image
            with Image.open(source_path) as img:
                # Store scan image information for metadata
                img_metadata = {
                    "scan_format": img.format,
                    "scan_size": [img.width, img.height],
                    "scan_mode": img.mode,
                }

                # Extract and preserve EXIF data if available
                if hasattr(img, "_getexif") and img._getexif() is not None:
                    exif_dict = {
                        ExifTags.TAGS.get(tag_id, tag_id): value
                        for tag_id, value in img._getexif().items()
                        if tag_id in ExifTags.TAGS
                    }
                    img_metadata["scan_exif"] = exif_dict
                else:
                    img_metadata["scan_exif"] = None

                # Extract other image info
                scan_info = {}
                for key, value in img.info.items():
                    if (
                        key != "exif"
                        and key != "icc_profile"
                        and isinstance(
                            value, (str, int, float, bool, tuple, list)
                        )
                    ):
                        scan_info[key] = value

                img_metadata["scan_info"] = scan_info

                # Copy the file to the target location
                shutil.copy2(source_path, target_path)

                return img_metadata

        except (OSError, ValueError, Image.UnidentifiedImageError) as e:
            print(f"Error processing {source_path}: {e}")
            return None  # No metadata available

    def _save_metadata(self, metadata: Dict, metadata_path: str):
        """Save metadata to a JSON file."""
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        except (OSError, TypeError) as e:
            print(f"Error saving metadata to {metadata_path}: {e}")


class PreparationWidget(Container):
    """Widget for preparing sample images and metadata for ROXAS analysis."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        # Initialize settings manager to get settings
        self.settings_manager = SettingsManager()

        # Get authorized sample types from settings
        self.authorized_sample_types = self.settings_manager.get(
            "authorised_sample_types",
            [
                "conifer",
                "angiosperm",
            ],  # Default values if not found in settings
        )

        # Get authorized sample geometries from settings
        self.authorized_sample_geometries = self.settings_manager.get(
            "authorised_sample_geometries",
            [
                "linear",
                "circular",
            ],  # Default values if not found in settings
        )

        # Get default scale and angle from settings
        self.default_scale = self.settings_manager.get(
            "default_scale",
            2.2675,  # Default value if not found in settings (pixels/μm)
        )
        self.default_angle = self.settings_manager.get(
            "default_angle", 0.0  # Default value if not found in settings
        )

        # Get file extension settings
        scan_file_extension = self.settings_manager.get(
            "scan_file_extension", [".scan", ".jpg"]
        )
        # Just take the first element (prefix) from the scan_file_extension
        self.scan_file_prefix = (
            scan_file_extension[0] if scan_file_extension else ".scan"
        )

        metadata_file_extension_parts = self.settings_manager.get(
            "metadata_file_extension", [".metadata", ".json"]
        )
        self.metadata_file_extension = "".join(metadata_file_extension_parts)

        # Get ROXAS file extensions to avoid processing files that have already been prepared
        self.roxas_file_extensions = self.settings_manager.get(
            "roxas_file_extensions", []
        )

        # Source directory selector
        self._source_dialog_button = PushButton(text="Source Directory: None")
        self._source_dialog_button.changed.connect(self._open_source_dialog)

        # Project directory selector
        self._project_dialog_button = PushButton(
            text="Project Directory: None"
        )
        self._project_dialog_button.changed.connect(self._open_project_dialog)

        # Progress bar
        self._progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Processing Progress"
        )

        # Start button
        self._start_button = PushButton(text="Start Processing")
        self._start_button.changed.connect(self._start_processing)

        # Cancel button (initially hidden)
        self._cancel_button = PushButton(text="Cancel", visible=False)
        self._cancel_button.changed.connect(self._cancel_processing)

        # Append all widgets to the container
        self.extend(
            [
                self._source_dialog_button,
                self._project_dialog_button,
                self._start_button,
                self._cancel_button,
                self._progress_bar,
            ]
        )

        # Initialize paths
        self.source_directory = None
        self.project_directory = None

        # Worker and thread references
        self.worker = None
        self.worker_thread = None

        # Metadata dialog
        self.metadata_dialog = None

        # Flag to track if source and project are the same
        self.same_directory = False

    def _open_source_dialog(self):
        """Open file dialog to select source directory."""
        self.source_directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Source Directory with Images",
        )
        if self.source_directory:
            self._source_dialog_button.text = (
                f"Source Directory: {self.source_directory}"
            )

            # If project directory is not defined yet, set it to the source directory
            if not self.project_directory:
                self.project_directory = self.source_directory
                self._project_dialog_button.text = (
                    f"Project Directory: {self.project_directory}"
                )

    def _open_project_dialog(self):
        """Open file dialog to select project directory."""
        self.project_directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
        )
        if self.project_directory:
            self._project_dialog_button.text = (
                f"Project Directory: {self.project_directory}"
            )

    def _start_processing(self):
        """Start the file processing in a separate thread."""
        # Validate inputs
        if not self.source_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a source directory."
            )
            return

        if not self.project_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a project directory."
            )
            return

        # Check if source and project directories are the same
        self.same_directory = os.path.normpath(
            self.source_directory
        ) == os.path.normpath(self.project_directory)

        # If the directories are the same, show warning and confirm with user
        if self.same_directory:
            confirm = QMessageBox()
            confirm.setIcon(QMessageBox.Warning)
            confirm.setWindowTitle("Warning: Source Modification")
            confirm.setText("The source and project directories are the same.")
            confirm.setInformativeText(
                f"Original image files will be replaced with {self.scan_file_prefix}* files. This operation cannot be undone."
            )
            confirm.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            confirm.setDefaultButton(QMessageBox.Cancel)

            result = confirm.exec_()

            if result == QMessageBox.Cancel:
                return

        # Update UI for processing
        self._start_button.visible = False
        self._cancel_button.visible = True
        self._progress_bar.visible = True
        self._progress_bar.value = 0

        # Start processing in a separate thread
        self._run_in_thread(
            self.source_directory,
            self.project_directory,
            None,  # No default metadata
            False,  # Not applying to all initially
        )

    def _cancel_processing(self):
        """Cancel the current processing job."""
        if self.worker:
            self.worker.stop()

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self._progress_bar.value = percentage

    def _request_metadata(self, filename):
        """Show dialog to request metadata for a file."""
        # Create default metadata with just the sample name
        default_metadata = {
            "sample_name": filename,
            "sample_type": (
                self.authorized_sample_types[0]
                if self.authorized_sample_types
                else "conifer"
            ),  # Use first authorized type
            "sample_geometry": (
                self.authorized_sample_geometries[0]
                if self.authorized_sample_geometries
                else "linear"
            ),  # Use first authorized geometry
            "sample_scale": self.default_scale,  # Use default scale from settings
            "sample_angle": self.default_angle,  # Use default angle from settings
        }

        # Show dialog to get metadata
        self.metadata_dialog = MetadataDialog(
            default_metadata,
            self.authorized_sample_types,
            self.authorized_sample_geometries,
        )
        result = self.metadata_dialog.exec_()

        if result:
            # User confirmed the dialog
            metadata = self.metadata_dialog.get_metadata()
            apply_to_all = self.metadata_dialog.apply_to_all()

            # Send metadata back to worker
            self.worker.set_metadata(metadata, apply_to_all)
        else:
            # User cancelled - stop processing
            self._cancel_processing()

    def _processing_finished(self):
        """Handle the completion of processing."""
        # Reset UI
        self._start_button.visible = True
        self._cancel_button.visible = False

        # Show completion message
        QMessageBox.information(
            None, "Complete", "File processing completed successfully."
        )

    def _run_in_thread(
        self,
        source_directory: str,
        target_directory: str,
        default_metadata: Optional[Dict] = None,
        apply_to_all: bool = False,
    ):
        """Run the processing in a separate thread."""
        self.worker_thread = QThread()
        self.worker = Worker(
            source_directory,
            target_directory,
            default_metadata,
            apply_to_all,
            self.authorized_sample_types,
            self.default_scale,
            self.default_angle,
            self.same_directory,
            self.scan_file_prefix,
            self.metadata_file_extension,
            self.roxas_file_extensions,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._processing_finished)

        # Connect progress signal
        self.worker.progress.connect(self._update_progress)

        # Connect metadata request signal
        self.worker.metadata_request.connect(self._request_metadata)

        # Start the thread
        self.worker_thread.start()
