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
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from .._settings._settings_manager import SettingsManager
from ._metadata_dialog import MetadataDialog

if TYPE_CHECKING:
    import napari

# Disable PIL's decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None


class FileOverwriteDialog(QDialog):
    """
    Dialog to confirm file overwrite when a duplicate file is detected.

    Provides options to keep or overwrite the file, and to apply the choice
    to all subsequent duplicate files.
    """

    def __init__(self, filename: str):
        """
        Initialize the overwrite confirmation dialog.

        Args:
            filename: Name of the file being processed
        """
        super().__init__()

        self.setWindowTitle("Duplicate File Detected")

        # Set up the dialog layout
        layout = QVBoxLayout()

        # Add file info
        layout.addWidget(
            QLabel(f"File already exists: {os.path.basename(filename)}")
        )
        layout.addWidget(QLabel("What would you like to do?"))

        # Add buttons
        self.keep_button = QPushButton("Keep Original")
        self.overwrite_button = QPushButton("Overwrite")

        self.keep_button.clicked.connect(self._keep_clicked)
        self.overwrite_button.clicked.connect(self._overwrite_clicked)

        layout.addWidget(self.keep_button)
        layout.addWidget(self.overwrite_button)

        # Add "apply to all" checkbox
        self.apply_to_all_checkbox = QCheckBox("Apply to all duplicate files")
        layout.addWidget(self.apply_to_all_checkbox)

        self.setLayout(layout)

        # Result storage
        self.overwrite = False  # Default to keeping original

    def _keep_clicked(self):
        """Handle the 'Keep Original' button click."""
        self.overwrite = False
        self.accept()

    def _overwrite_clicked(self):
        """Handle the 'Overwrite' button click."""
        self.overwrite = True
        self.accept()

    def should_overwrite(self) -> bool:
        """Return whether the file should be overwritten."""
        return self.overwrite

    def apply_to_all(self) -> bool:
        """Return whether to apply this choice to all duplicate files."""
        return self.apply_to_all_checkbox.isChecked()


class Worker(QObject):
    """
    Worker for processing files in a separate thread.

    This worker handles finding images, copying them, extracting metadata,
    and managing the processing flow.
    """

    # Signal definitions
    finished = Signal()  # Emitted when processing is complete
    progress = Signal(int, int)  # current, total - for progress reporting
    metadata_request = Signal(str)  # filename - requests metadata for a file
    abort_signal = Signal()  # Signals to abort processing
    overwrite_request = Signal(
        str
    )  # filename - requests overwrite confirmation

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
        image_file_extensions: List[str] = None,
    ):
        """Initialize the worker with necessary parameters."""
        super().__init__()
        # Directories
        self.source_directory = source_directory
        self.target_directory = target_directory

        # Metadata and processing settings
        self.default_metadata = default_metadata
        self.apply_to_all = apply_to_all
        self.authorized_sample_types = authorized_sample_types or [
            "conifer",
            "angiosperm",
        ]
        self.default_scale = default_scale
        self.default_angle = default_angle

        # File handling settings
        self.same_directory = same_directory
        self.scan_file_prefix = scan_file_prefix
        self.metadata_file_extension = metadata_file_extension
        self.roxas_file_extensions = roxas_file_extensions or []
        self.image_file_extensions = image_file_extensions or []

        # Processing state
        self.should_stop = False
        self.current_file_index = 0
        self.all_files = []
        self._current_img_metadata = None  # Temporary cache for image metadata

        # File overwrite handling
        self.overwrite_all = False
        self.keep_all = False

    def run(self):
        """
        Process all image files from source directory.

        This is the main entry point for the worker thread.
        """
        # Find all image files in source directory
        self.all_files = []

        # Gather all files with supported image extensions
        for ext in self.image_file_extensions:
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

        # Process files or finish if none found
        total = len(self.all_files)
        self.current_file_index = 0

        if total > 0:
            self._process_next_file()
        else:
            self.finished.emit()

    def _process_next_file(self):
        """
        Process the next file in the queue.

        This function handles the main processing logic for each file.
        """
        # Check if we should stop or if we've processed all files
        if self.should_stop or self.current_file_index >= len(self.all_files):
            # Processing complete or stopped
            if not self.should_stop:
                self.progress.emit(len(self.all_files), len(self.all_files))
            self.finished.emit()
            return

        # Get current file path
        file_path = self.all_files[self.current_file_index]

        # Update progress indication
        self.progress.emit(self.current_file_index, len(self.all_files))

        # Extract path components
        rel_path = os.path.relpath(file_path, self.source_directory)
        dir_path = os.path.dirname(rel_path)
        file_name = os.path.basename(rel_path)
        base_name, file_ext = os.path.splitext(file_name)

        # Create target directory structure
        target_dir = os.path.join(self.target_directory, dir_path)
        os.makedirs(target_dir, exist_ok=True)

        # Define output paths
        new_image_path = os.path.join(
            target_dir, f"{base_name}{self.scan_file_prefix}{file_ext}"
        )
        metadata_path = os.path.join(
            target_dir, f"{base_name}{self.metadata_file_extension}"
        )

        # Check if the target file already exists
        if os.path.exists(new_image_path) and not self.same_directory:
            # If we have a global policy, apply it
            if self.overwrite_all:
                # Copy and extract metadata, overwriting existing file
                self._copy_and_handle_file(
                    file_path, new_image_path, metadata_path, base_name
                )
            elif self.keep_all:
                # Skip this file, move to next one
                print(f"Keeping original file: {new_image_path}")
                self.current_file_index += 1
                self._process_next_file()
                return
            else:
                # Ask user what to do with this file
                self.overwrite_request.emit(new_image_path)
                # Processing will continue when overwrite response is received
                return
        else:
            # No conflict or same directory mode, proceed normally
            self._copy_and_handle_file(
                file_path, new_image_path, metadata_path, base_name
            )

    def _copy_and_handle_file(
        self, file_path, new_image_path, metadata_path, base_name
    ):
        """
        Copy the file, extract metadata, and handle subsequent processing.

        Args:
            file_path: Source file path
            new_image_path: Destination file path
            metadata_path: Path for metadata file
            base_name: Base name of the file (without extension)
        """
        # Copy the image and extract metadata
        img_metadata = self._copy_image(file_path, new_image_path)

        # Cache metadata for use in set_metadata if needed
        self._current_img_metadata = img_metadata

        # Handle same directory mode - remove original file if different from target
        if self.same_directory and file_path != new_image_path:
            try:
                os.remove(file_path)
                print(f"Removed original file: {file_path}")
            except (PermissionError, OSError) as e:
                print(f"Error removing original file {file_path}: {e}")

        # Apply default metadata or request from user
        if self.apply_to_all and self.default_metadata:
            # Use default metadata for this file
            metadata = self._create_metadata(
                base_name, self.default_metadata, img_metadata
            )
            self._save_metadata(metadata, metadata_path)

            # Move to next file
            self.current_file_index += 1
            self._process_next_file()
        else:
            # Request metadata from user via dialog
            self.metadata_request.emit(base_name)
            # Processing will continue when metadata is provided via set_metadata

    def set_overwrite_choice(self, overwrite: bool, apply_to_all: bool):
        """
        Handle user's overwrite choice and continue processing.

        Args:
            overwrite: Whether to overwrite the existing file
            apply_to_all: Whether to apply this choice to all files
        """
        # Update global policies if apply_to_all is selected
        if apply_to_all:
            if overwrite:
                self.overwrite_all = True
                self.keep_all = False
            else:
                self.keep_all = True
                self.overwrite_all = False

        # Get current file info
        file_path = self.all_files[self.current_file_index]
        rel_path = os.path.relpath(file_path, self.source_directory)
        dir_path = os.path.dirname(rel_path)
        file_name = os.path.basename(rel_path)
        base_name, file_ext = os.path.splitext(file_name)

        # Create paths
        target_dir = os.path.join(self.target_directory, dir_path)
        new_image_path = os.path.join(
            target_dir, f"{base_name}{self.scan_file_prefix}{file_ext}"
        )
        metadata_path = os.path.join(
            target_dir, f"{base_name}{self.metadata_file_extension}"
        )

        if overwrite:
            # Process the file, overwriting the existing one
            self._copy_and_handle_file(
                file_path, new_image_path, metadata_path, base_name
            )
        else:
            # Skip this file and move to the next one
            print(f"Keeping original file: {new_image_path}")
            self.current_file_index += 1
            self._process_next_file()

    def _create_metadata(
        self, base_name: str, default_data: Dict, img_metadata: Optional[Dict]
    ) -> Dict:
        """
        Create metadata dictionary with consistent field order.

        Args:
            base_name: Sample name (usually the filename without extension)
            default_data: Default metadata fields to use
            img_metadata: Extracted image metadata to include

        Returns:
            Dict: Complete metadata dictionary
        """
        metadata = {
            # Start with sample_name to maintain field order
            "sample_name": base_name,
            # Then add other fields in expected order
            "sample_type": default_data.get("sample_type", ""),
            "sample_geometry": default_data.get("sample_geometry", ""),
            "sample_scale": default_data.get(
                "sample_scale", self.default_scale
            ),
            "sample_angle": default_data.get(
                "sample_angle", self.default_angle
            ),
        }

        # Include image metadata if available
        if img_metadata:
            metadata.update(img_metadata)

        return metadata

    def set_metadata(self, metadata: Dict, apply_to_all: bool = False):
        """
        Set metadata for current file and continue processing.

        Args:
            metadata: User-provided metadata dictionary
            apply_to_all: Whether to apply this metadata to all remaining files
        """
        # Get current file info
        file_path = self.all_files[self.current_file_index]
        rel_path = os.path.relpath(file_path, self.source_directory)
        dir_path = os.path.dirname(rel_path)
        base_name = metadata["sample_name"]

        # Create target metadata path
        target_dir = os.path.join(self.target_directory, dir_path)
        metadata_path = os.path.join(
            target_dir, f"{base_name}{self.metadata_file_extension}"
        )

        # Use the cached image metadata (from _process_next_file)
        img_metadata = self._current_img_metadata

        # Include image metadata in the user-provided metadata
        if img_metadata:
            metadata.update(img_metadata)

        # Save the metadata file
        self._save_metadata(metadata, metadata_path)

        # Update apply_to_all flag and store default metadata if needed
        self.apply_to_all = apply_to_all
        if apply_to_all:
            # Create a copy of metadata for future files
            self.default_metadata = {
                # Maintain field order by explicitly copying fields
                "sample_type": metadata.get("sample_type", ""),
                "sample_geometry": metadata.get("sample_geometry", ""),
                "sample_scale": metadata.get(
                    "sample_scale", self.default_scale
                ),
                "sample_angle": metadata.get(
                    "sample_angle", self.default_angle
                ),
            }

        # Clean up the temporary metadata cache
        self._current_img_metadata = None

        # Move to next file
        self.current_file_index += 1
        self._process_next_file()

    def _copy_image(
        self, source_path: str, target_path: str
    ) -> Optional[Dict]:
        """
        Copy image file without conversion, maintaining original format.

        Args:
            source_path: Path to source image file
            target_path: Path to destination image file

        Returns:
            Dict: Image metadata or None if processing failed
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

                # Extract other image info (excluding binary data)
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
        """
        Save metadata to a JSON file.

        Args:
            metadata: Metadata dictionary to save
            metadata_path: Path to save the JSON file
        """
        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
        except (OSError, TypeError) as e:
            print(f"Error saving metadata to {metadata_path}: {e}")


class PreparationWidget(Container):
    """
    Widget for preparing sample images and metadata for ROXAS analysis.

    This widget provides UI controls for selecting directories and
    initiating the image preparation process.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        """
        Initialize the preparation widget.

        Args:
            viewer: The napari viewer instance
        """
        super().__init__()
        self._viewer = viewer

        # Load settings
        self._load_settings()

        # Create UI components
        self._create_ui_components()

        # Initialize state variables
        self.source_directory = None
        self.project_directory = None
        self.worker = None
        self.worker_thread = None
        self.metadata_dialog = None
        self.same_directory = False

    def _load_settings(self):
        """Load settings from the settings manager."""
        # Initialize settings manager
        self.settings_manager = SettingsManager()

        # Get authorized sample types and geometries
        self.authorized_sample_types = self.settings_manager.get(
            "authorised_sample_types",
            ["conifer", "angiosperm"],
        )
        self.authorized_sample_geometries = self.settings_manager.get(
            "authorised_sample_geometries",
            ["linear", "circular"],
        )

        # Get default scale and angle
        self.default_scale = self.settings_manager.get(
            "default_scale",
            2.2675,  # Default value in pixels/μm
        )
        self.default_angle = self.settings_manager.get("default_angle", 0.0)

        # Get file extension settings
        scan_file_extension = self.settings_manager.get(
            "scan_file_extension", [".scan", ".jpg"]
        )
        self.scan_file_prefix = (
            scan_file_extension[0] if scan_file_extension else ".scan"
        )

        metadata_file_extension_parts = self.settings_manager.get(
            "metadata_file_extension", [".metadata", ".json"]
        )
        self.metadata_file_extension = "".join(metadata_file_extension_parts)

        # Get ROXAS file extensions to avoid processing already prepared files
        self.roxas_file_extensions = self.settings_manager.get(
            "roxas_file_extensions", []
        )

        # Get supported image file extensions
        self.image_file_extensions = self.settings_manager.get(
            "image_file_extensions",
            [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".jp2"],
        )

    def _create_ui_components(self):
        """Create and configure UI components."""
        # Source directory selector
        self._source_dialog_button = PushButton(text="Source Directory: None")
        self._source_dialog_button.changed.connect(self._open_source_dialog)

        # Project directory selector
        self._project_dialog_button = PushButton(
            text="Project Directory: None"
        )
        self._project_dialog_button.changed.connect(self._open_project_dialog)

        # Progress bar (initially hidden)
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
        if not self._validate_inputs():
            return

        # Check if source and project are the same directory
        self.same_directory = os.path.normpath(
            self.source_directory
        ) == os.path.normpath(self.project_directory)

        # Show warning for same directory mode
        if self.same_directory and not self._confirm_same_directory():
            return

        # Update UI for processing
        self._update_ui_for_processing(True)

        # Start processing in a separate thread
        self._run_in_thread(
            self.source_directory,
            self.project_directory,
            None,  # No default metadata initially
            False,  # Not applying to all initially
        )

    def _validate_inputs(self) -> bool:
        """Validate user inputs before processing."""
        if not self.source_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a source directory."
            )
            return False

        if not self.project_directory:
            QMessageBox.warning(
                None, "Warning", "Please select a project directory."
            )
            return False

        return True

    def _confirm_same_directory(self) -> bool:
        """Show confirmation dialog for same-directory mode."""
        confirm = QMessageBox()
        confirm.setIcon(QMessageBox.Warning)
        confirm.setWindowTitle("Warning: Source Modification")
        confirm.setText("The source and project directories are the same.")
        confirm.setInformativeText(
            f"Original image files will be replaced with {self.scan_file_prefix}* files. "
            "This operation cannot be undone."
        )
        confirm.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
        confirm.setDefaultButton(QMessageBox.Cancel)

        result = confirm.exec_()
        return result == QMessageBox.Ok

    def _update_ui_for_processing(self, is_processing: bool):
        """Update UI elements based on processing state."""
        self._start_button.visible = not is_processing
        self._cancel_button.visible = is_processing
        self._progress_bar.visible = is_processing

        if is_processing:
            self._progress_bar.value = 0

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
        """
        Show dialog to request metadata for a file.

        Args:
            filename: Base filename (without extension) for the current file
        """
        # Create default metadata with sample name and defaults
        default_metadata = {
            "sample_name": filename,
            "sample_type": (
                self.authorized_sample_types[0]
                if self.authorized_sample_types
                else "conifer"
            ),
            "sample_geometry": (
                self.authorized_sample_geometries[0]
                if self.authorized_sample_geometries
                else "linear"
            ),
            "sample_scale": self.default_scale,
            "sample_angle": self.default_angle,
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

    def _request_overwrite_confirmation(self, filename):
        """
        Show dialog to confirm file overwrite.

        Args:
            filename: Path of the file that would be overwritten
        """
        # Create and show the overwrite dialog
        dialog = FileOverwriteDialog(filename)
        result = dialog.exec_()

        if result:
            # User made a choice
            overwrite = dialog.should_overwrite()
            apply_to_all = dialog.apply_to_all()

            # Send the choice back to worker
            self.worker.set_overwrite_choice(overwrite, apply_to_all)
        else:
            # Dialog was canceled - stop processing
            self._cancel_processing()

    def _processing_finished(self):
        """Handle the completion of processing."""
        # Reset UI
        self._update_ui_for_processing(False)

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
        """
        Run the processing in a separate thread.

        Args:
            source_directory: Directory containing source images
            target_directory: Directory to save processed files
            default_metadata: Optional default metadata to use
            apply_to_all: Whether to apply default metadata to all files
        """
        # Create thread and worker
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
            self.image_file_extensions,
        )
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._processing_finished)
        self.worker.progress.connect(self._update_progress)
        self.worker.metadata_request.connect(self._request_metadata)
        self.worker.overwrite_request.connect(
            self._request_overwrite_confirmation
        )

        # Start the thread
        self.worker_thread.start()
