import glob
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from PIL import ExifTags, Image
from qtpy.QtCore import QObject, Signal


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
        default_scale: float = None,
        default_angle: float = None,
        default_outmost_year: Optional[int] = None,
        same_directory: bool = False,
        scan_content_extension: str = None,
        metadata_file_extension: str = None,
        roxas_file_extensions: List[str] = None,
        image_file_extensions: List[str] = None,
        selected_files: Optional[List[str]] = None,
    ):
        """Initialize the worker with necessary parameters."""
        super().__init__()
        # Directories
        self.source_directory = source_directory
        self.target_directory = target_directory

        # Metadata and processing settings
        self.default_metadata = default_metadata
        self.apply_to_all = apply_to_all
        self.authorized_sample_types = authorized_sample_types or []
        self.default_scale = default_scale
        self.default_angle = default_angle
        self.default_outmost_year = default_outmost_year

        # File handling settings
        self.same_directory = same_directory
        self.scan_content_extension = scan_content_extension
        self.metadata_file_extension = metadata_file_extension
        self.roxas_file_extensions = roxas_file_extensions or []
        self.image_file_extensions = image_file_extensions or []

        # File selection
        self.selected_files = selected_files

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
        all_source_files = []

        # Gather all files with supported image extensions
        for ext in self.image_file_extensions:
            all_source_files.extend(
                glob.glob(f"{self.source_directory}/**/*{ext}", recursive=True)
            )
            all_source_files.extend(
                glob.glob(
                    f"{self.source_directory}/**/*{ext.upper()}",
                    recursive=True,
                )
            )

        # Filter based on selected files if provided
        if self.selected_files and len(self.selected_files) > 0:
            # Create a set of absolute paths for fast lookup
            selected_paths = {os.path.abspath(f) for f in self.selected_files}
            # Only include files that are in the selected paths
            filtered_files = [
                f
                for f in all_source_files
                if os.path.abspath(f) in selected_paths
            ]
            all_source_files = filtered_files

        # Filter out files that already have ROXAS extensions
        if self.roxas_file_extensions:
            filtered_files = []
            for file_path in all_source_files:
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
        else:
            self.all_files = all_source_files

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
            target_dir,
            f"{base_name}{self.scan_content_extension}{file_ext}",
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
            target_dir,
            f"{base_name}{self.scan_content_extension}{file_ext}",
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
        # Extract metadata content extension (without the file extension)
        metadata_content_extension = self.metadata_file_extension.split(
            ".json"
        )[0]

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
            "sample_outmost_year": default_data.get(
                "sample_outmost_year", self.default_outmost_year
            ),
            # Add the sample_files field with content extensions
            "sample_files": [
                self.scan_content_extension,
                metadata_content_extension,
            ],
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
                "sample_outmost_year": metadata.get(
                    "sample_outmost_year", self.default_outmost_year
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
                    exif_dict = {}
                    for tag_id, value in img._getexif().items():
                        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                        # Convert bytes to string or skip if not serializable
                        if isinstance(value, bytes):
                            try:
                                # Try to decode bytes as UTF-8
                                exif_dict[tag_name] = value.decode(
                                    "utf-8", errors="replace"
                                )
                            except (UnicodeDecodeError, AttributeError):
                                # If decoding fails, convert to hex string
                                exif_dict[tag_name] = f"0x{value.hex()}"
                        elif isinstance(
                            value, (str, int, float, bool, list, tuple, dict)
                        ):
                            exif_dict[tag_name] = value
                        else:
                            # Skip non-serializable types
                            exif_dict[tag_name] = str(value)

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
            # Ensure all data is JSON serializable
            sanitized_metadata = self._sanitize_for_json(metadata)

            with open(metadata_path, "w") as f:
                json.dump(sanitized_metadata, f, indent=4)
        except (OSError, TypeError) as e:
            print(f"Error saving metadata to {metadata_path}: {e}")

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        Recursively convert an object to a JSON-serializable format.

        Args:
            obj: Object to sanitize

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, bytes):
            try:
                return obj.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                return f"0x{obj.hex()}"
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Convert other types to string
            return str(obj)

    def stop(self):
        """Stop the processing."""
        self.should_stop = True
