"""
Worker class for processing files in a separate thread.
"""

import glob
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from PIL import ExifTags, Image
from qtpy.QtCore import QObject, Signal


class Worker(QObject):
    """
    Worker for processing files in a separate thread.

    This worker handles finding images, renaming them, extracting metadata,
    and managing the processing flow.
    """

    # Signal definitions
    finished = Signal()  # Emitted when processing is complete
    progress = Signal(int, int)  # current, total - for progress reporting
    metadata_request = Signal(str)  # filename - requests metadata for a file
    abort_signal = Signal()  # Signals to abort processing

    def __init__(
        self,
        project_directory: str,
        scan_content_extension: str,
        metadata_file_extension: str,
        image_file_extensions: List[str],
        selected_files: Optional[List[str]] = None,
        process_processed: bool = False,
        overwrite_files: bool = True,
    ):
        """Initialize the worker with necessary parameters."""
        super().__init__()

        # Directories and file settings
        self.project_directory = project_directory
        self.scan_content_extension = scan_content_extension
        self.metadata_file_extension = metadata_file_extension
        self.image_file_extensions = image_file_extensions
        self.selected_files = selected_files
        self.process_processed = process_processed
        self.overwrite_files = overwrite_files

        # Processing state
        self.should_stop = False
        self.current_file_index = 0
        self.all_files = []
        self.default_metadata = None
        self.apply_to_all = False
        self._current_img_metadata = None  # Temporary cache for image metadata

    def run(self):
        """
        Process all image files from the project directory.

        This is the main entry point for the worker thread.
        """
        # Find all image files in the project directory
        all_files = []

        # Gather all files with supported image extensions
        for ext in self.image_file_extensions:
            all_files.extend(
                glob.glob(
                    str(Path(self.project_directory) / "**" / f"*{ext}"),
                    recursive=True,
                )
            )
            all_files.extend(
                glob.glob(
                    str(
                        Path(self.project_directory) / "**" / f"*{ext.upper()}"
                    ),
                    recursive=True,
                )
            )

        # Filter based on selected files if provided
        if self.selected_files and len(self.selected_files) > 0:
            # Create a set of absolute paths for fast lookup
            selected_paths = {Path(f).absolute() for f in self.selected_files}
            # Only include files that are in the selected paths
            filtered_files = [
                f for f in all_files if Path(f).absolute() in selected_paths
            ]
            all_files = filtered_files

        # Filter out already processed files if needed
        if not self.process_processed:
            filtered_files = []
            for file_path in all_files:
                basename = Path(file_path).name
                if self.scan_content_extension not in basename:
                    filtered_files.append(file_path)
            all_files = filtered_files

        # Sort files for consistent processing
        self.all_files = sorted(all_files)

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
        dir_path = Path(file_path).parent
        file_name = Path(file_path).name
        base_name = Path(file_name).stem
        file_ext = Path(file_name).suffix

        # Create Path object for advanced path manipulation
        file_path_obj = Path(file_path)

        # Get the parent directory and pure stem (removing all extensions)
        parent_dir = file_path_obj.parent
        stem_name = file_path_obj.stem
        # Remove all suffixes to get the pure stem name if we are processing scan files again

        if stem_name.endswith(self.scan_content_extension):
            stem_name = Path(stem_name).stem

        # Create the sample_stem_path by joining parent and stem
        sample_stem_path = str(parent_dir / stem_name)

        # Check if base_name already has the scan extension and remove it to avoid duplication
        original_has_scan_ext = False
        if self.scan_content_extension in base_name:
            # This is an already processed file, extract the clean base name
            original_has_scan_ext = True
            clean_base_name = base_name.replace(
                self.scan_content_extension, ""
            )
            new_image_path = file_path  # Keep the original path
            print(f"Reprocessing already processed file: {file_path}")
            print(f"Clean base name: {clean_base_name}")
        else:
            # This is a new file, add the scan extension
            clean_base_name = base_name
            new_image_path = (
                dir_path
                / f"{clean_base_name}{self.scan_content_extension}{file_ext}"
            )
            # Rename or copy file if needed
            if file_path != str(new_image_path):
                try:
                    # Extract metadata before modifying the file
                    img_metadata = self._extract_image_metadata(file_path)

                    if self.overwrite_files:
                        # If overwrite is enabled, rename the file (replace original)
                        if Path(new_image_path).exists():
                            Path(
                                new_image_path
                            ).unlink()  # Remove destination if it exists
                        shutil.move(file_path, new_image_path)
                        print(f"Renamed file: {file_path} -> {new_image_path}")
                    else:
                        # Otherwise, just copy it (keep original)
                        shutil.copy2(file_path, new_image_path)
                        print(f"Copied file: {file_path} -> {new_image_path}")

                    # Update file path to the new location
                    file_path = str(new_image_path)
                except OSError as e:
                    print(f"Error processing file {file_path}: {e}")
                    # Skip to next file
                    self.current_file_index += 1
                    self._process_next_file()
                    return

        # Define metadata path - always use the clean base name without scan extension
        metadata_path = (
            dir_path / f"{clean_base_name}{self.metadata_file_extension}"
        )

        # Extract image metadata if we haven't already
        if "img_metadata" not in locals():
            img_metadata = self._extract_image_metadata(file_path)

        # Add sample_stem_path to the image metadata
        img_metadata["sample_stem_path"] = sample_stem_path

        # Cache metadata for use in set_metadata
        self._current_img_metadata = img_metadata

        # If we have default metadata and apply_to_all is true, use it
        if self.apply_to_all and self.default_metadata:
            # Use the cached metadata directly - no need to modify it
            # Just make a copy to ensure we don't modify the original
            metadata = self.default_metadata.copy()

            # Only update the sample name to match this file
            metadata["sample_name"] = clean_base_name

            # Add sample_stem_path to this file's metadata
            metadata["sample_stem_path"] = sample_stem_path

            # Add image metadata
            self._add_image_metadata_to_file_metadata(metadata, img_metadata)

            # Save metadata file
            self._save_metadata(metadata, metadata_path)

            # Move to next file
            self.current_file_index += 1
            self._process_next_file()
        else:
            # Request metadata from user via dialog - pass the clean base name
            self.metadata_request.emit(clean_base_name)
            # Processing will continue when metadata is provided via set_metadata

    def set_metadata(self, metadata: Dict, apply_to_all: bool = False):
        """
        Set metadata for current file and continue processing.

        Args:
            metadata: User-provided metadata dictionary
            apply_to_all: Whether to apply this metadata to all remaining files
        """
        # Get current file info
        file_path = self.all_files[self.current_file_index]
        dir_path = Path(file_path).parent
        file_name = Path(file_path).name
        base_name = Path(file_name).stem

        # Extract clean base name (without scan extension) for the metadata file path
        if self.scan_content_extension in base_name:
            clean_base_name = base_name.replace(
                self.scan_content_extension, ""
            )
        else:
            clean_base_name = base_name

        # Create metadata path using clean base name
        metadata_path = (
            dir_path / f"{clean_base_name}{self.metadata_file_extension}"
        )

        # Use the cached image metadata
        img_metadata = self._current_img_metadata

        # Add image metadata to the file metadata
        self._add_image_metadata_to_file_metadata(metadata, img_metadata)

        # Save the metadata file
        self._save_metadata(metadata, metadata_path)

        # Update apply_to_all flag and store default metadata if needed
        self.apply_to_all = apply_to_all
        if apply_to_all:
            # Store the complete metadata for future files
            self.default_metadata = metadata.copy()

        # Clean up the temporary metadata cache
        self._current_img_metadata = None

        # Move to next file
        self.current_file_index += 1
        self._process_next_file()

    def _add_image_metadata_to_file_metadata(
        self, metadata: Dict, img_metadata: Dict
    ):
        """
        Add image metadata to the file metadata.

        Args:
            metadata: File metadata dictionary to update
            img_metadata: Image metadata to add
        """
        if img_metadata:
            # Add image metadata fields without overwriting existing metadata
            # This preserves the sample_ fields from the dialog
            for key, value in img_metadata.items():
                if key not in metadata:
                    metadata[key] = value

    def _extract_image_metadata(self, image_path: str) -> Dict:
        """
        Extract metadata from an image file.

        Args:
            image_path: Path to the image file

        Returns:
            Dict: Image metadata
        """
        try:
            with Image.open(image_path) as img:
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

                return img_metadata

        except (OSError, ValueError, Image.UnidentifiedImageError) as e:
            print(f"Error extracting metadata from {image_path}: {e}")
            return {}  # Return empty dict on error

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

    def _sanitize_for_json(self, obj):
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
