import errno
import glob
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import ExifTags, Image

from napari_roxas_ai._jobs import Progress

from .._edition import update_rings_geometries
from .._settings._settings_manager import SettingsManager
from .._utils._metadata_keys import (
    NO_REFERENCE_SERIES,
    sanitize_metadata_floats,
)
from .._utils._ringtraces_utils import (
    get_instance_labels_linear,
    ring_labels_from_roxas,
)
from .._utils._scl_utils import read_scl
from .._utils._segmentation_postprocess import (
    remove_border_touching_components,
)
from .._writer import write_single_layer

# Output files of ROXAS Classic that may sit next to the input images
ROXAS_CLASSIC_OUTPUT_SUFFIXES = (
    "_annotated.jpg",
    "_annotated_cells.jpg",
    "_annotated_twin.jpg",
    "_ReferenceSeries.gif",
    "_ReferenceSeries.jpg",
    "_ReferenceSeriesLong.jpg",
    "_Preview.jpg",
)

# The answer of a metadata dialog: (metadata, apply to all, loading params)
Answer = tuple[dict, bool, dict]


def is_roxas_ai_output_file(file_path: str) -> bool:
    """
    Tell whether a file is an output of ROXAS AI (cells/rings rasters, the
    annotated rings image or the exported crossdating plot) rather than an
    input image to prepare.

    Parameters
    ----------
    file_path : str
        Path of the file to check.

    Returns
    -------
    bool
        True if the file was written by ROXAS AI.
    """
    settings = SettingsManager()
    output_suffixes = (
        "".join(settings.get("file_extensions.cells_file_extension")),
        "".join(settings.get("file_extensions.rings_file_extension")),
        "_annotated.jpg",
        "_ReferenceSeries.jpg",
    )
    name = Path(file_path).name.lower()
    return name.endswith(tuple(suffix.lower() for suffix in output_suffixes))


def sample_base_name(file_path: str, scan_content_extension: str) -> str:
    """The sample name of an image file: its name without the extensions."""
    base_name = Path(file_path).stem
    if scan_content_extension in base_name:
        return base_name.replace(scan_content_extension, "")
    return base_name


def discover_files(
    project_directory: str,
    image_file_extensions: list[str],
    scan_content_extension: str,
    selected_files: Optional[list[str]] = None,
    process_processed: bool = False,
):
    """The image files of the project to prepare, sorted."""
    yield Progress(0, 0, "Finding the image files…")
    all_files = []

    # Gather all files with supported image extensions
    for ext in image_file_extensions:
        all_files.extend(
            glob.glob(
                str(Path(project_directory) / "**" / f"*{ext}"),
                recursive=True,
            )
        )
        all_files.extend(
            glob.glob(
                str(Path(project_directory) / "**" / f"*{ext.upper()}"),
                recursive=True,
            )
        )

    # ROXAS AI outputs and ROXAS Classic outputs are never inputs
    all_files = [
        f
        for f in all_files
        if not is_roxas_ai_output_file(f)
        and not f.endswith(ROXAS_CLASSIC_OUTPUT_SUFFIXES)
    ]

    # Filter based on selected files if provided
    if selected_files and len(selected_files) > 0:
        # Create a set of absolute paths for fast lookup
        selected_paths = {Path(f).absolute() for f in selected_files}
        # Only include files that are in the selected paths
        all_files = [
            f for f in all_files if Path(f).absolute() in selected_paths
        ]

    # Filter out already processed files if needed
    if not process_processed:
        all_files = [
            file_path
            for file_path in all_files
            if scan_content_extension not in Path(file_path).name
        ]

    # Sort files for consistent processing and remove duplicates (e.g. on
    # case-insensitive filesystems)
    return sorted(set(all_files))


def prepare_files(
    files: list[str],
    answers: list[Answer],
    project_directory: str,
    scan_content_extension: str,
    metadata_file_extension: str,
    cells_extension: list[str],
    rings_extension: list[str],
):
    """
    Prepare the image files, in order, with the answers of the metadata
    dialogs: one per file, until one applies to all the remaining files.

    Without such an answer only the answered files are prepared (the dialogs
    were cancelled). Cancelling stops between two files.

    Returns whether all the files were prepared.
    """
    total = len(files)
    count = total if answers and answers[-1][1] else len(answers)
    apply_to_all = False
    default_metadata = default_loading_params = None
    index = 0

    try:
        for index in range(count):
            file_path = files[index]
            file_name = Path(file_path).name
            yield Progress(
                index, total, f"Preparing {file_name} ({index + 1}/{total})"
            )

            dir_path = Path(file_path).parent
            base_name = Path(file_name).stem
            file_ext = Path(file_name).suffix
            sample_stem_path = _sample_stem_path(
                file_path, scan_content_extension, project_directory
            )

            # Check if base_name already has the scan extension and remove it
            # to avoid duplication
            original_has_scan_ext = scan_content_extension in base_name
            clean_base_name = sample_base_name(
                file_path, scan_content_extension
            )
            if original_has_scan_ext:
                # This is an already processed file
                print(f"Reprocessing already processed file: {file_path}")
                print(f"Clean base name: {clean_base_name}")
                new_image_path = file_path  # Keep the original path
            else:
                # This is a new file, add the scan extension
                new_image_path = (
                    dir_path
                    / f"{clean_base_name}{scan_content_extension}{file_ext}"
                )

            # Metadata path - always the clean base name without scan extension
            metadata_path = (
                dir_path / f"{clean_base_name}{metadata_file_extension}"
            )

            # Extract image metadata BEFORE renaming
            img_metadata = extract_image_metadata(file_path)
            img_metadata["sample_stem_path"] = sample_stem_path

            if apply_to_all and default_metadata:
                # The metadata of the file answered for all, for this file
                metadata = default_metadata.copy()
                metadata["sample_name"] = clean_base_name
                metadata["sample_stem_path"] = sample_stem_path
                loading_params = default_loading_params
                add_image_metadata(metadata, img_metadata)
                save_metadata(metadata, metadata_path)
            else:
                metadata, apply_to_all, loading_params = answers[index]
                add_image_metadata(metadata, img_metadata)
                save_metadata(metadata, metadata_path)
                if apply_to_all:
                    # The complete metadata, for the remaining files
                    default_metadata = metadata.copy()
                    default_loading_params = loading_params.copy()

            tasks = ["Metadata"]

            # Load data from old roxas file formats if needed
            if loading_params.get("load_cells_from_roxas", False):
                tasks.append(
                    _cells_task(
                        load_cells_from_roxas_scl(
                            file_path, metadata, cells_extension
                        )
                    )
                )
            if loading_params.get("load_rings_from_roxas", False):
                tasks.append(
                    _rings_task(
                        load_rings_from_roxas_ringtraces(
                            file_path, metadata, rings_extension
                        )
                    )
                )

            # Rename or copy file if needed AFTER metadata and data extraction
            if not original_has_scan_ext and file_path != str(new_image_path):
                tasks.append(
                    _rename_or_copy(file_path, new_image_path, loading_params)
                )

            print(
                f"[Preparation] [{index + 1}/{total}] "
                f"{file_name}: {', '.join(tasks)}"
            )
    except GeneratorExit:
        # Cancelled before the file at index
        print(f"[Preparation] Stopped after {index} of {total} file(s).")
        raise

    if count == total:
        print(f"[Preparation] Completed {total} file(s).")
    else:
        print(f"[Preparation] Stopped after {count} of {total} file(s).")
    return count == total


def _sample_stem_path(
    file_path: str, scan_content_extension: str, project_directory: str
) -> str:
    """
    The sample stem (directory / name without extensions), relative to the
    project root if the file is in it.
    """
    file_path_obj = Path(file_path)
    parent_dir = file_path_obj.parent
    stem_name = file_path_obj.stem
    # Remove all suffixes to get the pure stem name if we are processing scan
    # files again
    if stem_name.endswith(scan_content_extension):
        stem_name = Path(stem_name).stem

    sample_stem_abs = (parent_dir / stem_name).resolve()
    project_root = Path(project_directory).resolve()
    try:
        # Stored with portable separators
        return sample_stem_abs.relative_to(project_root).as_posix()
    except ValueError:
        # If file isn't under project root, store as POSIX absolute
        return sample_stem_abs.as_posix()


def _rename_or_copy(
    file_path: str, new_image_path, loading_params: dict
) -> str:
    """Rename (or copy for ROXAS Classic imports) to the scan file name."""
    try:
        is_roxas_import = bool(
            loading_params
            and (
                loading_params.get("load_cells_from_roxas", False)
                or loading_params.get("load_rings_from_roxas", False)
            )
        )
        if is_roxas_import:
            # Keep original ROXAS Classic file set unchanged
            Path(new_image_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, new_image_path)
            print(f"Copied file: {file_path} -> {new_image_path}")
            return "Copied"
        # Always avoid duplicates: rename if possible, else copy+delete
        moved_path = safe_rename_or_copy_delete(
            Path(file_path), Path(new_image_path)
        )
        print(f"Renamed file: {file_path} -> {moved_path}")
        return "Renamed"
    except OSError as e:
        print(f"Error processing file {file_path}: {e}")
        return "Rename failed"


def _cells_task(loaded: bool) -> str:
    """Log text for loading cells from ROXAS Classic."""
    return "Cells (ROXAS SCL)" if loaded else "Cells skipped (no ROXAS SCL)"


def _rings_task(loaded: bool) -> str:
    """Log text for loading rings from ROXAS Classic."""
    return (
        "Rings (ROXAS RingTraces)"
        if loaded
        else "Rings skipped (no ROXAS RingTraces)"
    )


def safe_rename_or_copy_delete(src: Path, dst: Path) -> Path:
    """
    Rename/move a file without leaving duplicates.
    Uses atomic replace when possible; falls back to copy+delete on
    cross-device moves.
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source does not exist: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # If destination exists, remove it to allow replace semantics
    if dst.exists():
        dst.unlink()

    # Handle case-only rename on case-insensitive filesystems
    # (Windows/macOS default)
    try:
        if (
            src.resolve().as_posix().lower()
            == dst.resolve().as_posix().lower()
            and src.name != dst.name
        ):
            tmp = dst.with_name(dst.name + ".__tmp__")
            os.replace(src, tmp)
            os.replace(tmp, dst)
            return dst
    except OSError:
        # If resolve fails, ignore and proceed with normal path
        pass

    try:
        os.replace(src, dst)  # atomic on same filesystem
        return dst
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        # Cross-device rename not possible: copy+delete (still no duplicates
        # at end)
        shutil.copy2(src, dst)
        src.unlink()
        return dst


def load_cells_from_roxas_scl(
    file_path: str, metadata: dict, cells_extension: list[str]
) -> bool:
    """
    Load cells from a .scl file generated by ROXAS and write them as the
    cells file of the image. Returns whether the cells were loaded.
    """
    cells_content_ext, cells_ext = cells_extension[0], cells_extension[1]
    print(f"Attempting to load cells from ROXAS SCL for {file_path}")
    try:
        if os.path.exists(
            scl_file_name := file_path.replace(
                Path(file_path).suffix, "_Vessels.scl"
            )
        ):
            loaded_cells = read_scl(scl_file_name)
            cells_binary = (loaded_cells > 0).astype("uint8")
            cells_binary = remove_border_touching_components(cells_binary)
            cells_data = (cells_binary * 255).astype("uint8")
            cells_layer_name = f"{metadata['sample_name']}{cells_content_ext}"
            # Ensure spatial_resolution is a float and present in metadata.
            # We harmonize spatial_resolution and sample_scale keys for
            # cross-plotter compatibility.
            spatial_resolution = float(
                metadata.get("spatial_resolution")
                or metadata.get("sample_scale", 1.0)
            )
            metadata["spatial_resolution"] = spatial_resolution
            metadata["sample_scale"] = spatial_resolution

            cells_add_kwargs = {
                "name": cells_layer_name,
                # Scale factor is 1/resolution (e.g. pixels to micrometers)
                "scale": [1 / spatial_resolution, 1 / spatial_resolution],
                "features": pd.DataFrame(),
                "metadata": {
                    **metadata,
                    "cells_segmentation_model": "ROXAS SCL",
                    "cells_segmentation_datetime": datetime.now().isoformat(),
                },
            }

            cell_path = (
                Path(file_path).parent
                / f"{Path(file_path).stem}{cells_content_ext}{cells_ext}"
            )
            write_single_layer(
                path=cell_path, data=cells_data, meta=cells_add_kwargs
            )
            print(f"Successfully loaded cells from ROXAS SCL for {file_path}")
            return True
        else:
            raise FileNotFoundError(
                f"SCL file not found for {file_path}: expected at "
                f"{scl_file_name}"
            )
    except FileNotFoundError as e:
        print(f"Error loading cells from {file_path}: {e}")
        print("Continue without loading cells for this file.")
        return False


def load_rings_from_roxas_ringtraces(
    file_path: str, metadata: dict, rings_extension: list[str]
) -> bool:
    """
    Load rings from a .ringtraces file generated by ROXAS and write them as
    the rings file of the image. Returns whether the rings were loaded.
    """
    rings_content_ext, rings_ext = rings_extension[0], rings_extension[1]
    print(f"Attempting to load rings from ROXAS RingTraces for {file_path}")
    try:
        rings_boundaries = ring_labels_from_roxas(
            file_path,
            image_shape=metadata["scan_size"][::-1],
            return_img_coordinates=True,
        )
        rings_labels = get_instance_labels_linear(
            rings_boundaries,
            image_shape=metadata["scan_size"][::-1],
            draw_lines=False,
        )
        boundary_data = []
        for boundary in rings_boundaries:
            # flip x and y to get (row, col) format
            coords = np.flip(boundary.T, axis=1)
            boundary_data.append({"RBXY": coords})

        boundaries_df = pd.DataFrame(boundary_data)

        rings_layer_name = f"{metadata['sample_name']}{rings_content_ext}"

        # Get ring years from the ROXAS output file and determine the last
        # complete year based on the maximum year found in the file.
        last_year = max(
            pd.read_csv(
                file_path.replace(Path(file_path).suffix, "_Output_Rings.txt"),
                sep="\t",
            )["YEAR"]
        )

        new_rings_table, rings_raster, _cmap = update_rings_geometries(
            rings_table=boundaries_df,
            last_year=int(last_year),
            image_shape=rings_labels.shape,
        )
        # Ensure spatial_resolution is a float and present in metadata.
        # We harmonize spatial_resolution and sample_scale keys for
        # cross-plotter compatibility.
        spatial_resolution = float(
            metadata.get("spatial_resolution")
            or metadata.get("sample_scale", 1.0)
        )
        metadata["spatial_resolution"] = spatial_resolution
        metadata["sample_scale"] = spatial_resolution

        rings_add_kwargs = {
            "name": rings_layer_name,
            # Scale factor is 1/resolution (e.g. pixels to micrometers)
            "scale": [1 / spatial_resolution, 1 / spatial_resolution],
            "features": new_rings_table,
            "metadata": {
                **metadata,
                "rings_outmost_complete_year": int(last_year),
                "rings_segmentation_model": "ROXAS RingTraces",
                "rings_segmentation_datetime": datetime.now().isoformat(),
            },
        }
        ring_path = (
            Path(file_path).parent
            / f"{Path(file_path).stem}{rings_content_ext}{rings_ext}"
        )
        write_single_layer(
            path=ring_path, data=rings_raster, meta=rings_add_kwargs
        )
        print(
            f"Successfully loaded rings from ROXAS RingTraces for {file_path}"
        )
        return True

    except FileNotFoundError as e:
        print(f"Error loading rings from {file_path}: {e}")
        print("Continue without loading rings for this file.")
        return False


def add_image_metadata(metadata: dict, img_metadata: dict) -> None:
    """Add the image metadata fields to the file metadata."""
    if img_metadata:
        for key, value in img_metadata.items():
            metadata[key] = value


def extract_image_metadata(image_path: str) -> dict:
    """The metadata of an image file (format, size, EXIF, ...)."""
    try:
        with Image.open(image_path) as img:
            # Store scan image information for metadata.
            # img_size is the size of the image file on disk, formatted with
            # its unit (MB = 10^6 bytes). Preparation renames/copies the
            # scan without re-encoding it, so this holds for the resulting
            # scan file too.
            img_metadata = {
                "scan_format": img.format,
                "scan_size": [img.width, img.height],
                "scan_mode": img.mode,
                "img_size": (
                    f"{Path(image_path).stat().st_size / 1_000_000:.2f} MB"
                ),
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
                        value, str | int | float | bool | list | tuple | dict
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
                        value, str | int | float | bool | tuple | list
                    )
                ):
                    scan_info[key] = value

            img_metadata["scan_info"] = scan_info

            return img_metadata

    except (OSError, ValueError, Image.UnidentifiedImageError) as e:
        print(f"Error extracting metadata from {image_path}: {e}")
        return {}  # Return empty dict on error


def save_metadata(metadata: dict, metadata_path) -> None:
    """Save metadata to a JSON file."""
    try:
        # A sample has no reference series until one is picked in the
        # crossdating widget, but the key is written from the start so that
        # every prepared sample carries it
        metadata.setdefault("reference_series", NO_REFERENCE_SERIES)

        # Ensure all data is JSON serializable
        sanitized_metadata = _sanitize_for_json(metadata)
        sanitized_metadata = sanitize_metadata_floats(sanitized_metadata)

        with open(metadata_path, "w") as f:
            json.dump(sanitized_metadata, f, indent=4)
    except (OSError, TypeError) as e:
        print(f"Error saving metadata to {metadata_path}: {e}")


def _sanitize_for_json(obj):
    """Recursively convert an object to a JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except UnicodeDecodeError:
            return f"0x{obj.hex()}"
    elif isinstance(obj, str | int | float | bool | type(None)):
        return obj
    else:
        # Convert other types to string
        return str(obj)
