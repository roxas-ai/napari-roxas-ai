"""
This module provides functionality to check for the existence of model files and to rebuild them from their parts.
Model files can be cut into parts for easier storage and transfer.
Thy are typicalley broken down into <50 MB chunks, with an instruction of type 'split -b 45M model.pth model_parts/part_'
"""

import glob
from pathlib import Path
from typing import Union


def ensure_model_file_exists(dir_path: Union[str, Path]) -> bool:
    """
    Check if a model file exists at the specified path, and if not,
    attempt to rebuild it from its parts.

    Parameters
    ----------
    dir_path : Union[str, Path]
        Path to the model file without extension.

    Returns
    -------
    bool
        True if the model file exists or was successfully rebuilt,
        False otherwise.
    """
    dir_path = Path(dir_path)

    # Check if model file already exists
    if dir_path.with_suffix(".pth").exists():
        return True

    # Check if parts directory exists
    parts_dir = Path(f"{dir_path}_parts")
    if not parts_dir.is_dir():
        return False

    # Get all part files and sort them naturally
    # This handles both numeric (part_1, part_2) and alphabetic (part_aa, part_ab) suffixes
    part_files = sorted(parts_dir.glob("part_*"))

    if not part_files:
        return False

    # Create the model file by concatenating all parts
    with open(dir_path.with_suffix(".pth"), "wb") as outfile:
        for part_file in part_files:
            with open(part_file, "rb") as infile:
                outfile.write(infile.read())

    return True


def scan_and_build_models(base_dir: Union[str, Path]) -> dict:
    """
    Scan the base directory and all subdirectories for potential model files
    that need to be rebuilt.

    Parameters
    ----------
    base_dir : Union[str, Path]
        Base directory to start scanning from.

    Returns
    -------
    dict
        Dictionary mapping model paths to rebuild status (True/False).
    """
    base_dir = Path(base_dir)
    results = {}

    for parts_dir in glob.glob(
        str(base_dir / "**" / "*_parts"), recursive=True
    ):
        model_path = parts_dir[:-6]  # Remove "_parts" suffix
        results[model_path] = ensure_model_file_exists(model_path)

    return results
