"""
Reader plugin for ROXAS AI-specific file formats.
"""

import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def napari_get_reader(path: Union[str, List[str]]) -> Optional[Callable]:
    """
    Return a reader function if the path is recognized by this reader plugin.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    # Handle the case where a single path is provided as a string
    if isinstance(path, str):
        # Check if the path is a directory
        if os.path.isdir(path):
            return read_directory

        # Check if the path is a file that can be read
        if is_supported_file(path):
            return read_files

    # Handle the case where a list of paths is provided
    elif isinstance(path, list):
        # Check if any file in the list is one we can read
        if any(is_supported_file(p) for p in path if isinstance(p, str)):
            return read_files

    # If we can't read the file, return None
    return None


def is_supported_file(path: str) -> bool:
    """
    Check if a file is supported by this reader.

    Parameters
    ----------
    path : str
        Path to the file

    Returns
    -------
    bool
        True if the file is supported, False otherwise
    """
    path_lower = path.lower()

    # Check for all supported file types with a single endswith call
    return path_lower.endswith(
        (".cells.png", ".jpg", ".jpeg", ".tif", ".tiff")
    )


def read_files(paths: Union[str, List[str]]) -> List[Tuple[Any, dict, str]]:
    """
    Read supported files and return layer data.

    Parameters
    ----------
    paths : str or list of str
        Path(s) to file(s)

    Returns
    -------
    list of tuples
        List of (data, metadata, layer_type) tuples
    """
    # Ensure paths is a list
    if isinstance(paths, str):
        paths = [paths]

    # Initialize return list
    layers = []

    # Process each path
    for path in paths:
        # Skip unsupported files
        if not is_supported_file(path):
            continue

        # Process based on file type
        if path.lower().endswith(".cells.png"):
            # Read cells image as labels
            with Image.open(path) as img:
                # Convert to numpy array and rescale to 0-1
                data = np.array(img).astype(float) / 255

            # Create metadata and add to layers
            filename = os.path.basename(path)
            layer_name = os.path.splitext(filename)[0]
            metadata = {"name": layer_name}
            layers.append((data.astype(int), metadata, "labels"))

        elif path.lower().endswith((".tif", ".tiff")):
            # Read TIFF file directly as labels
            with Image.open(path) as img:
                data = np.array(img)

            # Create metadata and add to layers
            filename = os.path.basename(path)
            layer_name = os.path.splitext(filename)[0]
            metadata = {"name": layer_name}
            layers.append((data, metadata, "labels"))

        elif path.lower().endswith((".jpg", ".jpeg")):
            # Read regular image file
            with Image.open(path) as img:
                data = np.array(img)

            # Create metadata and add to layers
            filename = os.path.basename(path)
            layer_name = os.path.splitext(filename)[0]
            metadata = {"name": layer_name}
            layers.append((data, metadata, "image"))

    return layers


def read_directory(path: str) -> List[Tuple[Any, dict, str]]:
    """
    Read all supported files from a directory.

    Parameters
    ----------
    path : str
        Path to directory

    Returns
    -------
    list of tuples
        List of (data, metadata, layer_type) tuples
    """
    # Get all files in the directory
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]

    # Use the existing read_files function to process the files
    return read_files(files)
