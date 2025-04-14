"""
Reader plugin for ROXAS AI-specific file formats.
"""

import json
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

# Import SettingsManager to get file extensions
from napari_roxas_ai._settings import SettingsManager

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None


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

    # Get file extensions from settings
    settings = SettingsManager()
    # Get file extension settings and join the parts
    scan_content_ext = settings.get("file_extensions.scan_file_extension")[0]
    cells_content_ext = settings.get("file_extensions.cells_file_extension")[0]
    rings_content_ext = settings.get("file_extensions.rings_file_extension")[0]

    # Check for all supported file types with a single endswith call
    return any(
        ext in path_lower
        for ext in [scan_content_ext, cells_content_ext, rings_content_ext]
    )


def get_metadata_from_json(path):
    """
    Get metadata from a JSON file.

    Parameters
    ----------
    path : str
        Path to the image file associated with the JSON metadata file

    Returns
    -------
    dict or None
        Metadata from JSON file, or None if no JSON metadata file exists
    """

    # Use the new nested structure for settings
    settings = SettingsManager()
    metadata_file_extension_parts = settings.get(
        "file_extensions.metadata_file_extension"
    )
    metadata_file_extension = "".join(metadata_file_extension_parts)

    # Check if metadata file exists
    metadata_path = "".join(path.split(".")[:-2]) + metadata_file_extension

    if not os.path.exists(metadata_path):
        print(
            f"Error: Could not find metadata file: {metadata_path} for {path}"
        )
        return None

    # Load metadata from JSON file
    try:
        with open(metadata_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        print(f"Error: Could not parse JSON metadata file: {metadata_path}")
        return None


def read_cells_file(path: str) -> Tuple[np.ndarray, dict, str]:
    """
    Read a cells file and return it as a labels layer.

    Parameters
    ----------
    path : str
        Path to the cells file

    Returns
    -------
    tuple
        (data, metadata, layer_type) for the cells image
    """
    # Open the image file
    with Image.open(path) as img:
        # Convert to numpy array
        data = np.array(img).astype(float)

    # Normalize the data to 0-1 range (except if there is a single value)
    if np.max(data) - np.min(data) > 0:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        data = np.zeros_like(data)

    # Create metadata
    filename = os.path.basename(path)
    layer_name = os.path.splitext(filename)[0]
    metadata = {"name": layer_name}

    # Try to get sample scale from metadata file
    json_metadata = get_metadata_from_json(path)
    if json_metadata and "sample_scale" in json_metadata:
        scale_value = 1 / float(json_metadata["sample_scale"])
        metadata["scale"] = [scale_value, scale_value]

    return data.astype(int), metadata, "labels"


def read_rings_file(path: str) -> Tuple[np.ndarray, dict, str]:
    """
    Read a rings file and return it as a labels layer.

    Parameters
    ----------
    path : str
        Path to the rings file

    Returns
    -------
    tuple
        (data, metadata, layer_type) for the rings image
    """
    with Image.open(path) as img:
        data = np.array(img)

    # Create metadata
    filename = os.path.basename(path)
    layer_name = os.path.splitext(filename)[0]
    metadata = {"name": layer_name}

    # Try to get sample scale from metadata file
    json_metadata = get_metadata_from_json(path)
    if json_metadata and "sample_scale" in json_metadata:
        scale_value = 1 / float(json_metadata["sample_scale"])
        metadata["scale"] = [scale_value, scale_value]

    return data, metadata, "labels"


def read_image_file(path):
    """
    Read an image file and return it as a napari layer.

    Parameters
    ----------
    path : str
        Path to the image file

    Returns
    -------
    napari.layers.Image
        Image layer for napari
    """

    # Skip non-image files
    settings = SettingsManager()
    image_file_extensions = settings.get(
        "file_extensions.image_file_extensions",
    )

    file_ext = os.path.splitext(path)[1].lower()
    if file_ext not in image_file_extensions:
        return None

    # Get metadata from associated JSON file
    json_metadata = get_metadata_from_json(path)

    with Image.open(path) as img:
        data = np.array(img)

    # Create metadata
    filename = os.path.basename(path)
    layer_name = os.path.splitext(filename)[0]
    metadata = {"name": layer_name}

    # Try to get sample scale from metadata file
    if json_metadata and "sample_scale" in json_metadata:
        scale_value = 1 / float(json_metadata["sample_scale"])
        metadata["scale"] = [scale_value, scale_value]

    return data, metadata, "image"


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

    # Get file extensions from settings
    settings = SettingsManager()
    # Get file extension settings and join the parts
    # Get file extension settings and join the parts
    scan_content_ext = settings.get("file_extensions.scan_file_extension")[0]
    cells_content_ext = settings.get("file_extensions.cells_file_extension")[0]
    rings_content_ext = settings.get("file_extensions.rings_file_extension")[0]

    # Process each path
    for path in paths:
        # Skip unsupported files
        if not is_supported_file(path):
            continue

        # Process based on file type
        if cells_content_ext in path.lower():
            layers.append(read_cells_file(path))
        elif rings_content_ext in path.lower():
            layers.append(read_rings_file(path))
        elif scan_content_ext in path.lower():
            layers.append(read_image_file(path))

    return layers


def read_directory(path: str) -> List[Tuple[Any, dict, str]]:
    """
    Read all supported files from a directory and its subdirectories.

    Parameters
    ----------
    path : str
        Path to directory

    Returns
    -------
    list of tuples
        List of (data, metadata, layer_type) tuples
    """
    # List to store all found files
    files = []

    # Walk through the directory and its subdirectories
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                files.append(file_path)

    # Use the existing read_files function to process the files
    return read_files(files)
