"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from PIL import Image

# Import SettingsManager to get file extensions
from napari_roxas_ai._settings import SettingsManager

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]

# Get file extensions from settings
settings = SettingsManager()


def update_metadata_file(path: str, metadata: dict, keys_prefix: str) -> str:
    """
    Update the metadata file with new metadata.
    If the metadata file does not exist, it will be created.
    If it exists, the new metadata will be added to the existing metadata.
    Parameters
    ----------
    path : str
        The path to the metadata file.
    metadata : dict
        The metadata to be added to the file.
    keys_prefix : str
        The prefix of the keys to be included in the metadata file.
    Returns
    -------
    str
        The path to the updated metadata file.
    """

    if os.path.exists(path):
        # Load existing metadata from the file
        with open(path) as f:
            existing_metadata = json.load(f)

    else:
        # Create a new metadata file if it doesn't exist
        existing_metadata = {
            k: v for k, v in metadata.items() if k.startswith("sample_")
        }

    # Filter metadata to only include keys with the specified prefix
    filtered_metadata = {
        k: v for k, v in metadata.items() if k.startswith(keys_prefix)
    }

    # Update the existing metadata with the new filtered metadata
    existing_metadata.update(filtered_metadata)

    # Write the updated metadata to the file
    with open(path, "w") as f:
        json.dump(existing_metadata, f, indent=4)

    return path


def save_image(path: str, image: np.ndarray, rescale: bool = False) -> str:
    """
    Save the image to a file.

    Parameters
    ----------
    path : str
        The path to save the image to.
    image : np.ndarray
        The image to save.

    Returns
    -------
    str
        The path to the saved image.
    """
    # Rescale the image to 0-255 uint8 if required
    if rescale:
        if np.max:
            # Normalize the image to the range [0, 255]
            image = (
                (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            ).astype(np.uint8)
        else:  # If the image has a single value, set it to zero
            image = np.zeros_like(image, dtype=np.uint8)

    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(image)

    # Save the image to the specified path
    pil_image.save(path)

    return path


def write_scan_file(path: str, data: Any, meta: dict) -> str:
    """Writes a scan file.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0]
    sample_path = os.path.join(dirname, basename)

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "scan_")

    # Save the image data
    scan_file_extension = "".join(
        settings.get("file_extensions.scan_file_extension")
    )
    scan_file_path = f"{sample_path}{scan_file_extension}"
    save_image(scan_file_path, data, rescale=False)

    # return path to any file(s) that were successfully written
    return [scan_file_path, metadata_file_path]


def write_cells_file(path: str, data: Any, meta: dict) -> str:
    """Writes a cells file.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """

    dirname = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0]
    sample_path = os.path.join(dirname, basename)

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "cells_")

    # Save the tabular data
    if not meta["features"].empty:
        cells_tablefile_extension = "".join(
            settings.get("file_extensions.cells_table_file_extension")
        )
        cells_tablefile_path = f"{sample_path}{cells_tablefile_extension}"
        meta["features"].to_csv(
            cells_tablefile_path,
            sep=settings.get("tables.separator"),
            index_label=settings.get("tables.index_column"),
        )

    # Save the image data
    cells_file_extension = "".join(
        settings.get("file_extensions.cells_file_extension")
    )
    cells_file_path = f"{sample_path}{cells_file_extension}"
    save_image(cells_file_path, data, rescale=True)

    # return path to any file(s) that were successfully written
    return [cells_file_path, metadata_file_path]


def write_rings_file(path: str, data: Any, meta: dict) -> str:
    """Writes a rings file.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """
    dirname = os.path.dirname(path)
    basename = os.path.basename(path).split(".")[0]
    sample_path = os.path.join(dirname, basename)

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "rings_")

    # Save the tabular data
    if not meta["features"].empty:
        rings_tablefile_extension = "".join(
            settings.get("file_extensions.rings_table_file_extension")
        )
        rings_tablefile_path = f"{sample_path}{rings_tablefile_extension}"
        meta["features"].to_csv(
            rings_tablefile_path,
            sep=settings.get("tables.separator"),
            index_label=settings.get("tables.index_column"),
        )

    # Save the image data
    rings_file_extension = "".join(
        settings.get("file_extensions.rings_file_extension")
    )
    rings_file_path = f"{sample_path}{rings_file_extension}"
    save_image(rings_file_path, data, rescale=True)

    # return path to any file(s) that were successfully written
    return [rings_file_path, metadata_file_path]


def write_single_layer(path: str, data: Any, meta: dict) -> list[str]:
    """Writes a single image layer.

    Parameters
    ----------
    path : str
        A string path indicating where to save the image file.
    data : The layer data
        The `.data` attribute from the napari layer.
    meta : dict
        A dictionary containing all other attributes from the napari layer
        (excluding the `.data` layer attribute).

    Returns
    -------
    [path] : A list containing the string path to the saved file.
    """

    layer_name = meta["name"]

    # Get file extension settings and join the parts
    scan_content_ext = settings.get("file_extensions.scan_file_extension")[0]
    cells_content_ext = settings.get("file_extensions.cells_file_extension")[0]
    rings_content_ext = settings.get("file_extensions.rings_file_extension")[0]

    if layer_name.endswith(scan_content_ext):
        write_scan_file(path, data, meta)
    elif layer_name.endswith(cells_content_ext):
        write_cells_file(path, data, meta)
    elif layer_name.endswith(rings_content_ext):
        write_rings_file(path, data, meta)
    else:
        return None
    # return path to any file(s) that were successfully written
    return [path]


def write_multiple_layers(path: str, data: list[FullLayerData]) -> list[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s).
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    [path] : A list containing (potentially multiple) string paths to the saved file(s).
    """

    # implement your writer logic here ...

    # return path to any file(s) that were successfully written
    return [path]
