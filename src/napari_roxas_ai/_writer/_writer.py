"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd
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

    if Path(path).exists():
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

    # Update the sample name in the metadata
    existing_metadata.update({"sample_name": Path(Path(path).stem).stem})

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
    pil_image.save(path, compression="tiff_deflate")

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
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    dirname = Path(path).parent
    basename = Path(Path(path).stem).stem
    sample_path = dirname / basename

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "scan_")
    written_file_paths.append(metadata_file_path)

    # Save the image data
    scan_file_extension = "".join(
        settings.get("file_extensions.scan_file_extension")
    )
    scan_file_path = f"{sample_path}{scan_file_extension}"
    save_image(scan_file_path, data, rescale=False)
    written_file_paths.append(scan_file_path)

    # return path to any file(s) that were successfully written
    return written_file_paths


def format_output_table(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Standardize ROXAS-AI output table
    """

    # Reset index so CID is stable and no index leaks into CSV
    df = df.reset_index(drop=True)

    # Remove internal numeric id if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Remove centroid column if present
    if "centroid" in df.columns:
        df = df.drop(columns=["centroid"])

    # Insert ID and CID
    df.insert(0, "ID", sample_name)
    df.insert(1, "CID", range(1, len(df) + 1))

    # Rename ring_year → YEAR (if present)
    if "ring_year" in df.columns:
        df.rename(columns={"ring_year": "YEAR"}, inplace=True)

    # Rename lumen_area → LA (if present)
    if "lumen_area" in df.columns:
        df.rename(columns={"lumen_area": "LA"}, inplace=True)

    # Rename bot_angled_dist → RADDISTR (if present)
    if "top_angled_dist" in df.columns:
        df.rename(columns={"top_angled_dist": "RADDISTR"}, inplace=True)

    df = (df
          .pipe(shift_column, 2, "YEAR")    # Move YEAR to the 3. position
          .pipe(shift_column, 3, "LA")            # Move LA to the 4. position
          .pipe(shift_column, 4, "XPIX")          # Move XPIX to the 5. position
          .pipe(shift_column, 5, "YPIX")          # Move YPIX to the 6. position
          .pipe(shift_column, 6, "RADDISTR")      # Move RADDISTR to the 7. position
          .pipe(shift_column, 7, "RRADDISTR")     # Move RRADDISTR to the 8. position
          .pipe(shift_column, 8, "NBRNO")         # Move NBRNO to the 9. position
          .pipe(shift_column, 9, "NBRID")         # Move NBRID to the 10. position
          )

    df = (df
          .pipe(round_column, "LA", 2)            # round LA to 2 decimals
          .pipe(round_column, "RADDISTR", 0, integer=True)      # round RADDISTR to 0 decimals
          .pipe(round_column, "RRADDISTR", 0, integer=True)     # round RRADDISTR to 0 decimals
          )

    return df

def round_column(df, col, decimals, integer=False):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
        if integer:
            df[col] = df[col].astype("Int64")
    return df

def shift_column(df, index, colname):
    if colname not in df.columns:
        return df

    cols = df.columns.tolist()
    index = min(index, len(cols) - 1)
    cols.remove(colname)
    cols.insert(index, colname)
    return df[cols]



def write_cells_file(path: str, data: Any, meta: dict) -> list[str]:
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
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    dirname = Path(path).parent
    basename = Path(Path(path).stem).stem
    sample_path = dirname / basename

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "cells_")
    written_file_paths.append(metadata_file_path)

    # Save the tabular data
    if not meta["features"].empty:
        cells_table_file_extension = "".join(
            settings.get("file_extensions.cells_table_file_extension")
        )
        cells_table_file_path = f"{sample_path}{cells_table_file_extension}"
        features = format_output_table(
            meta["features"], meta["metadata"]["sample_name"]
        )
        features.to_csv(
            cells_table_file_path,
            sep=settings.get("tables.separator"),
            index=False,
        )
        written_file_paths.append(cells_table_file_path)

    # Save the image data
    cells_file_extension = "".join(
        settings.get("file_extensions.cells_file_extension")
    )
    cells_file_path = f"{sample_path}{cells_file_extension}"
    save_image(cells_file_path, data, rescale=True)
    written_file_paths.append(cells_file_path)

    # return path to any file(s) that were successfully written
    return written_file_paths


def write_rings_file(path: str, data: Any, meta: dict) -> list[str]:
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
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    dirname = Path(path).parent
    basename = Path(Path(path).stem).stem
    sample_path = dirname / basename

    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "rings_")
    written_file_paths.append(metadata_file_path)

    # Save the tabular data
    if not meta["features"].empty:
        rings_table_file_extension = "".join(
            settings.get("file_extensions.rings_table_file_extension")
        )
        rings_table_file_path = f"{sample_path}{rings_table_file_extension}"
        meta["features"].to_csv(
            rings_table_file_path,
            sep=settings.get("tables.separator"),
            index_label=settings.get("tables.index_column"),
        )
        written_file_paths.append(rings_table_file_path)

    # Save the image data
    rings_file_extension = "".join(
        settings.get("file_extensions.rings_file_extension")
    )
    rings_file_path = f"{sample_path}{rings_file_extension}"
    save_image(rings_file_path, data, rescale=False)
    written_file_paths.append(rings_file_path)

    # return path to any file(s) that were successfully written
    return written_file_paths


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
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    # Get the layer name from the metadata
    layer_name = meta["name"]

    # Get file extension settings and join the parts
    scan_content_ext = settings.get("file_extensions.scan_file_extension")[0]
    cells_content_ext = settings.get("file_extensions.cells_file_extension")[0]
    rings_content_ext = settings.get("file_extensions.rings_file_extension")[0]

    if layer_name.endswith(scan_content_ext):
        written_file_paths += write_scan_file(path, data, meta)
    elif layer_name.endswith(cells_content_ext):
        written_file_paths += write_cells_file(path, data, meta)
    elif layer_name.endswith(rings_content_ext):
        written_file_paths += write_rings_file(path, data, meta)
    else:
        written_file_paths += None
    # return path to any file(s) that were successfully written
    return written_file_paths


def write_multiple_layers(path: str, data: list[FullLayerData]) -> list[str]:
    """Writes multiple layers of different types.

    Parameters
    ----------
    path : str
        A string path indicating where to save the data file(s), and their basename.
    data : A list of layer tuples.
        Tuples contain three elements: (data, meta, layer_type)
        `data` is the layer data
        `meta` is a dictionary containing all other metadata attributes
        from the napari layer (excluding the `.data` layer attribute).
        `layer_type` is a string, eg: "image", "labels", "surface", etc.

    Returns
    -------
    written_file_paths : A list containing (potentially multiple) string paths to the saved file(s).
    """

    written_file_paths = []

    for layer_data, layer_meta, _ in data:
        written_file_paths += write_single_layer(path, layer_data, layer_meta)

    written_file_paths = list(set(written_file_paths))

    # return path to any file(s) that were successfully written
    return written_file_paths
