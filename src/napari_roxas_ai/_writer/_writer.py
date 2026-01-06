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


def rename_column(df: pd.DataFrame, old_name: str, roxas_name: str):
    if old_name in df.columns:
        df.rename(columns={old_name: roxas_name}, inplace=True)

def format_rings_output_table(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Standardize ROXAS-AI rings output table
    """

    # Remove internal numeric id if present
    df = df.reset_index(drop=True)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df.insert(0, "ID", sample_name)

    rename_column(df, "ring_year", "YEAR")
    rename_column(df, "ring_angle_width", "MRW")

    columns_order = [
        "ID",
        "YEAR",
        "RA",
        "MRW",
        "CNO",
        "CD",
        "CTA",
        "RCTA",
        "MLA",
        "MINLA",
        "MAXLA",
        "KH",
        "KS",
        "RVGI",
        "RVSF",
        "RGSGV",
        "CWTPI",
        "CWTBA",
        "CWTLE",
        "CWTRI",
        "CWTTAN",
        "CWTRAD",
        "CWTALL",
        "RTSR",
        "CTSR",
        "DH",
        "DH2",
        "DRAD",
        "DTAN",
        "TB2",
        "CWA",
        "RWD",
    ]

    # Keep only those that actually exist in df
    columns_order_existing = [c for c in columns_order if c in df.columns]

    # All columns not listed are appended at the end
    # other_cols = [c for c in df.columns if c not in columns_order_existing]
    # df = df[columns_order_existing + other_cols]

    df = df[columns_order_existing]
    # round columns with column name and number of decimals
    df = (df
          .pipe(round_column, "MRW", 2)
          .pipe(round_column, "RA", 3)
          .pipe(round_column, "CD", 2)
          .pipe(round_column, "CTA", 3)
          .pipe(round_column, "RCTA", 2)
          .pipe(round_column, "MLA", 2)
          .pipe(round_column, "MINLA", 2)
          .pipe(round_column, "MAXLA", 2)
          .pipe(round_column, "CWTPI", 2)
          .pipe(round_column, "CWTBA", 2)
          .pipe(round_column, "CWTLE", 2)
          .pipe(round_column, "CWTRI", 2)
          .pipe(round_column, "CWTTAN", 2)
          .pipe(round_column, "CWTRAD", 2)
          .pipe(round_column, "CWTALL", 2)
          .pipe(round_column, "RTSR", 2)
          .pipe(round_column, "CTSR", 2)
          .pipe(round_column, "DH", 2)
          .pipe(round_column, "DH2", 2)
          .pipe(round_column, "DRAD", 2)
          .pipe(round_column, "DTAN", 2)
          .pipe(round_column, "TB2", 2)
          .pipe(round_column, "CWA", 2)
          .pipe(round_column, "RWD", 2)
          )

    return df

def format_cells_output_table(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Standardize ROXAS-AI cells output table
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

    rename_column(df, "ring_year", "YEAR")
    rename_column(df, "lumen_area", "LA")
    rename_column(df, "top_angled_dist", "RADDISTR")
    rename_column(df, "CWT_pith", "CWTPI")
    rename_column(df, "CWT_bark", "CWTBA")
    rename_column(df, "CWT_left", "CWTLE")
    rename_column(df, "CWT_right", "CWTRI")
    rename_column(df, "lumen_diam_rad", "DRAD")
    rename_column(df, "lumen_diam_tang", "DTAN")

    columns_order = [
        "ID",
        "CID",
        "YEAR",
        "LA",
        "XPIX",
        "YPIX",
        "RADDISTR",
        "RRADDISTR",
        "NBRNO",
        "NBRID",
        "ASP",
        "MAJAX",
        "KH",
        "CWTPI",
        "CWTBA",
        "CWTLE",
        "CWTRI",
        "CWTTAN",
        "CWTRAD",
        "CWTALL",
        "RTSR",
        "CTSR",
        "DH",
        "DRAD",
        "DTAN",
        "TB2",
        "CWA",
        "RWD"
    ]

    # Keep only those that actually exist in df
    columns_order_existing = [c for c in columns_order if c in df.columns]

    # All columns not listed are appended at the end
    # other_cols = [c for c in df.columns if c not in columns_order_existing]

    df = df[columns_order_existing]

    df = (df
          .pipe(round_column, "LA", 2)                          # round LA to 2 decimals
          .pipe(round_column, "RADDISTR", 0, integer=True)      # round RADDISTR to 0 decimals
          .pipe(round_column, "RRADDISTR", 0, integer=True)     # round RRADDISTR to 0 decimals
          .pipe(round_column, "ASP", 3)                         # round ASP to 3 decimals
          .pipe(round_column, "MAJAX", 0, integer=True)         # round MAJAX to 0 decimals
          )

    return df

def round_column(df, col, decimals, integer=False):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
        if integer:
            df[col] = df[col].astype("Int64")
    return df


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
        features = format_cells_output_table(
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
        features = format_rings_output_table(
            meta["features"], meta["metadata"]["sample_name"]
        )
        features.to_csv(
            rings_table_file_path,
            sep=settings.get("tables.separator"),
            index=False,
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
