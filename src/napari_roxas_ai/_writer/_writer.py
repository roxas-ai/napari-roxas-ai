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
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageColor
import cv2

# Import SettingsManager to get file extensions
from napari_roxas_ai._settings import SettingsManager

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = tuple[DataType, dict, str]

# Get file extensions from settings
settings = SettingsManager()

from napari_roxas_ai._utils._metadata_keys import (
    RUN_METADATA_PREFIXES,
    SAMPLE_METADATA_PREFIXES,
    migrate_legacy_metadata_keys,
)


def update_metadata_file(
    path: str, metadata: dict, keys_prefix: Union[str, tuple[str, ...]]
) -> str:
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
    keys_prefix : str or tuple of str
        The prefix(es) of the keys to be included in the metadata file. Keys
        that match none of them are dropped, so any new metadata key must be
        covered by a prefix passed here to end up in the file.
    Returns
    -------
    str
        The path to the updated metadata file.
    """
    path_p = Path(path)

    if path_p.exists():
        with path_p.open("r", encoding="utf-8") as f:
            existing_metadata = json.load(f)
        # Rewrite legacy key names, so that saving a sample prepared with an
        # older version upgrades its metadata file instead of keeping both
        # spellings side by side
        existing_metadata = migrate_legacy_metadata_keys(existing_metadata)
    else:
        existing_metadata = {
            k: v for k, v in (metadata or {}).items()
            if isinstance(k, str)
            and k.startswith(SAMPLE_METADATA_PREFIXES)
        }

    src_meta = metadata or {}

    # Only include keys with the specified prefix
    filtered_metadata = {
        k: v for k, v in src_meta.items()
        if isinstance(k, str) and k.startswith(keys_prefix)
    }

    existing_metadata.update(filtered_metadata)

    # Prefer explicit sample_name from metadata
    sample_name = src_meta.get("sample_name")
    if not isinstance(sample_name, str) or not sample_name.strip():
        # Fallback: derive from filename by stripping metadata extension
        metadata_ext = "".join(settings.get("file_extensions.metadata_file_extension"))
        filename = path_p.name
        if metadata_ext and filename.endswith(metadata_ext):
            sample_name = filename[: -len(metadata_ext)]
        else:
            sample_name = path_p.stem

    existing_metadata["sample_name"] = sample_name

    with path_p.open("w", encoding="utf-8") as f:
        json.dump(existing_metadata, f, indent=4)

    return str(path_p)


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
        if image.size == 0:
            image = np.zeros_like(image, dtype=np.uint8)
        else:
            min_val = np.min(image)
            max_val = np.max(image)
            if min_val == max_val:
                image = np.zeros_like(image, dtype=np.uint8)
            elif min_val == 0 and max_val == 1:
                image = (image.astype(np.uint8)) * 255
            elif min_val == 0 and max_val == 255 and image.dtype == np.uint8:
                pass
            else:
                image = (
                    (image - min_val) / (max_val - min_val) * 255
                ).astype(np.uint8)

    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(image)

    # Save the image to the specified path
    pil_image.save(path, compression="tiff_deflate")

    return path

def _parse_color_rgb(color: Any) -> tuple[int, int, int]:
    """Parse color into (R, G, B) tuple in 0-255 range."""
    if isinstance(color, str):
        try:
            return ImageColor.getrgb(color)[:3]
        except Exception:
            return (0, 255, 255)
    elif isinstance(color, (list, tuple)) and len(color) >= 3:
        c = color[:3]
        if all(isinstance(x, float) and x <= 1.0 for x in c):
            return tuple(int(round(x * 255)) for x in c)
        return tuple(int(x) for x in c)
    return (0, 255, 255)


def interpolate_row_at_col(coords_rc: Any, col_query: float) -> float:
    """Compute the row (y) coordinate where a polyline crosses col_query."""
    if not isinstance(coords_rc, (list, tuple, np.ndarray)):
        return float("nan")
    pts = np.asarray(coords_rc, dtype=float)
    if pts.ndim != 2 or len(pts) == 0:
        return float("nan")
    if len(pts) == 1:
        return float(pts[0, 0])

    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        if (c0 <= col_query <= c1) or (c1 <= col_query <= c0):
            if c1 == c0:
                return float(r0)
            t = (col_query - c0) / (c1 - c0)
            return float(r0 + t * (r1 - r0))

    idx = int(np.argmin(np.abs(pts[:, 1] - col_query)))
    return float(pts[idx, 0])


def _draw_rings_and_years(img: np.ndarray, rings_features: pd.DataFrame) -> bool:
    """Draw the ring polylines and their YEAR labels onto img (BGR, in place).

    Returns True if at least one ring was labelled.
    """
    df = rings_features.copy()
    h, w = img.shape[:2]

    # Prefer YEAR ordering; otherwise fall back to cells_above if present
    if "YEAR" in df.columns:
        df = df.sort_values("YEAR").reset_index(drop=True)
    elif "cells_above" in df.columns:
        df = df.sort_values("cells_above").reset_index(drop=True)

    if df.empty:
        return False

    # Estimate mean ring width in pixels using consecutive cells_above differences
    mean_ring_width_px = 40.0  # fallback
    ring_width_px = None
    if "cells_above" in df.columns and len(df) >= 2:
        band_area = df["cells_above"].shift(-1) - df["cells_above"]
        rw = (band_area / float(w)).abs()
        # For the last ring, reuse median of previous widths
        if len(rw) > 1:
            rw.iloc[-1] = float(rw.iloc[:-1].median())
        ring_width_px = rw
        rw_valid = rw.dropna()
        if not rw_valid.empty:
            mean_ring_width_px = float(rw_valid.mean())

    # Target font height is determined by configured year_label_size percentage of mean ring width (min 12 px)
    raw_label_size = settings.get("annotated_image.year_label_size", 20)
    try:
        year_label_fraction = max(0.0, float(raw_label_size) / 100.0)
    except (ValueError, TypeError):
        year_label_fraction = 0.20

    target_h = max(12.0, year_label_fraction * mean_ring_width_px)

    # Convert desired pixel height to OpenCV font scale
    sample_label = "9999"
    (_tw, th), _baseline = cv2.getTextSize(
        sample_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1
    )
    font_scale = target_h / max(th, 1)

    # Make text thicker
    thickness_text = max(3, int(round(font_scale * 3.0)))

    # Thicker black ring separator lines
    ring_line_thickness = 6  # was 3 (double thickness)

    did_draw = False
    # Horizontally center the calendar years across the image width
    center_c = w / 2.0

    # Precompute row (y) positions at the center column for each ring boundary
    rows_at_center = []
    for i, row in df.iterrows():
        coords = row["RBXY"]
        r = interpolate_row_at_col(coords, center_c)
        rows_at_center.append(r)

    # Draw each ring and label
    for i, row in df.iterrows():
        coords = row["RBXY"]
        if not isinstance(coords, (list, tuple, np.ndarray)) or len(coords) < 2:
            continue

        # RBXY is [row, col]; OpenCV expects (x=col, y=row)
        pts = (
            np.array([[p[1], p[0]] for p in coords], dtype=np.float32)
            .round()
            .astype(np.int32)
            .reshape((-1, 1, 2))
        )

        cv2.polylines(
            img,
            [pts],
            isClosed=False,
            color=(0, 0, 0),
            thickness=ring_line_thickness,
        )

        year = None
        if "YEAR" in row.index:
            year = row["YEAR"]
        elif "ring_year" in row.index:
            year = row["ring_year"]
        if year is None:
            continue

        this_r = rows_at_center[i]
        if np.isnan(this_r):
            continue

        prev_r = (
            rows_at_center[i - 1]
            if i > 0 and not np.isnan(rows_at_center[i - 1])
            else 0.0
        )
        center_r = 0.5 * (prev_r + this_r)

        label = str(int(year))

        # Center the text around (center_c, center_r)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text
        )
        x = int(round(center_c - tw / 2.0))
        y = int(round(center_r + th / 2.0))

        # Keep text inside the image bounds
        x = max(0, min(x, w - tw - 1))
        y = max(th + 1, min(y, h - 1))

        cv2.putText(
            img,
            label,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness_text,
            cv2.LINE_AA,
        )

        did_draw = True

    return did_draw


def save_annotated_scan_image(
    scan_path: Optional[str] = None,
    annotated_path: str = "",
    rings_features: Optional[pd.DataFrame] = None,
    cells_data: Optional[np.ndarray] = None,
    scan_data: Optional[np.ndarray] = None,
) -> None:
    """Draw cells layer + ring polylines + YEAR labels onto scan image and save.

    - Overlays cell lumina with configured opacity and fill color.
    - Without rings (empty or missing rings_features), only the cell lumina
      are drawn onto the scan.
    - Uses a single, constant font size for all rings.
    - Font height is set based on configured year_label_size percentage of mean ring width (min 12 px).
    - YEAR label is placed vertically centered between this ring and the next one.
    - Ring separator lines are drawn black and thicker.
    - Draws black text (no outline).
    - Downscales the image according to configured scale_factor.
    """

    has_rings = (
        rings_features is not None
        and not rings_features.empty
        and "RBXY" in rings_features.columns
    )
    has_cells = (
        cells_data is not None
        and isinstance(cells_data, np.ndarray)
        and cells_data.size > 0
    )
    if not has_rings and not has_cells:
        return

    # Load or use in-memory scan image
    if scan_data is not None and isinstance(scan_data, np.ndarray) and scan_data.size > 0:
        img = scan_data.copy()
    elif scan_path is not None and Path(scan_path).exists():
        img = np.array(Image.open(scan_path))
    else:
        return

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w = img.shape[:2]

    # Blend cells layer if available
    if has_cells:
        if cells_data.ndim == 3:
            cells_mask = np.any(cells_data != 0, axis=-1)
        else:
            cells_mask = cells_data != 0

        if cells_mask.shape[:2] != (h, w):
            cells_mask = cv2.resize(
                cells_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if np.any(cells_mask):
            raw_opacity = settings.get("annotated_image.cell_opacity", 50)
            try:
                cell_opacity = float(raw_opacity) / 100.0
            except (ValueError, TypeError):
                cell_opacity = 0.50
            cell_opacity = max(0.0, min(1.0, cell_opacity))

            fill_color = settings.get("annotated_image.cell_fill_color", "yellow")
            r, g, b = _parse_color_rgb(fill_color)
            color_bgr = np.array([b, g, r], dtype=np.uint8)

            lut_3ch = np.zeros((1, 256, 3), dtype=np.uint8)
            for c in range(3):
                lut_3ch[0, :, c] = np.clip(
                    np.round(
                        np.arange(256) * (1.0 - cell_opacity) + color_bgr[c] * cell_opacity
                    ),
                    0,
                    255,
                ).astype(np.uint8)

            blended = cv2.LUT(img, lut_3ch)
            mask_u8 = cells_mask.astype(np.uint8) if cells_mask.dtype != np.uint8 else cells_mask
            cv2.copyTo(blended, mask_u8, img)

    if has_rings and not _draw_rings_and_years(img, rings_features):
        return

    # Resize image if scale factor is configured and different from 1.0
    raw_scale = settings.get("annotated_image.scale_factor", 1.0)
    try:
        scale_factor = float(raw_scale)
    except (ValueError, TypeError):
        scale_factor = 1.0

    if scale_factor > 0 and scale_factor != 1.0:
        new_w = max(1, int(round(w * scale_factor)))
        new_h = max(1, int(round(h * scale_factor)))
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR,
        )

    raw_quality = settings.get("annotated_image.quality", 75)
    try:
        quality = int(raw_quality)
    except (ValueError, TypeError):
        quality = 75
    quality = max(1, min(100, quality))

    success = cv2.imwrite(
        str(annotated_path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        is_success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if is_success:
            with open(annotated_path, "wb") as f:
                f.write(buf)



def write_scan_file(path: str, data: Any, meta: dict) -> str:
    """Writes the metadata of a scan layer.

    The scan image itself is never written: it is the untouched input image,
    and re-encoding it would either lose quality (JPEG) or create a second
    scan file next to the original when its extension differs from the
    default one (e.g. ".scan.jpeg" next to a new ".scan.jpg").

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
    written_file_paths : A list containing the string path to the metadata file.
    """

    written_file_paths = []

    md = meta.get("metadata") or {}

    stem = md.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        project_dir = Path(settings.get("project_directory")).resolve()
        sample_path = (project_dir / stem).resolve()  # stem is stored relative
    else:
        # Fallback: sample_name in current dir of 'path'
        sample_name = md.get("sample_name")
        if not isinstance(sample_name, str) or not sample_name.strip():
            sample_name = Path(Path(path).stem).stem
        sample_path = Path(path).parent / sample_name


    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(metadata_file_path, meta["metadata"], "scan_")
    written_file_paths.append(metadata_file_path)

    # return path to any file(s) that were successfully written
    return written_file_paths


def rename_column(df: pd.DataFrame, old_name: str, roxas_name: str):
    if old_name in df.columns:
        df.rename(columns={old_name: roxas_name}, inplace=True)

def format_rings_output_table(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Standardize ROXAS-AI rings output table
    """

    df = df.copy()

    # Remove internal numeric id if present
    df = df.reset_index(drop=True)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    if "ID" in df.columns:
        df["ID"] = sample_name
    else:
        df.insert(0, "ID", sample_name)

    if "MRW" in df.columns and "ring_angle_width" in df.columns:
        df = df.drop(columns=["ring_angle_width"])
    elif "MRW" not in df.columns and "ring_angle_width" in df.columns:
        df = df.rename(columns={"ring_angle_width": "MRW"})

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
        "AOIAR",
        "RAOIAR",
        "CWTPI",
        "CWTBA",
        "CWTLE",
        "CWTRI",
        "CWTTAN",
        "CWTRAD",
        "CWTALL",
        "RTSR",
        "CTSR",
        "DHW",
        "DHM",
        "DRAD",
        "DTAN",
        "TB2",
        "CWA",
        "RWD",
        "RBXY",
        "cells_above",
        "enabled"
    ]

    # Keep only those that actually exist in df
    columns_order_existing = [c for c in columns_order if c in df.columns]

    # All columns not listed are appended at the end
    # other_cols = [c for c in df.columns if c not in columns_order_existing]
    # df = df[columns_order_existing + other_cols]

    df = df[columns_order_existing]

    # round columns with column name and number of decimals
    df = (df
          .pipe(round_column, "MRW", 0, integer=True)
          .pipe(round_column, "RA", 3)
          .pipe(round_column, "CD", 2)
          .pipe(round_column, "CTA", 3)
          .pipe(round_column, "RCTA", 2)
          .pipe(round_column, "MLA", 2)
          .pipe(round_column, "MINLA", 2)
          .pipe(round_column, "MAXLA", 2)
          .pipe(round_column_sci, "KH", 3)  # round KH using a scientific notation format
          .pipe(round_column_sci, "KS", 3)  # round KS using a scientific notation format
          .pipe(round_column, "CWTPI", 2)
          .pipe(round_column, "CWTBA", 2)
          .pipe(round_column, "CWTLE", 2)
          .pipe(round_column, "CWTRI", 2)
          .pipe(round_column, "CWTTAN", 2)
          .pipe(round_column, "CWTRAD", 2)
          .pipe(round_column, "CWTALL", 2)
          .pipe(round_column, "RTSR", 3)
          .pipe(round_column, "CTSR", 3)
          .pipe(round_column, "DHW", 2)
          .pipe(round_column, "DHM", 2)
          .pipe(round_column, "DRAD", 2)
          .pipe(round_column, "DTAN", 2)
          .pipe(round_column, "TB2", 4)
          .pipe(round_column, "CWA", 2)
          .pipe(round_column, "RWD", 3)
          )

    return df

def format_cells_output_table(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    """
    Standardize ROXAS-AI cells output table
    """

    df = df.copy()
    # Reset index so CID is stable and no index leaks into CSV
    df = df.reset_index(drop=True)

    # Remove internal numeric id if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Remove centroid column if present
    if "centroid" in df.columns:
        df = df.drop(columns=["centroid"])

    # Insert ID and CID
    if "ID" in df.columns:
        df["ID"] = sample_name
    else:
        df.insert(0, "ID", sample_name)
    if "CID" not in df.columns:
        df.insert(1, "CID", range(1, len(df) + 1))

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
        "AOI",
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
          .pipe(round_column, "LA", 1)
          .pipe(round_column, "RADDISTR", 1)
          .pipe(round_column, "RRADDISTR", 2)
          .pipe(round_column, "ASP", 2)
          .pipe(round_column, "MAJAX", 2)
          .pipe(round_column_sci, "KH", 3)  # round KH using a scientific notation format
          .pipe(round_column, "CWTPI", 2)
          .pipe(round_column, "CWTBA", 2)
          .pipe(round_column, "CWTLE", 2)
          .pipe(round_column, "CWTRI", 2)
          .pipe(round_column, "CWTTAN", 2)
          .pipe(round_column, "CWTRAD", 2)
          .pipe(round_column, "CWTALL", 2)
          .pipe(round_column, "RTSR", 3)
          .pipe(round_column, "CTSR", 3)
          .pipe(round_column, "DH", 2)
          .pipe(round_column, "DRAD", 2)
          .pipe(round_column, "DTAN", 2)
          .pipe(round_column, "TB2", 4)
          .pipe(round_column, "CWA", 0, integer=True)
          .pipe(round_column, "RWD", 3)
          )

    return df

def round_column(df, col, decimals, integer=False):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
        if integer:
            df[col] = df[col].astype("Int64")
    return df

def round_column_sci(df, col, decimals):
    if col in df.columns:
        df[col] = (
            pd.to_numeric(df[col], errors="coerce")
            .apply(lambda x: f"{x:.{decimals}E}" if pd.notna(x) else np.nan)
        )
    return df

def write_cells_file(
    path: str,
    data: Any,
    meta: dict,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> list[str]:
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
    progress_callback : callable, optional
        A callback receiving completion fraction (0.0 to 1.0) for this layer.

    Returns
    -------
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    md = meta.get("metadata") or {}
    cb = progress_callback or md.get("progress_callback")

    stem = md.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        project_dir = Path(settings.get("project_directory")).resolve()
        sample_path = (project_dir / stem).resolve()  # stem is stored relative
    else:
        # Fallback: sample_name in current dir of 'path'
        sample_name = md.get("sample_name")
        if not isinstance(sample_name, str) or not sample_name.strip():
            sample_name = Path(Path(path).stem).stem
        sample_path = Path(path).parent / sample_name


    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(
        metadata_file_path, meta["metadata"], ("cells_", *RUN_METADATA_PREFIXES)
    )
    written_file_paths.append(metadata_file_path)
    if cb:
        cb(0.1)

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
    if cb:
        cb(0.4)

    # Save the image data
    cells_file_extension = "".join(
        settings.get("file_extensions.cells_file_extension")
    )
    cells_file_path = f"{sample_path}{cells_file_extension}"
    save_image(cells_file_path, data, rescale=True)
    written_file_paths.append(cells_file_path)
    if cb:
        cb(1.0)

    # return path to any file(s) that were successfully written
    return written_file_paths


def write_rings_file(
    path: str,
    data: Any,
    meta: dict,
    progress_callback: Optional[Callable[[float], None]] = None,
    save_annotated: bool = True,
) -> list[str]:
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
    progress_callback : callable, optional
        A callback receiving completion fraction (0.0 to 1.0) for this layer.
    save_annotated : bool, optional
        Whether to (re)draw the annotated scan image of the sample. Manual
        saves skip it, as it is the slowest part of writing a rings layer.

    Returns
    -------
    written_file_paths : A list containing the string path to the saved file.
    """

    written_file_paths = []

    md = meta.get("metadata") or {}
    cb = progress_callback or md.get("progress_callback")

    stem = md.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        project_dir = Path(settings.get("project_directory")).resolve()
        sample_path = (project_dir / stem).resolve()  # stem is stored relative
    else:
        # Fallback: sample_name in current dir of 'path'
        sample_name = md.get("sample_name")
        if not isinstance(sample_name, str) or not sample_name.strip():
            sample_name = Path(Path(path).stem).stem
        sample_path = Path(path).parent / sample_name


    # Update the metadata file
    metadata_file_extension = "".join(
        settings.get("file_extensions.metadata_file_extension")
    )
    metadata_file_path = f"{sample_path}{metadata_file_extension}"
    update_metadata_file(
        metadata_file_path, meta["metadata"], ("rings_", *RUN_METADATA_PREFIXES)
    )
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
    if cb:
        cb(0.3)

    # Save the image data
    rings_file_extension = "".join(
        settings.get("file_extensions.rings_file_extension")
    )
    rings_file_path = f"{sample_path}{rings_file_extension}"
    save_image(rings_file_path, data, rescale=False)
    written_file_paths.append(rings_file_path)
    if cb:
        cb(0.6)

    # --- Save annotated scan image ---
    if save_annotated and not meta["features"].empty:
        cells_data = (meta.get("metadata") or {}).get("cells_data")
        scan_data = (meta.get("metadata") or {}).get("scan_data")
        annotated_path = save_sample_annotated_image(
            sample_path,
            meta["features"],
            cells_data=cells_data,
            scan_data=scan_data,
        )
        if annotated_path is not None:
            written_file_paths.append(annotated_path)
    if cb:
        cb(1.0)

    return written_file_paths


def save_sample_annotated_image(
    sample_path: Union[str, Path],
    rings_features: pd.DataFrame,
    cells_data: Optional[np.ndarray] = None,
    scan_data: Optional[np.ndarray] = None,
) -> Union[str, None]:
    """
    Draw the cells and rings onto the scan of a sample and save it next to the sample.

    Parameters
    ----------
    sample_path : str or Path
        The sample stem (sample directory / sample name, without extensions).
    rings_features : pd.DataFrame
        The rings table holding the ring boundaries and years.
    cells_data : np.ndarray, optional
        In-memory cell labels/mask. If None, looks for the sample's cells file on disk.
    scan_data : np.ndarray, optional
        In-memory scan image. If None, looks for the sample's scan file on disk.

    Returns
    -------
    str or None
        The path to the annotated image, or None if it was not written.
    """
    sample_path = Path(sample_path)
    basename = sample_path.name  # use for annotated naming
    scan_file_extension = "".join(settings.get("file_extensions.scan_file_extension"))
    scan_file_path = f"{sample_path}{scan_file_extension}"
    if not Path(scan_file_path).exists():
        # The scan may use another image extension (e.g. ".scan.jpeg"), as when loading
        scan_content_ext = settings.get("file_extensions.scan_file_extension")[0]
        for ext in settings.get("file_extensions.image_file_extensions"):
            alt_path = f"{sample_path}{scan_content_ext}{ext}"
            if Path(alt_path).exists():
                scan_file_path = alt_path
                break

    # If cells_data not provided, look for cells raster file on disk
    if cells_data is None:
        cells_file_extension = "".join(
            settings.get("file_extensions.cells_file_extension")
        )
        cells_file_path = f"{sample_path}{cells_file_extension}"
        if not Path(cells_file_path).exists():
            cells_content_ext = settings.get("file_extensions.cells_file_extension")[0]
            for ext in settings.get("file_extensions.image_file_extensions"):
                alt_cells = f"{sample_path}{cells_content_ext}{ext}"
                if Path(alt_cells).exists():
                    cells_file_path = alt_cells
                    break
        if Path(cells_file_path).exists():
            try:
                with Image.open(cells_file_path) as c_img:
                    cells_data = np.array(c_img)
            except Exception as e:
                print(f"Could not load cells image from {cells_file_path}: {e}")

    # annotated filename: <sample>_annotated.jpg, whatever the scan format:
    # JPEG keeps it small, and the preparation skips exactly this name
    annotated_path = str(sample_path.with_name(f"{basename}_annotated.jpg"))

    try:
        has_scan_source = (scan_data is not None and isinstance(scan_data, np.ndarray) and scan_data.size > 0) or Path(scan_file_path).exists()
        if not has_scan_source:
            print(f"Annotated image export skipped for {basename}: no scan image available")
            return None
        save_annotated_scan_image(
            scan_path=scan_file_path,
            annotated_path=annotated_path,
            rings_features=rings_features,
            cells_data=cells_data,
            scan_data=scan_data,
        )
        return annotated_path
    except Exception as e:
        print(f"Annotated image export failed for {basename}: {e}")
        return None


def save_layers_annotated_image(
    layers: Sequence[Any], sample_stem_path: Optional[str]
) -> Union[str, None]:
    """
    Draw the annotated scan image of a sample from its layers in the viewer.

    The scan, cells and rings of the sample are taken from the in-memory
    layers, so nothing has to be read from disk; the cells fall back to the
    sample's cells file when no cells layer is loaded.

    Parameters
    ----------
    layers : sequence of napari layers
        The layers to look for the sample's scan, cells and rings layers in.
    sample_stem_path : str
        The sample stem, relative to the project directory (as stored in the
        layers' metadata).

    Returns
    -------
    str or None
        The path to the annotated image, or None if it was not written.
    """
    if not isinstance(sample_stem_path, str) or not sample_stem_path.strip():
        print("Annotated image export skipped: the sample stem path is unknown")
        return None

    sample_path = Path(sample_stem_path)
    if not sample_path.is_absolute():
        project_dir = settings.get("project_directory")
        if not project_dir:
            print("Annotated image export skipped: project_directory is not configured")
            return None
        sample_path = Path(project_dir) / sample_path

    scan_ext = settings.get("file_extensions.scan_file_extension")[0]
    cells_ext = settings.get("file_extensions.cells_file_extension")[0]
    rings_ext = settings.get("file_extensions.rings_file_extension")[0]

    scan_data = None
    cells_data = None
    rings_features = pd.DataFrame()
    for layer in layers:
        layer_meta = getattr(layer, "metadata", None) or {}
        if layer_meta.get("sample_stem_path") != sample_stem_path:
            continue
        if layer.name.endswith(scan_ext):
            scan_data = layer.data
        elif layer.name.endswith(cells_ext):
            cells_data = layer.data
        elif layer.name.endswith(rings_ext):
            features = getattr(layer, "features", None)
            if features is not None:
                rings_features = features

    return save_sample_annotated_image(
        sample_path,
        rings_features,
        cells_data=cells_data,
        scan_data=scan_data,
    )


def write_single_layer(
    path: str,
    data: Any,
    meta: dict,
    progress_callback: Optional[Callable[[float], None]] = None,
    save_annotated: bool = True,
) -> list[str]:
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
    progress_callback : callable, optional
        A callback receiving completion fraction (0.0 to 1.0) for this layer.
    save_annotated : bool, optional
        Whether writing a rings layer also (re)draws the annotated scan image.

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
        written_file_paths += write_cells_file(
            path, data, meta, progress_callback=progress_callback
        )
    elif layer_name.endswith(rings_content_ext):
        written_file_paths += write_rings_file(
            path,
            data,
            meta,
            progress_callback=progress_callback,
            save_annotated=save_annotated,
        )
    else:
        return written_file_paths
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
