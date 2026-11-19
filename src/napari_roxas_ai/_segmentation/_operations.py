"""
torch, torch.package.PackageImporter and ._cells_model are imported lazily
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

from napari_roxas_ai._edition import update_rings_geometries
from napari_roxas_ai._jobs import Message, Progress
from napari_roxas_ai._reader import get_metadata_from_file, read_scan_file
from napari_roxas_ai._writer import write_single_layer
from napari_roxas_ai._writer._writer import save_sample_annotated_image

from .._utils._metadata_keys import SAMPLE_METADATA_PREFIXES
from .._utils._segmentation_postprocess import (
    remove_border_touching_components,
)
from .._utils._torch_device import select_torch_device

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

# Progress steps of each phase of a single segmentation


def segment_sample(
    image: np.ndarray,
    cells_model_path: Optional[Path],
    rings_model_path: Optional[Path],
    try_gpu: bool,
    cells_name: str,
    rings_name: str,
):
    """
    Segment the cells and the rings of one image.

    Returns {"cells": {"data", "name"}, "rings": {"data", "name", "features"}}
    with the segmented parts.
    """
    PHASE_STEPS = 100
    phases = 2 * (
        int(cells_model_path is not None) + int(rings_model_path is not None)
    )
    total = phases * PHASE_STEPS
    start = 0  # of the current phase
    results = {}

    if cells_model_path is not None:
        yield Progress(start, total, "Loading the cells model…")
        _import_torch()
        cells_model = load_cells_model(cells_model_path, try_gpu)
        start += PHASE_STEPS

        yield Progress(start, total, "Detecting cells…")
        steps = cells_model.infer_steps(image)
        while True:
            try:
                done, patches = next(steps)
            except StopIteration as stop:
                cells_labels = stop.value
                break
            yield Progress(
                start + PHASE_STEPS * done // patches,
                total,
                f"Detecting cells {done}/{patches}",
            )

        # Normalize to 0/1 and remove border-touching components
        cells_binary = (cells_labels > 0).astype("uint8")
        cells_binary = remove_border_touching_components(cells_binary)
        results["cells"] = {"data": cells_binary, "name": cells_name}
        start += PHASE_STEPS

    if rings_model_path is not None:
        yield Progress(start, total, "Loading the rings model…")
        _import_torch()
        rings_model = load_rings_model(rings_model_path, try_gpu)
        start += PHASE_STEPS

        # The rings model infers in one step (its code is part of the model)
        yield Progress(start, total, "Detecting rings…")
        rings_labels, rings_boundaries = rings_model.infer(image)
        results["rings"] = {
            "data": rings_labels.astype("int32"),
            "name": rings_name,
            "features": boundaries_table(rings_boundaries, image.shape[1]),
        }
        # Lets a cancel during the detection take effect
        yield Progress(total, total, "Detecting rings…")

    return results


def segment_samples(
    scan_file_paths: list[str],
    cells_model_path: Optional[Path],
    rings_model_path: Optional[Path],
    try_gpu: bool,
    cells_extension: str,
    rings_extension: str,
    default_rings_year: Any,
    project_dir: str,
):
    """
    Segment the scans one after the other and write the cells and/or rings
    files and the annotated image of each.

    A scan that fails is reported and skipped.
    """
    total = len(scan_file_paths)
    yield Progress(0, total, "Loading the models…")
    _import_torch()
    cells_model = rings_model = None
    if cells_model_path is not None:
        cells_model = load_cells_model(cells_model_path, try_gpu)
    if rings_model_path is not None:
        rings_model = load_rings_model(rings_model_path, try_gpu)

    processed = 0
    for file_idx, scan_file_path in enumerate(scan_file_paths, 1):
        file_name = Path(scan_file_path).name
        yield Progress(
            file_idx - 1, total, f"Segmenting {file_name} ({file_idx}/{total})"
        )
        try:
            scan_data, scan_add_kwargs, _ = read_scan_file(scan_file_path)

            sample_metadata = {
                k: v
                for k, v in scan_add_kwargs["metadata"].items()
                if isinstance(k, str)
                and k.startswith(SAMPLE_METADATA_PREFIXES)
            }

            cells_data = None
            rings_features = pd.DataFrame()

            if cells_model is not None:
                cells_labels = cells_model.infer(scan_data)
                cells_binary = (cells_labels > 0).astype("uint8")
                cells_binary = remove_border_touching_components(cells_binary)
                cells_data = (cells_binary * 255).astype("uint8")

                cells_layer_name = (
                    f"{sample_metadata['sample_name']}{cells_extension}"
                )
                cells_add_kwargs = {
                    "name": cells_layer_name,
                    "scale": scan_add_kwargs.get("scale"),
                    "features": pd.DataFrame(),
                    "metadata": {
                        **sample_metadata,
                        "cells_segmentation_model": Path(
                            cells_model_path
                        ).name,
                        "cells_segmentation_datetime": (
                            datetime.now().isoformat()
                        ),
                    },
                }

                write_single_layer(
                    path=scan_file_path, data=cells_data, meta=cells_add_kwargs
                )

            if rings_model is not None:
                rings_labels, rings_boundaries = rings_model.infer(scan_data)
                boundaries_df = boundaries_table(
                    rings_boundaries, scan_data.shape[1]
                )

                sample_name = sample_metadata.get(
                    "sample_name", Path(scan_file_path).stem
                )
                rings_layer_name = f"{sample_name}{rings_extension}"

                sample_stem_path = sample_metadata.get("sample_stem_path")
                if not sample_stem_path:
                    # Fallback to file path without extensions if not in metadata
                    sample_stem_path = str(
                        Path(scan_file_path).parent / Path(scan_file_path).stem
                    )

                metadata_file_contents = get_metadata_from_file(
                    path=sample_stem_path, path_is_stem=True
                )
                last_year = (
                    metadata_file_contents.get(
                        "rings_outmost_complete_year", default_rings_year
                    )
                    if metadata_file_contents
                    else default_rings_year
                )

                new_rings_table, rings_raster, _cmap_tmp = (
                    update_rings_geometries(
                        rings_table=boundaries_df,
                        last_year=int(last_year),
                        image_shape=rings_labels.shape,
                    )
                )

                rings_add_kwargs = {
                    "name": rings_layer_name,
                    "scale": scan_add_kwargs.get("scale"),
                    "features": new_rings_table,
                    "metadata": {
                        **sample_metadata,
                        "rings_outmost_complete_year": int(last_year),
                        "rings_segmentation_model": Path(
                            rings_model_path
                        ).name,
                        "rings_segmentation_datetime": (
                            datetime.now().isoformat()
                        ),
                    },
                }

                # The annotated image is drawn below, once for all modes
                write_single_layer(
                    path=scan_file_path,
                    data=rings_raster,
                    meta=rings_add_kwargs,
                    save_annotated=False,
                )
                rings_features = new_rings_table

            # Annotated image: cells and rings (if any) drawn onto the scan.
            # Without new cells, the cells file on disk (if any) is used.
            save_sample_annotated_image(
                _sample_path(scan_file_path, sample_metadata, project_dir),
                rings_features,
                cells_data=cells_data,
                scan_data=scan_data,
            )

            tasks = []
            if cells_model is not None:
                tasks.append(f"Cells ({Path(cells_model_path).name})")
            if rings_model is not None:
                tasks.append(f"Rings ({Path(rings_model_path).name})")
            print(
                f"[Batch Segmentation] [{file_idx}/{total}] {file_name}: "
                f"{', '.join(tasks)}"
            )
            processed += 1

        except Exception as e:  # noqa: BLE001
            print(
                f"[Batch Segmentation] [{file_idx}/{total}] Error processing "
                f"{file_name}: {e}"
            )

    if total > 0:
        summary = (
            f"[Batch Segmentation] Completed {processed} of {total} file(s) "
            "successfully."
        )
        print(summary)
        yield Message(summary)


def _import_torch() -> None:
    import torch

    # Prevent OpenMP crash when running inside a QThread with PyQt6 on macOS
    torch.set_num_threads(1)


def load_cells_model(weights_path: Path, try_gpu: bool):
    import torch

    from ._cells_model import CellsSegmentationModel

    cells_model = CellsSegmentationModel()
    cells_model.load_weights(weights_path)
    cells_model.available_device = select_torch_device(try_gpu)
    cells_model.to(device=cells_model.available_device)
    print(f"Cells model running on: {cells_model.available_device}")
    # Synchronize internal device attribute for models that use it in infer()
    device = cells_model.available_device
    try:
        cells_model.device = device
    except Exception:  # noqa: BLE001
        try:
            setattr(cells_model, "device", device)  # noqa: B010
        except Exception:  # noqa: BLE001
            if hasattr(cells_model, "__dict__"):
                cells_model.__dict__["device"] = device
    # Use local device variable for autocast checks to avoid restricted
    # attribute assignment
    device_obj = torch.device(cells_model.available_device)
    cells_model.use_autocast = bool(
        torch.amp.autocast_mode.is_autocast_available(device_obj.type)
        and try_gpu
        and cells_model.available_device != "cpu"
    )
    return cells_model


def load_rings_model(package_path: Path, try_gpu: bool):
    import torch
    from torch.package import PackageImporter

    rings_model = PackageImporter(package_path).load_pickle(
        "LinearRingModel", "model.pkl"
    )
    rings_model.available_device = select_torch_device(try_gpu)
    rings_model.to(device=rings_model.available_device)
    print(f"Rings model running on: {rings_model.available_device}")
    # Synchronize internal device attribute for packaged models
    device = rings_model.available_device
    try:
        rings_model.device = device
    except Exception:  # noqa: BLE001
        try:
            setattr(rings_model, "device", device)  # noqa: B010
        except Exception:  # noqa: BLE001
            if hasattr(rings_model, "__dict__"):
                rings_model.__dict__["device"] = device
    # Use local device variable for autocast checks
    device_obj = torch.device(rings_model.available_device)
    rings_model.use_autocast = bool(
        torch.amp.autocast_mode.is_autocast_available(device_obj.type)
        and try_gpu
        and rings_model.available_device != "cpu"
    )
    return rings_model


def boundaries_table(boundaries, image_width: int) -> pd.DataFrame:
    """
    The ring boundaries found by the rings model as a table (column RBXY),
    after a dummy boundary at the top (row 0) that marks the incomplete ring.
    """
    import torch

    boundary_data = [{"RBXY": [[0, 0], [0, image_width - 1]]}]
    for boundary in boundaries:
        if isinstance(boundary, torch.Tensor):
            coords = boundary.cpu().numpy().tolist()
        else:
            coords = boundary
        boundary_data.append({"RBXY": coords})
    return pd.DataFrame(boundary_data)


def _sample_path(
    scan_file_path: str, sample_metadata: dict, project_dir: str
) -> Path:
    """The sample stem (directory / sample name) the outputs are written to."""
    stem = sample_metadata.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        project_path = Path(project_dir).resolve()
        return (project_path / stem).resolve()  # stem is stored relative
    sample_name = sample_metadata.get("sample_name")
    if not isinstance(sample_name, str) or not sample_name.strip():
        sample_name = Path(Path(scan_file_path).stem).stem
    return Path(scan_file_path).parent / sample_name
