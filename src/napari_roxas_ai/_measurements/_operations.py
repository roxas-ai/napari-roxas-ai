from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from napari_roxas_ai._jobs import Progress
from napari_roxas_ai._saving._operations import LayerSnapshot, save_layers
from napari_roxas_ai._writer._writer import save_sample_annotated_image

from .._utils._metadata_keys import MEASUREMENT_PARAMETER_KEYS
from .._utils._version_utils import (
    get_measurement_operator,
    get_software_version,
)
from ._sample_measurer import SampleAnalyzer


def measure_sample(
    config: dict[str, Any],
    measurement: str,
    cells_array: np.ndarray,
    rings_table: pd.DataFrame | None = None,
):
    """
    Measure the cells and/or rings of a sample ("both", "cells" or "rings").

    Measuring the cells needs the cells array only, measuring the rings needs
    the rings table and a cells array of the sample's shape (zeros if the
    cells are not measured).

    Returns (cells_table, rings_table, cleaned_cells_array / None)
    """
    if rings_table is None:
        rings_table = pd.DataFrame()
    analyzer = SampleAnalyzer(config, cells_array, rings_table)
    cleaned_cells_array = None
    if measurement == "both":
        cells_table, rings_table = analyzer.analyze_sample()
        cleaned_cells_array = analyzer.cells_array
    elif measurement == "cells":
        cells_table = analyzer.analyze_cells()
        cleaned_cells_array = analyzer.cells_array
        rings_table = pd.DataFrame()
    elif measurement == "rings":
        cells_table = pd.DataFrame()
        rings_table = analyzer.analyze_rings()
    else:
        raise ValueError(f"Unknown measurement {measurement!r}")
    return cells_table, rings_table, cleaned_cells_array


@dataclass
class MeasuredLayer:
    """What changes on a measured layer: applied to it once saved."""

    features: pd.DataFrame
    metadata: dict
    data: np.ndarray | None = None  # replaced cells data, if any


@dataclass
class MeasurementResult:
    cells: MeasuredLayer | None = None
    rings: MeasuredLayer | None = None
    annotated_path: str | None = None


def measure_and_save(
    config: dict[str, Any],
    measurement: str,
    cells_layer: LayerSnapshot | None,
    rings_layer: LayerSnapshot | None,
    cells_array: np.ndarray,
    rings_table: pd.DataFrame | None,
    project_dir: str,
):
    """
    Measure a sample, write the measured layers and the annotated image.
    Returns a MeasurementResult, that should be applied to the layers by
    the caller.
    """
    # Steps: measuring, saving, drawing the annotated image (with rings)
    loaded_rings = rings_layer.kwargs.get("features") if rings_layer else None
    has_rings = measurement in ("both", "rings") or (
        loaded_rings is not None and not loaded_rings.empty
    )
    total = 2 + int(has_rings)

    yield Progress(0, total, "Measuring…")
    cells_table, rings_table, cleaned_cells_array = measure_sample(
        config,
        measurement,
        cells_array.astype("uint8") * 255,
        rings_table,
    )

    # One timestamp per measurement run: cells and rings carry the same value
    run_metadata = {
        "meas_created_at": datetime.now().isoformat(),
        "sw_version": get_software_version(),
        "meas_by": get_measurement_operator(),
    }
    project_path = Path(project_dir)
    result = MeasurementResult()
    to_save = []

    if not cells_table.empty:
        data = cells_layer.data
        if cleaned_cells_array is not None:
            data = _cells_layer_data(cells_layer.data, cleaned_cells_array)
        metadata = {
            **cells_layer.kwargs["metadata"],
            **run_metadata,
            # Cells-only parameters, recorded so the run can be reproduced
            **{key: config[key] for key in MEASUREMENT_PARAMETER_KEYS},
        }
        _default_file_path(
            metadata, cells_layer.name, project_path, ".cells.png"
        )
        result.cells = MeasuredLayer(
            features=cells_table,
            metadata=metadata,
            data=data if cleaned_cells_array is not None else None,
        )
        to_save.append(
            _snapshot(cells_layer.name, data, metadata, cells_table)
        )

    if not rings_table.empty:
        metadata = {**rings_layer.kwargs["metadata"], **run_metadata}
        _default_file_path(
            metadata, rings_layer.name, project_path, ".rings.png"
        )
        result.rings = MeasuredLayer(features=rings_table, metadata=metadata)
        # The written table has the ring boundaries (RBXY) as last column
        written_table = rings_table
        if "RBXY" in rings_table.columns:
            columns = [c for c in rings_table.columns if c != "RBXY"]
            written_table = rings_table[[*columns, "RBXY"]]
        to_save.append(
            _snapshot(
                rings_layer.name, rings_layer.data, metadata, written_table
            )
        )

    # Annotated image, next to the sample: with the new rings, or the
    # loaded ones when only the cells were measured
    rings_features = rings_table
    if rings_features.empty and rings_layer is not None:
        rings_features = rings_layer.kwargs.get("features", pd.DataFrame())
    measured = result.rings or result.cells
    sample_metadata = (
        measured.metadata
        if measured is not None
        else (rings_layer or cells_layer).kwargs["metadata"]
    )
    sample_path = _sample_path(sample_metadata, project_path)
    draw_annotated = (
        sample_path is not None
        and rings_features is not None
        and not rings_features.empty
    )

    # Saving is one step: its own progress (per layer) is not shown
    total = 2 + int(draw_annotated)
    yield Progress(1, total, "Saving…")
    for _event in save_layers(to_save, project_dir):
        pass

    if draw_annotated:
        yield Progress(2, total, "Drawing the annotated image…")
        result.annotated_path = save_sample_annotated_image(
            sample_path, rings_features
        )
    return result


def _snapshot(name, data, metadata, features) -> LayerSnapshot:
    return LayerSnapshot(
        name=name,
        data=data,
        kwargs={"name": name, "metadata": metadata, "features": features},
    )


def _cells_layer_data(original: np.ndarray, cleaned: np.ndarray) -> np.ndarray:
    """The cleaned cells, in the data type of the original layer data."""
    if original.dtype == bool:
        return cleaned > 0
    if np.issubdtype(original.dtype, np.integer):
        if original.max() <= 1:
            return (cleaned > 0).astype(original.dtype)
        return cleaned.astype(original.dtype)
    return (cleaned > 0).astype(original.dtype)


def _default_file_path(metadata, layer_name, project_path, extension) -> None:
    """Layers without a file path are written to the project root."""
    if metadata.get("file_path") is None:
        sample_name = metadata.get("sample_name", layer_name)
        metadata["file_path"] = str(project_path / f"{sample_name}{extension}")


def _sample_path(metadata: dict, project_path: Path) -> Path | None:
    """The sample stem of a layer, as the writer computes it."""
    stem = metadata.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        return (project_path / stem).resolve()
    if not metadata.get("file_path"):
        return None
    file_path = Path(metadata["file_path"])
    sample_name = metadata.get("sample_name")
    if not isinstance(sample_name, str) or not sample_name.strip():
        sample_name = Path(file_path.stem).stem
    return file_path.parent / sample_name
