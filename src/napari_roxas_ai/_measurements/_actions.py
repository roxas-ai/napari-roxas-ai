from typing import Any

import numpy as np
import pandas as pd
from napari.utils.notifications import show_info

from napari_roxas_ai._job_runner import JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag
from napari_roxas_ai._saving._operations import LayerSnapshot

from ._operations import MeasurementResult, measure_and_save


def start_measuring(
    measurement: str,
    config: dict[str, Any],
    cells_layer,
    rings_layer,
    project_dir: str,
) -> JobHandle:
    """
    Measure the sample's "cells", "rings" or "both" in the background.
    """
    if measurement == "both":
        cells_input, rings_input = cells_layer, rings_layer
        cells_array = cells_layer.data
        rings_table = rings_layer.features
        cells_table = cells_layer.features
    elif measurement == "cells":
        cells_input = cells_layer
        cells_array = cells_layer.data
        # The ring boundaries help measuring the cells, when available
        if rings_layer is not None and not rings_layer.features.empty:
            rings_input = rings_layer
            rings_table = rings_layer.features
        else:
            rings_input = None
            rings_table = pd.DataFrame()
        cells_table = pd.DataFrame()
    elif measurement == "rings":
        cells_input, rings_input = None, rings_layer
        cells_array = np.zeros_like(rings_layer.data)
        rings_table = rings_layer.features
        cells_table = pd.DataFrame()
    else:
        raise ValueError(f"Unknown measurement {measurement!r}")

    def on_measured(result: MeasurementResult) -> None:
        _apply(result, cells_input, rings_input)

    return get_runner().submit(
        Job(
            "Measuring",
            measure_and_save,
            args=(
                config,
                measurement,
                LayerSnapshot.of(cells_input) if cells_input else None,
                LayerSnapshot.of(rings_input) if rings_input else None,
                cells_array,
                rings_table,
                cells_table,
                project_dir,
            ),
            tags={JobTag.MEASUREMENTS},
        ),
        on_result=on_measured,
    )


def _apply(result: MeasurementResult, cells_layer, rings_layer) -> None:
    """Update the measured layers with what was written."""
    if result.cells is not None:
        cells_layer.features = result.cells.features
        if result.cells.data is not None:
            cells_layer.data = result.cells.data
        cells_layer.metadata.update(result.cells.metadata)
        # Written with the measurements: nothing left to save
        cells_layer.metadata["_is_dirty"] = False
        show_info(f"Cells exported to: {result.cells.metadata['file_path']}")

    if result.rings is not None:
        rings_layer.features = result.rings.features
        rings_layer.metadata.update(result.rings.metadata)
        rings_layer.metadata["_is_dirty"] = False
        show_info(f"Rings exported to: {result.rings.metadata['file_path']}")

    if result.annotated_path is not None:
        show_info(f"Annotated image exported to: {result.annotated_path}")
    show_info("Measurements completed and saved.")
