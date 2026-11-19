from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

from napari_roxas_ai._crossdating._cross_dating_plotter import (
    build_crossdating_plot_df,
    find_crossdating_files,
    reference_plot_path,
    ring_width_series,
    save_crossdating_plot,
    simplify_string,
)
from napari_roxas_ai._jobs import Message, Progress
from napari_roxas_ai._reader import read_cells_file, read_rings_file
from napari_roxas_ai._reader._crossdating_reader import read_crossdating_file
from napari_roxas_ai._writer import write_single_layer
from napari_roxas_ai._writer._writer import (
    save_sample_annotated_image,
    update_metadata_file,
)

from .._utils._metadata_keys import (
    MEASUREMENT_PARAMETER_KEYS,
    NO_REFERENCE_SERIES,
    SAMPLE_METADATA_PREFIXES,
)
from .._utils._version_utils import (
    get_measurement_operator,
    get_software_version,
)
from ._operations import _cells_layer_data, measure_sample
from ._sample_measurer import save_cwt_visualization_json

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

# Default outmost complete ring year of an undated sample (project convention)
UNDATED_OUTMOST_YEAR = 9999

Rings = tuple[np.ndarray, dict]  # (data, add_kwargs) as read from a file


def measure_samples(
    config: dict[str, Any],
    samples: dict[str, dict[str, Optional[str]]],
    measure_cells: bool = True,
    measure_rings: bool = True,
    update_annotated_images: bool = True,
    update_reference_plots: bool = True,
):
    """
    Measure and update the samples one after the other.

    `samples` maps a sample stem to its {"cells": path, "rings": path}. A
    sample that fails is reported and skipped.
    """
    measurer = BatchMeasurer(
        config,
        measure_cells,
        measure_rings,
        update_annotated_images,
        update_reference_plots,
    )
    total = len(samples)
    processed = 0
    for index, (stem, files) in enumerate(samples.items(), 1):
        name = Path(stem).name
        yield Progress(index - 1, total, f"Measuring {name} ({index}/{total})")
        try:
            tasks = measurer.process(stem, files)
            yield Message(
                f"[Batch Measurements] [{index}/{total}] {name}: {', '.join(tasks)}"
            )
            processed += 1
        except Exception as e:  # noqa: BLE001
            yield Message(
                f"[Batch Measurements] [{index}/{total}] Error processing {name}: {e}"
            )

    if total > 0:
        yield Message(
            f"[Batch Measurements] Completed {processed} of {total} file(s) successfully."
        )


class BatchMeasurer:
    """Measures and updates single samples, with the options of a batch run."""

    def __init__(
        self,
        config: dict[str, Any],
        measure_cells: bool = True,
        measure_rings: bool = True,
        update_annotated_images: bool = True,
        update_reference_plots: bool = True,
    ):
        self.config = dict(config)
        self.measure_cells = measure_cells
        self.measure_rings = measure_rings
        self.update_annotated_images = update_annotated_images
        self.update_reference_plots = update_reference_plots

        # Constant for the whole batch run
        self._sw_version = get_software_version()
        self._meas_by = get_measurement_operator()

        # Crossdating files read so far, as most samples of a project share one
        self._crossdating_dataframes: dict[Path, pd.DataFrame] = {}

    def process(self, stem: str, files: dict[str, Optional[str]]) -> list[str]:
        """Measure and update one sample. Returns what was done (for the log)."""
        # The rings as read or measured, shared by the later steps
        rings = None
        rings_written = False
        tasks = []

        if self.measure_cells or self.measure_rings:
            rings, rings_written, task = self._measure(files)
            tasks.append(task)

        if self.update_annotated_images:
            if rings_written:
                # Writing a rings file already updates the annotated image
                tasks.append("Annotated image")
            else:
                rings = rings or self._read_rings(files)
                tasks.append(self._update_annotated_image(stem, rings))

        if self.update_reference_plots:
            rings = rings or self._read_rings(files)
            tasks.append(self._update_reference_plot(stem, rings))

        return tasks

    def _measure(
        self, files: dict[str, Optional[str]]
    ) -> tuple[Optional[Rings], bool, str]:
        """
        Measure one sample and save the results.

        Returns the rings (data, add_kwargs) of the sample if they were read,
        whether the rings file was written, and what was done (for the log).
        """
        cells_file_path = files["cells"]
        rings_file_path = files["rings"]

        if self.measure_cells and cells_file_path is None:
            return None, False, "Measurement skipped (no cells file)"
        if self.measure_rings and rings_file_path is None:
            return None, False, "Measurement skipped (no rings file)"

        cells_data = cells_add_kwargs = rings_data = rings_add_kwargs = None
        if self.measure_cells:
            cells_data, cells_add_kwargs, _ = read_cells_file(cells_file_path)
        if self.measure_rings:
            rings_data, rings_add_kwargs, _ = read_rings_file(rings_file_path)
        measured = [
            add_kwargs
            for add_kwargs in (cells_add_kwargs, rings_add_kwargs)
            if add_kwargs is not None
        ]

        # Scale of the cells, else the rings; sample type of the first that has one
        self.config["pixels_per_um"] = measured[0]["metadata"][
            "spatial_resolution"
        ]
        self.config["sample_type"] = next(
            (
                (add_kwargs.get("metadata") or {}).get("sample_type")
                for add_kwargs in measured
                if (add_kwargs.get("metadata") or {}).get("sample_type")
            ),
            "conifer",
        )

        if self.measure_cells and self.measure_rings:
            measurement = "both"
        elif self.measure_cells:
            measurement = "cells"
        else:
            measurement = "rings"
        if self.measure_cells:
            cells_array = cells_data.astype("uint8") * 255
        else:
            # The rings need the sample's shape only
            cells_array = np.zeros_like(rings_data, dtype="uint8")
        cells_table, rings_table, cleaned_cells_array, cwt_data = (
            measure_sample(
                self.config,
                measurement,
                cells_array,
                rings_add_kwargs["features"] if self.measure_rings else None,
            )
        )

        # One timestamp per sample, shared by its cells and rings output
        run_metadata = {
            "meas_created_at": datetime.now().isoformat(),
            "sw_version": self._sw_version,
            "meas_by": self._meas_by,
        }

        if self.measure_cells:
            if cleaned_cells_array is not None:
                cells_data = _cells_layer_data(cells_data, cleaned_cells_array)
            cells_add_kwargs["features"] = cells_table
            cells_add_kwargs["metadata"].update(run_metadata)
            # Cells-only parameters, recorded so the run can be reproduced
            for key in MEASUREMENT_PARAMETER_KEYS:
                cells_add_kwargs["metadata"][key] = self.config[key]

            # Save CWT visualization JSON
            sample_name = (
                (cells_add_kwargs.get("metadata") or {}).get("sample_name")
                or (
                    (rings_add_kwargs or {}).get("metadata") or {}
                ).get("sample_name")
                or Path(cells_file_path).stem.replace(".cells", "")
            )
            cwt_data["sample_name"] = sample_name
            cwt_path = Path(cells_file_path).parent / f"{sample_name}.cwt.json"
            save_cwt_visualization_json(cwt_path, cwt_data)

            # Save to file (the file extension in the path argument is ignored)
            write_single_layer(
                path=cells_file_path, data=cells_data, meta=cells_add_kwargs
            )

        if not self.measure_rings:
            return None, False, "Cells"

        rings_add_kwargs["features"] = rings_table
        rings_add_kwargs["metadata"].update(run_metadata)
        write_single_layer(
            path=rings_file_path, data=rings_data, meta=rings_add_kwargs
        )
        task = "Cells, Rings" if self.measure_cells else "Rings"
        return (rings_data, rings_add_kwargs), True, task

    @staticmethod
    def _read_rings(files: dict[str, Optional[str]]) -> Optional[Rings]:
        """Read the rings file of a sample, None if it has none."""
        if files["rings"] is None:
            return None
        rings_data, rings_add_kwargs, _ = read_rings_file(files["rings"])
        return rings_data, rings_add_kwargs

    @staticmethod
    def _rings_features(rings: Optional[Rings]) -> Optional[pd.DataFrame]:
        """The rings table, None if there is none or it is empty."""
        if rings is None:
            return None
        features = rings[1].get("features")
        if features is None or features.empty:
            return None
        return features

    def _update_annotated_image(
        self, stem: str, rings: Optional[Rings]
    ) -> str:
        """
        Redraw the annotated image of a sample from its rings table.

        Returns what was done (for the log).
        """
        features = self._rings_features(rings)
        if features is None:
            return "Annotated image skipped (no rings table)"
        if save_sample_annotated_image(stem, features) is None:
            return "Annotated image skipped (no scan file)"
        return "Annotated image"

    def _crossdating_dataframe(self, path: Path) -> pd.DataFrame:
        """Read a crossdating file and add its average series (cached)."""
        if path not in self._crossdating_dataframes:
            dataframe = read_crossdating_file(str(path))
            dataframe["average"] = dataframe.mean(axis=1, skipna=True)
            self._crossdating_dataframes[path] = dataframe
        return self._crossdating_dataframes[path]

    def _update_reference_plot(self, stem: str, rings: Optional[Rings]) -> str:
        """
        Redraw the crossdating reference plot of a sample.

        The sample is plotted as dated by the outmost complete ring year of its
        metadata, against its stored reference series. A sample without one
        ("NA") is plotted against the series that matches it best, as found by
        the automatic alignment of the visual crossdating. Samples without a
        crossdating file are skipped.

        Returns what was done (for the log).
        """
        name = Path(stem).name
        features = self._rings_features(rings)
        if (
            features is None
            or "YEAR" not in features.columns
            or "cells_above" not in features.columns
        ):
            return "Reference plot skipped (no rings table)"

        rings_data, rings_add_kwargs = rings
        metadata = rings_add_kwargs.get("metadata") or {}
        if "spatial_resolution" not in metadata:
            return "Reference plot skipped (no spatial resolution)"

        crossdating_files = find_crossdating_files(Path(stem).parent)
        if not crossdating_files:
            if "reference_series" not in metadata:
                metadata["reference_series"] = NO_REFERENCE_SERIES
                update_metadata_file(
                    f"{stem}.metadata.json",
                    metadata,
                    SAMPLE_METADATA_PREFIXES,
                )
            return "Reference plot skipped (no crossdating file)"

        width_series = ring_width_series(
            features,
            metadata.get("rings_outmost_complete_year", UNDATED_OUTMOST_YEAR),
            rings_data.shape[1],
            metadata["spatial_resolution"],
        )

        plot = self._stored_reference_plot_df(
            metadata.get("reference_series"), crossdating_files, width_series
        )
        sample_name = metadata.get("sample_name") or name
        best_match = plot is None
        if best_match:
            plot = self._best_matching_plot_df(
                crossdating_files, sample_name, width_series
            )
        if plot is None:
            if (
                metadata.get("reference_series") != NO_REFERENCE_SERIES
                or "reference_series" not in metadata
            ):
                metadata["reference_series"] = NO_REFERENCE_SERIES
                update_metadata_file(
                    f"{stem}.metadata.json",
                    metadata,
                    SAMPLE_METADATA_PREFIXES,
                )
            return "Reference plot skipped (no matching reference series)"

        reference_name, plot_df = plot
        if metadata.get("reference_series") != reference_name:
            metadata["reference_series"] = reference_name
            update_metadata_file(
                f"{stem}.metadata.json",
                metadata,
                SAMPLE_METADATA_PREFIXES,
            )

        save_crossdating_plot(
            plot_df,
            reference_name,
            sample_name,
            reference_plot_path(Path(stem).parent, sample_name),
        )
        if best_match:
            return f"Reference plot ({reference_name}, best match)"
        return f"Reference plot ({reference_name})"

    def _stored_reference_plot_df(
        self,
        reference_series: Optional[str],
        crossdating_files: list[Path],
        width_series: pd.Series,
    ) -> Optional[tuple[str, pd.DataFrame]]:
        """Plot data against the stored reference series, if it is found."""
        if (
            not isinstance(reference_series, str)
            or reference_series == NO_REFERENCE_SERIES
        ):
            return None
        for path in crossdating_files:
            dataframe = self._crossdating_dataframe(path)
            if reference_series in dataframe.columns:
                return reference_series, build_crossdating_plot_df(
                    dataframe, reference_series, width_series
                )
        return None

    def _best_matching_plot_df(
        self,
        crossdating_files: list[Path],
        sample_name: str,
        width_series: pd.Series,
    ) -> Optional[tuple[str, pd.DataFrame]]:
        """
        Plot data against the reference series whose name matches the sample name.
        """
        for path in crossdating_files:
            dataframe = self._crossdating_dataframe(path)
            column_names = sorted(
                dataframe.drop("average", axis=1, errors="ignore").columns,
                key=lambda x: len(x),
                reverse=True,
            )
            matching_columns = [
                col
                for col in column_names
                if simplify_string(col) in simplify_string(sample_name)
            ]
            if matching_columns:
                col = matching_columns[0]
                return col, build_crossdating_plot_df(
                    dataframe, col, width_series
                )
        return None
