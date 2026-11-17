from typing import TYPE_CHECKING, Optional
import napari.layers
from magicgui.widgets import (
    CheckBox,
    Container,
    PushButton,
)
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QInputDialog
from napari_roxas_ai._job_runner import JobButton, JobHandle
from napari_roxas_ai._settings import SettingsManager
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from napari_roxas_ai.shortcuts import prevent_duplicate_dock_if_already_present
from ._actions import start_measuring


settings = SettingsManager()

if TYPE_CHECKING:
    import napari


class SingleSampleMeasurementsWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self._viewer = viewer
        prevent_duplicate_dock_if_already_present(
            self, viewer, "9A - Measure cells & rings (individual image)"
        )
        self._selected_sample = None

        # Cells measurement checkbox
        self._measure_cells_checkbox = CheckBox(
            value=True, label="Measure Cells"
        )

        # Rings measurement checkbox
        self._measure_rings_checkbox = CheckBox(
            value=True, label="Measure Rings"
        )

        # Button to open settings focused on measurements
        self._check_settings_button = PushButton(text="Check Settings")
        self._check_settings_button.changed.connect(self._open_settings)

        # Create a button to launch the analysis (shows its progress; the
        # measured layers are written at the end, so it cannot be cancelled)
        self._run_analysis_button = JobButton(
            text="Measure", start=self._run_analysis, cancellable=False
        )

        # Append the widgets to the container
        self.extend(
            [
                self._measure_cells_checkbox,
                self._measure_rings_checkbox,
                self._check_settings_button,
                self._run_analysis_button,
            ]
        )

    def _open_settings(self):
        """Open SettingsWidget as a floating window centered on the active screen with measurements expanded."""
        SettingsWidget.open_floating(self._viewer, section_name="measurements")

    def _find_layer(
        self, layer_type: type, extension_key: str
    ) -> Optional["napari.layers.Layer"]:
        """Return the layer of the selected sample (or the first match if none is selected)."""
        extension = settings.get(f"file_extensions.{extension_key}")[0]
        sample_name = self._selected_sample
        for layer in self._viewer.layers:
            if not isinstance(layer, layer_type):
                continue
            if sample_name is not None:
                if layer.name == f"{sample_name}{extension}":
                    return layer
            elif layer.name.endswith(extension):
                return layer
        return None

    @property
    def _cells_layer(self) -> Optional["napari.layers.Labels"]:
        return self._find_layer(napari.layers.Labels, "cells_file_extension")

    @property
    def _rings_layer(self) -> Optional["napari.layers.Labels"]:
        return self._find_layer(napari.layers.Labels, "rings_file_extension")

    @property
    def _scan_layer(self) -> Optional["napari.layers.Image"]:
        return self._find_layer(napari.layers.Image, "scan_file_extension")

    def _loaded_sample_names(self) -> list:
        """Names of the samples that have a cells or rings layer loaded."""
        extensions = [
            settings.get("file_extensions.cells_file_extension")[0],
            settings.get("file_extensions.rings_file_extension")[0],
        ]
        names = set()
        for layer in self._viewer.layers:
            if not isinstance(layer, napari.layers.Labels):
                continue
            for extension in extensions:
                if layer.name.endswith(extension):
                    names.add(layer.name[: -len(extension)])
        return sorted(names)

    def _choose_sample(self) -> bool:
        """
        Select the sample to measure, asking the user if several are loaded.

        Returns False if the user cancelled the choice.
        """
        names = self._loaded_sample_names()
        if len(names) <= 1:
            self._selected_sample = names[0] if names else None
            return True

        name, ok = QInputDialog.getItem(
            self.native,
            "Select sample",
            "Several samples are loaded. Which one should be measured?",
            names,
            0,
            False,
        )
        if not ok:
            return False
        self._selected_sample = name
        return True

    def _run_analysis(self) -> JobHandle | None:
        """Measure the selected sample; None when nothing was started."""
        if not self._choose_sample():
            return None

        measure_cells = self._measure_cells_checkbox.value
        measure_rings = self._measure_rings_checkbox.value
        if not measure_cells and not measure_rings:
            show_info("Please select at least one measurement (cells or rings).")
            return None

        cells_layer = self._cells_layer
        if measure_cells and cells_layer is None:
            show_info(
                "Cells layer not found in the viewer. Please load the cells layer first or uncheck 'Measure Cells'."
            )
            return None

        rings_layer = self._rings_layer
        if measure_rings and rings_layer is None:
            show_info(
                "Rings layer not found in the viewer. Please load the rings layer first or uncheck 'Measure Rings'."
            )
            return None

        if measure_cells and measure_rings:
            measurement = "both"
        elif measure_cells:
            measurement = "cells"
        else:
            measurement = "rings"

        # Scale and sample type: from the cells layer when measuring cells
        reference_layer = cells_layer if measure_cells else rings_layer
        config = {
            "pixels_per_um": reference_layer.metadata["spatial_resolution"],
            "cluster_dbl_cwt_threshold": settings.get(
                "measurements.cluster_dbl_cwt_threshold"
            ),
            "smoothing_kernel_size": settings.get(
                "measurements.cells_smoothing_kernel_size"
            ),
            "relwidth_cwt_integration": settings.get(
                "measurements.relwidth_cwt_integration"
            ),
            "tangential_angle": settings.get(
                "measurements.cells_tangential_angle"
            ),
            "lower_limit_cwt_iqr_multiplier": settings.get(
                "measurements.lower_limit_cwt_iqr_multiplier"
            ),
            "upper_limit_cwt_iqr_multiplier": settings.get(
                "measurements.upper_limit_cwt_iqr_multiplier"
            ),
            "opposite_cwt_ratio_limit": settings.get(
                "measurements.opposite_cwt_ratio_limit"
            ),
            "adjacent_cwt_ratio_limit": settings.get(
                "measurements.adjacent_cwt_ratio_limit"
            ),
            "sample_type": reference_layer.metadata["sample_type"],
        }

        return start_measuring(
            measurement,
            config,
            cells_layer,
            rings_layer,
            settings.get("project_directory"),
        )
