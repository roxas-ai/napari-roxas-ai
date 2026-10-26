"""
Widget for preparing sample images and metadata for ROXAS analysis.
"""
from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers
from magicgui.widgets import Container, ProgressBar, PushButton
from napari.utils.notifications import show_info
from qtpy.QtCore import QObject, QThread, Signal

from napari_roxas_ai._settings._settings_manager import SettingsManager
from napari_roxas_ai._writer import write_single_layer

if TYPE_CHECKING:
    import napari

settings = SettingsManager()


class Worker(QObject):
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(
        self,
        layers: list = None,
        all_layers: list = None,
    ):
        super().__init__()
        self.layers = layers or []
        self.all_layers = all_layers or []

    def run(self):
        project_dir = settings.get("project_directory")
        if not project_dir:
            raise RuntimeError("project_directory is not configured")

        project_dir = Path(project_dir)

        rings_ext = settings.get("file_extensions.rings_file_extension")[0]
        cells_ext = settings.get("file_extensions.cells_file_extension")[0]
        scan_ext = settings.get("file_extensions.scan_file_extension")[0]

        total = len(self.layers)

        for i, layer in enumerate(self.layers):
            self.progress.emit(int(100 * i / total) if total > 0 else 0, 100)

            data, kwargs, layer_type = layer.as_layer_data_tuple()
            meta = (kwargs.get("metadata") or {}).copy()

            def make_callback(layer_idx):
                def cb(fraction):
                    if total > 0:
                        pct = int(100 * (layer_idx + fraction) / total)
                        self.progress.emit(pct, 100)
                return cb

            # Prefer explicit per-layer file_path (like SingleSampleMeasurementsWidget)
            file_path = meta.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                out_path = Path(file_path)
            else:
                # Fallback: build from sample_stem_path or sample_name
                stem = meta.get("sample_stem_path")
                sample_name = meta.get("sample_name") or layer.name

                if isinstance(stem, str) and stem.strip():
                    out_path = project_dir / stem
                else:
                    # Last resort: put it in project root under sample_name
                    out_path = project_dir / str(sample_name)

            # Ensure parent exists
            out_path.parent.mkdir(parents=True, exist_ok=True)

            # If this is a rings layer, check if matching cells or scan data is available in memory
            if layer.name.endswith(rings_ext):
                stem = meta.get("sample_stem_path")
                sample_name = meta.get("sample_name") or layer.name
                if isinstance(sample_name, str) and sample_name.endswith(rings_ext):
                    sample_name = sample_name[: -len(rings_ext)]

                cells_data = None
                scan_data = None
                for other_layer in self.all_layers:
                    other_meta = getattr(other_layer, "metadata", None) or {}
                    other_stem = other_meta.get("sample_stem_path")
                    other_sample_name = (
                        other_meta.get("sample_name") or other_layer.name
                    )

                    is_match = False
                    if other_layer.name.endswith(cells_ext):
                        if isinstance(other_sample_name, str) and other_sample_name.endswith(cells_ext):
                            other_sample_name = other_sample_name[: -len(cells_ext)]
                        if (
                            (stem and other_stem and stem == other_stem)
                            or (
                                sample_name
                                and other_sample_name
                                and sample_name == other_sample_name
                            )
                        ):
                            cells_data = getattr(other_layer, "data", None)

                    elif other_layer.name.endswith(scan_ext):
                        if isinstance(other_sample_name, str) and other_sample_name.endswith(scan_ext):
                            other_sample_name = other_sample_name[: -len(scan_ext)]
                        if (
                            (stem and other_stem and stem == other_stem)
                            or (
                                sample_name
                                and other_sample_name
                                and sample_name == other_sample_name
                            )
                        ):
                            scan_data = getattr(other_layer, "data", None)

                if cells_data is not None:
                    meta["cells_data"] = cells_data
                if scan_data is not None:
                    meta["scan_data"] = scan_data

            # Keep kwargs metadata in sync (optional but helps later)
            meta["file_path"] = str(out_path)
            kwargs["metadata"] = meta

            try:
                write_single_layer(
                    str(out_path),
                    data,
                    kwargs,
                    progress_callback=make_callback(i),
                )
            except TypeError:
                write_single_layer(
                    str(out_path),
                    data,
                    kwargs,
                )

        self.progress.emit(100, 100)
        self.finished.emit()


class SamplesSavingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._save_selected_layers_button = PushButton(
            text="Save Selected Layers"
        )
        self._save_selected_layers_button.changed.connect(
            lambda: self._save_layers_with_feedback(how="selected")
        )

        self._save_all_layers_button = PushButton(
            text="Save All Layers"
        )
        self._save_all_layers_button.changed.connect(
            lambda: self._save_layers_with_feedback(how="all")
        )

        # Add a progress bar with a description
        self._progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Progress"
        )

        # Append all widgets to the container
        self.extend(
            [
                self._save_selected_layers_button,
                self._save_all_layers_button,
                self._progress_bar,
            ]
        )

    def _save_layers_with_feedback(self, how: str):
        """Save layers and report the outcome to the user."""
        # Drop references to a previous (already finished) save run
        self.worker = None
        self.worker_thread = None
        self._save_layers(how=how)
        if self.worker is None or self.worker_thread is None:
            # Nothing to save: no worker was started
            show_info("No rings or cells layers to save")
            return

        n_layers = len(self.worker.layers)
        self.worker_thread.finished.connect(
            lambda: show_info(f"{n_layers} layer(s) saved successfully")
        )

    def _save_layers(self, how: str):
        """Save selected or all layers to their original location."""
        rings_ext = settings.get("file_extensions.rings_file_extension")[0]
        cells_ext = settings.get("file_extensions.cells_file_extension")[0]
        allowed_exts = (rings_ext, cells_ext)

        if how == "selected":
            candidate_layers = self._viewer.layers.selection
        elif how == "all":
            candidate_layers = self._viewer.layers
        else:
            candidate_layers = []

        layers_to_save = [
            layer
            for layer in candidate_layers
            if any(layer.name.endswith(ext) for ext in allowed_exts)
        ]

        if not layers_to_save:
            return

        # Disable the run button while processing
        self._save_selected_layers_button.enabled = False
        self._save_all_layers_button.enabled = False

        self.worker_thread = QThread()
        self.worker = Worker(layers_to_save)
        self.worker.all_layers = list(self._viewer.layers)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Connect progress signal
        self.worker.progress.connect(self._update_progress)

        # Reset buttons and progress after processing
        self.worker_thread.finished.connect(
            lambda: setattr(self._save_selected_layers_button, "enabled", True)
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._save_all_layers_button, "enabled", True)
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._progress_bar, "visible", False)
        )

        # Run the analysis in a separate thread
        self.worker_thread.start()

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self._progress_bar.value = percentage
            self._progress_bar.visible = True
