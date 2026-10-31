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

    def __init__(self, layers: list = None):
        super().__init__()
        self.layers = layers or []

    def run(self):
        project_dir = settings.get("project_directory")
        if not project_dir:
            raise RuntimeError("project_directory is not configured")

        project_dir = Path(project_dir)

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

            # Keep kwargs metadata in sync (optional but helps later)
            meta["file_path"] = str(out_path)
            kwargs["metadata"] = meta

            # Manual saves skip the annotated image: it is redrawn at the
            # workflow checkpoints (segmentation, year updates, crossdating,
            # measurements) instead
            try:
                write_single_layer(
                    str(out_path),
                    data,
                    kwargs,
                    progress_callback=make_callback(i),
                    save_annotated=False,
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

        # The running save, if any (see _save_layers)
        self.worker = None
        self.worker_thread = None

        self._save_selected_layers_button = PushButton(
            text="Save Selected Layers"
        )
        self._save_selected_layers_button.changed.connect(
            lambda: self._save_layers(how="selected")
        )

        self._save_all_layers_button = PushButton(
            text="Save All Layers"
        )
        self._save_all_layers_button.changed.connect(
            lambda: self._save_layers(how="all")
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

    def _save_layers(self, how: str):
        """
        Save selected or all layers to their original location.

        "all" only saves the cells and rings layers with unsaved changes
        (metadata "_is_dirty"; layers without the flag count as modified).
        The outcome is reported to the user; when a save runs, self.worker and
        self.worker_thread hold it, otherwise they are None.
        """
        # Drop references to a previous (already finished) save run
        self.worker = None
        self.worker_thread = None

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
            self._progress_bar.visible = False
            show_info("No rings or cells layers to save")
            return

        if how == "all":
            layers_to_save = [
                layer
                for layer in layers_to_save
                if layer.metadata.get("_is_dirty", True)
            ]
            if not layers_to_save:
                self._progress_bar.visible = False
                show_info("No unsaved changes in cells or rings layers.")
                return

        saved_types = []
        for layer in layers_to_save:
            if layer.name.endswith(cells_ext):
                saved_types.append("cells")
            elif layer.name.endswith(rings_ext):
                saved_types.append("rings")
            else:
                saved_types.append(layer.name)
        if len(saved_types) == 1:
            saved_message = f"Saved {saved_types[0]} layer."
        else:
            saved_message = f"Saved {' and '.join(saved_types)} layers."

        # Disable the run button while processing
        self._save_selected_layers_button.enabled = False
        self._save_all_layers_button.enabled = False

        self._progress_bar.value = 0
        self._progress_bar.visible = True

        self.worker_thread = QThread()
        self.worker = Worker(layers_to_save)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        # Connect progress signal
        self.worker.progress.connect(self._update_progress)

        def on_saved():
            for layer in layers_to_save:
                layer.metadata["_is_dirty"] = False
            show_info(saved_message)

        # Mark the layers as saved once they are written
        self.worker_thread.finished.connect(on_saved)

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
