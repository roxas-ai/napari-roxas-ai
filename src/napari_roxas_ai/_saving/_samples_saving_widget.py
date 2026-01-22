"""
Widget for preparing sample images and metadata for ROXAS analysis.
"""
from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers
from magicgui.widgets import Container, ProgressBar, PushButton
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
    ):
        super().__init__()
        self.layers = layers

    def run(self):
        project_dir = settings.get("project_directory")
        if not project_dir:
            raise RuntimeError("project_directory is not configured")

        project_dir = Path(project_dir)

        total = len(self.layers)

        for i, layer in enumerate(self.layers):
            self.progress.emit(i, total)

            data, kwargs, layer_type = layer.as_layer_data_tuple()
            meta = kwargs.get("metadata") or {}

            stem = meta.get("sample_stem_path")
            if not stem:
                raise RuntimeError(f"No sample_stem_path for layer {layer.name}")

            out_path = project_dir / stem

            write_single_layer(
                str(out_path),
                data,
                kwargs,
            )

        self.progress.emit(total, total)
        self.finished.emit()


class SamplesSavingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._save_selected_layers_button = PushButton(
            text="Save Selected Layers", visible=True
        )
        self._save_selected_layers_button.changed.connect(
            lambda: self._save_layers(how="selected")
        )

        self._save_all_layers_button = PushButton(
            text="Save All Layers", visible=True
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
        """Save selected or all layers to their original location."""

        # Disable the run button while processing
        self._save_selected_layers_button.enabled = False
        self._save_all_layers_button.enabled = False

        if how == "selected":
            layers_to_save = self._viewer.layers.selection
        elif how == "all":
            layers_to_save = self._viewer.layers

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
