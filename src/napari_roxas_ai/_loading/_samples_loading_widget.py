"""
Widget for preparing sample images and metadata for ROXAS analysis.
"""

import glob
from pathlib import Path
from typing import TYPE_CHECKING

import napari.layers
from magicgui.widgets import Container, ProgressBar, PushButton, Select
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QFileDialog,
)

from napari_roxas_ai._reader import (
    read_cells_file,
    read_rings_file,
    read_scan_file,
)
from napari_roxas_ai._settings._settings_manager import SettingsManager
from napari_roxas_ai._utils._fix_sample_stem_path import fix_sample_stem_paths_in_project
from napari_roxas_ai.shortcuts import (
    install_wasd_shortcuts,
    has_shortcuts_applied,
    mark_shortcuts_applied,
)
from napari_roxas_ai._edition._rings_layer_editor import RingsLayerEditorWidget



if TYPE_CHECKING:
    import napari

settings = SettingsManager()


class Worker(QObject):
    """Worker thread for loading samples."""

    finished = Signal()
    progress = Signal(int, int)  # current, total
    layer_data = Signal(tuple)

    def __init__(
        self,
        samples_stem_paths: list[str],
        project_directory: str,
    ):
        super().__init__()
        self.samples_stem_paths = samples_stem_paths
        self.project_directory = project_directory

        self.scan_file_extension = "".join(
            settings.get("file_extensions.scan_file_extension")
        )
        self.cells_file_extension = "".join(
            settings.get("file_extensions.cells_file_extension")
        )
        self.rings_file_extension = "".join(
            settings.get("file_extensions.rings_file_extension")
        )

    def run(self):

        total = len(self.samples_stem_paths) * 3

        project_dir = Path(self.project_directory).resolve()

        for i, sample_stem_path in enumerate(self.samples_stem_paths):

            sample_path = Path(sample_stem_path).resolve()
            try:
                relative_stem = sample_path.relative_to(project_dir)
            except Exception as e:
                relative_stem = sample_path


            scan_file_path = f"{sample_stem_path}{self.scan_file_extension}"
            if Path(scan_file_path).exists():
                self.progress.emit(i, total)
                scan_data, scan_add_kwargs, scan_layer_type = read_scan_file(
                    scan_file_path
                )

                scan_add_kwargs["metadata"]["sample_stem_path"] = str(relative_stem)

                self.layer_data.emit(
                    (scan_data, scan_add_kwargs, scan_layer_type)
                )

            cells_file_path = f"{sample_stem_path}{self.cells_file_extension}"
            if Path(cells_file_path).exists():
                self.progress.emit(i + 1, total)
                cells_data, cells_add_kwargs, cells_layer_type = read_cells_file(
                    cells_file_path
                )

                cells_add_kwargs["metadata"]["sample_stem_path"] = str(relative_stem)

                self.layer_data.emit(
                    (cells_data, cells_add_kwargs, cells_layer_type)
                )

            rings_file_path = f"{sample_stem_path}{self.rings_file_extension}"
            if Path(rings_file_path).exists():
                self.progress.emit(i + 2, total)
                rings_data, rings_add_kwargs, rings_layer_type = read_rings_file(
                    rings_file_path
                )

                rings_add_kwargs["metadata"]["sample_stem_path"] = str(relative_stem)

                self.layer_data.emit(
                    (rings_data, rings_add_kwargs, rings_layer_type)
                )

        self.progress.emit(total, total)
        self.finished.emit()


class SamplesLoadingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        try:
            if not has_shortcuts_applied(viewer):
                install_wasd_shortcuts(viewer)
                mark_shortcuts_applied(viewer)

        except Exception as e:
            print("Shortcut installation failed:", e)

        # Directory selection
        self.project_directory = settings.get("project_directory")

        # --- FIX OUTDATED METADATA STEMS (silent, filesystem-truth based) ---
        if self.project_directory:
            fix_sample_stem_paths_in_project(self.project_directory)

        self._project_dialog_button = PushButton(
            text=f"Project Directory: {self.project_directory or 'Not set'}"
        )
        self._project_dialog_button.changed.connect(self._open_project_dialog)

        # sample selection container
        self._samples_selection_container = Container(
            widgets=[], labels=False, layout="vertical", visible=True
        )

        # Reverse selection button (initially hidden)
        self._reverse_selection_button = PushButton(
            text="Reverse Selection", visible=True
        )
        self._reverse_selection_button.changed.connect(
            self._reverse_sample_selection
        )

        # Select all button
        self._select_all_button = PushButton(text="Select All", visible=True)
        self._select_all_button.changed.connect(self._select_all_samples)

        # Container for buttons side by side
        self._buttons_container = Container(
            widgets=[self._select_all_button, self._reverse_selection_button],
            labels=False,
            layout="horizontal",
            visible=True,
        )

        # Load selected samples button
        self._load_samples_button = PushButton(
            text="Load Selected Samples", visible=True
        )
        self._load_samples_button.changed.connect(self._load_selected_samples)

        # Add a progress bar with a description
        self._progress_bar = ProgressBar(
            value=0, min=0, max=100, visible=False, label="Progress"
        )

        # Append all widgets to the container
        self.extend(
            [
                self._project_dialog_button,
                self._samples_selection_container,
                self._buttons_container,
                self._load_samples_button,
                self._progress_bar,
            ]
        )
        self._refresh_samples_list()

        # --- BACKGROUND MONITORING INITIALIZATION ---
        # Initialize the RingsLayerEditorWidget to start tracking ring years in the background.
        # This ensures that 'Rings Years' labels appear even if the user hasn't opened the editor widget yet.
        # A small delay is used to allow the napari viewer and other widgets to settle.
        from qtpy.QtCore import QTimer
        def _init_rings_editor():
            # Check if an instance of the editor widget already exists in any dock window
            # to avoid redundant background monitors.
            found = False
            try:
                # Iterate through all dock widgets using public-compatible APIs.
                # Accessing Window._qt_window and findChildren is a safer alternative
                # to using internal viewer attributes directly.
                qt_window = getattr(self._viewer.window, "_qt_window", None)
                if qt_window is not None:
                    from qtpy.QtWidgets import QWidget
                    for dock in qt_window.findChildren(QWidget):
                        if "RingsLayerEditorWidget" in str(type(dock)):
                            found = True
                            break
            except Exception:
                pass
            
            if not found:
                try:
                    # Instantiate the widget. Even if not docked, it connects to viewer events
                    # to manage the 'Rings Years' layer. A reference is kept on the loader
                    # widget to prevent the editor from being garbage collected.
                    self._rings_editor = RingsLayerEditorWidget(self._viewer)
                except Exception as e:
                    print(f"Failed to auto-initialize RingsLayerEditorWidget: {e}")

        QTimer.singleShot(1000, _init_rings_editor)

    def refresh_from_settings(self):
        """
        Pick up settings changed while this widget was open.

        The project directory is copied into this widget when it is built and
        shown on the button, so both are re-read, and the samples list is
        rebuilt from the new directory.
        """
        self.project_directory = settings.get("project_directory")
        self._project_dialog_button.text = (
            f"Project Directory: {self.project_directory or 'Not set'}"
        )
        self._refresh_samples_list()

    def _open_project_dialog(self):
        """Open sample dialog to select project directory and refresh samples list."""
        directory = QFileDialog.getExistingDirectory(
            parent=None,
            caption="Select Project Directory",
            directory=self.project_directory,
        )
        if directory:
            self.project_directory = directory
            settings.set("project_directory", directory)
            self._project_dialog_button.text = (
                f"Project Directory: {directory}"
            )
            self._refresh_samples_list()

    def _refresh_samples_list(self):
        """Refresh the list of samples from the project directory."""
        # Reset selected samples list
        self.selected_samples = []

        # Get all image samples in project directory
        self.source_samples = []

        if self.project_directory and Path(self.project_directory).exists():

            # Gather all samples with supported image extensions
            ext = settings.get("file_extensions.metadata_file_extension")[0]
            self.source_samples.extend(
                glob.glob(
                    str(Path(self.project_directory) / "**" / f"*{ext}.*"),
                    recursive=True,
                )
            )
            self.source_samples.extend(
                glob.glob(
                    str(
                        Path(self.project_directory)
                        / "**"
                        / f"*{ext.upper()}.*"
                    ),
                    recursive=True,
                )
            )

            # Filter based on whether to include already processed samples

            # Sort samples for consistent display
            self.source_samples = sorted(set(self.source_samples))

            # Keep stem path only
            self.source_samples = [
                a.split(ext)[0] for a in self.source_samples
            ]

        # Update the sample selection widget if it's visible
        self._update_sample_selection_widget()

    def _update_sample_selection_widget(self):
        """Update the sample selection widget with current source files."""
        # Clear the container
        self._samples_selection_container.clear()
        # Create a new sample selection widget
        if self.source_samples:
            # Convert absolute paths to display names (relative to project directory)
            display_names = []
            self._path_mapping = {}  # Map display names to absolute paths

            for sample_path in self.source_samples:
                # Use relative path if possible, otherwise use base name
                try:
                    rel_path = str(
                        Path(sample_path).relative_to(self.project_directory)
                    )
                    display_names.append(rel_path)
                    self._path_mapping[rel_path] = sample_path
                except ValueError:
                    basename = Path(sample_path).name
                    display_names.append(basename)
                    self._path_mapping[basename] = sample_path

            # Create a select widget with multiple selection
            self._sample_select_widget = Select(
                choices=display_names,
                allow_multiple=True,
                label="Select Files to Process",
            )

            # Add the widget to the container
            self._samples_selection_container.append(
                self._sample_select_widget
            )

            # Make container and buttons visible
            self._samples_selection_container.visible = True
            self._buttons_container.visible = True
            self._load_samples_button.visible = True
        else:
            # Hide container and buttons if no files
            self._samples_selection_container.visible = False
            self._buttons_container.visible = False
            self._load_samples_button.visible = False

    def _reverse_sample_selection(self):
        """Reverse the current sample selection."""
        if hasattr(self, "_sample_select_widget"):
            # Get currently selected files
            current_selection = set(self._sample_select_widget.value)

            # Create a new selection with all files except the current selection
            all_display_names = set(self._sample_select_widget.choices)
            new_selection = list(all_display_names - current_selection)

            # Update the widget
            self._sample_select_widget.value = new_selection

    def _select_all_samples(self):
        """Select all available samples."""
        if hasattr(self, "_sample_select_widget"):
            # Get all available files
            all_display_names = list(self._sample_select_widget.choices)

            # Update the widget to select all files
            self._sample_select_widget.value = all_display_names

    def _load_selected_samples(self):
        """Load selected samples and add them to the viewer."""
        if not self.project_directory or not hasattr(self, "_sample_select_widget"):
            return

        # Disable the run button while processing
        self._load_samples_button.enabled = False

        self.worker_thread = QThread()
        self.worker = Worker(
            samples_stem_paths=[
                str(Path(self.project_directory) / sample)
                for sample in self._sample_select_widget.value
            ],
            project_directory=self.project_directory,
        )
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
            lambda: setattr(self._load_samples_button, "enabled", True)
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._load_samples_button, "visible", True)
        )
        self.worker_thread.finished.connect(
            lambda: setattr(self._progress_bar, "visible", False)
        )

        # Connect layer data signal to add layers to the viewer
        self.worker.layer_data.connect(self._add_loaded_layer)


        # Run the analysis in a separate thread
        self.worker_thread.start()

    def _add_loaded_layer(self, layer_data_tuple):
        data, add_kwargs, layer_type = layer_data_tuple

        layer = napari.layers.Layer.create(data, add_kwargs, layer_type)

        original_path = add_kwargs.get("metadata", {}).get("path")

        if original_path:
            print(f"[Loader] Setting file_path = {original_path}")
            layer.metadata["file_path"] = original_path
        else:
            print("[Loader] WARNING: no file_path in reader metadata")

        self._viewer.add_layer(layer)

    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int(100 * current / total)
            self._progress_bar.value = percentage
            self._load_samples_button.visible = False
            self._progress_bar.visible = True
