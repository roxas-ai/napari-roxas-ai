"""
Widget for saving the cells and rings layers of the samples.
"""

from typing import TYPE_CHECKING

from magicgui.widgets import Container

from napari_roxas_ai._job_runner import JobButton, JobHandle

from ._actions import start_saving

if TYPE_CHECKING:
    import napari


class SamplesSavingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self._save_selected_layers_button = JobButton(
            text="Save Selected Layers",
            start=lambda: self._save(
                self._viewer.layers.selection,
                only_unsaved=False,
                other_button=self._save_all_layers_button,
            ),
        )
        self._save_all_layers_button = JobButton(
            text="Save All Layers",
            start=lambda: self._save(
                self._viewer.layers,
                only_unsaved=True,
                other_button=self._save_selected_layers_button,
            ),
        )
        self.extend(
            [self._save_selected_layers_button, self._save_all_layers_button]
        )

    def _save(
        self, layers, only_unsaved: bool, other_button: JobButton
    ) -> JobHandle | None:
        """Start the save; the other button is locked meanwhile."""
        handle = start_saving(layers, only_unsaved=only_unsaved)
        if handle is None:
            return None
        other_button.enabled = False

        def unlock(_handle) -> None:
            other_button.enabled = True

        handle.connect(on_finished=unlock)
        return handle
