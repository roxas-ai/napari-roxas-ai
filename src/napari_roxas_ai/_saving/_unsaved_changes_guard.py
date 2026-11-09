"""
Ask for confirmation before layers with unsaved cells/rings changes are removed.

napari offers no cancellable hook before layers are removed, but every removal
from the UI (delete button, Delete key, shortcut) goes through
``viewer.layers.remove_selected()``, so that method is guarded per viewer.
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._saving._samples_saving_widget import Worker
from napari_roxas_ai._settings._settings_manager import SettingsManager

if TYPE_CHECKING:
    import napari

settings = SettingsManager()

EDIT_SESSION_LAYERS = ("Cells Modification", "Rings Modification")


def install_unsaved_changes_guard(viewer: "napari.viewer.Viewer") -> None:
    """Guard the layer removal of the viewer (idempotent)."""
    layers = viewer.layers
    if getattr(layers.remove_selected, "_unsaved_changes_guard", False):
        return

    remove_selected = layers.remove_selected

    def guarded_remove_selected():
        if _confirm_removal(viewer, list(layers.selection)):
            remove_selected()

    guarded_remove_selected._unsaved_changes_guard = True
    layers.remove_selected = guarded_remove_selected


def _confirm_removal(viewer, removed_layers) -> bool:
    """Return True if the layers may be removed, saving them if requested."""
    exts = (
        settings.get("file_extensions.cells_file_extension")[0],
        settings.get("file_extensions.rings_file_extension")[0],
    )
    removed_samples = [l for l in removed_layers if l.name.endswith(exts)]
    dirty_layers = [
        l for l in removed_samples if l.metadata.get("_is_dirty", False)
    ]
    # An active edit session is lost when its working layer or the
    # cells/rings layers it will be applied to are removed
    active_sessions = [
        name
        for name in EDIT_SESSION_LAYERS
        if name in viewer.layers
        and (removed_samples or viewer.layers[name] in removed_layers)
    ]
    if not dirty_layers and not active_sessions:
        return True

    lines = [f"• {l.name} (not saved)" for l in dirty_layers]
    lines += [f"• {name} (edits not applied)" for name in active_sessions]
    box = QMessageBox(
        QMessageBox.Icon.Warning,
        "Unsaved changes",
        "The following changes will be lost:\n\n" + "\n".join(lines),
    )
    # Unapplied edits cannot be saved: they must be applied in the editor first
    save_button = (
        box.addButton("Save Changes", QMessageBox.ButtonRole.AcceptRole)
        if dirty_layers and not active_sessions
        else None
    )
    discard_button = box.addButton(
        "Discard", QMessageBox.ButtonRole.DestructiveRole
    )
    box.addButton(QMessageBox.StandardButton.Cancel)
    box.exec()

    clicked = box.clickedButton()
    if save_button is not None and clicked is save_button:
        # Saved synchronously: the layers must be written before removal
        Worker(dirty_layers).run()
        for layer in dirty_layers:
            layer.metadata["_is_dirty"] = False
        return True
    return clicked is discard_button
