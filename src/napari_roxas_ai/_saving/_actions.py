from collections.abc import Iterable

from napari.utils.notifications import show_info

from napari_roxas_ai._job_runner import JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag
from napari_roxas_ai._settings._settings_manager import SettingsManager

from ._operations import LayerSnapshot, save_layers

settings = SettingsManager()


def start_saving(layers: Iterable, only_unsaved: bool) -> JobHandle | None:
    """
    Save the cells and rings layers among ``layers`` in the background.

    With ``only_unsaved`` the layers without unsaved changes are skipped
    (metadata "_is_dirty").

    Returns the running save, or None when there is nothing to
    save.
    """
    rings_ext = settings.get("file_extensions.rings_file_extension")[0]
    cells_ext = settings.get("file_extensions.cells_file_extension")[0]

    layers_to_save = [
        layer
        for layer in layers
        if layer.name.endswith((rings_ext, cells_ext))
    ]
    if not layers_to_save:
        show_info("No rings or cells layers to save")
        return None

    if only_unsaved:
        layers_to_save = [
            layer
            for layer in layers_to_save
            if layer.metadata.get("_is_dirty", True)
        ]
        if not layers_to_save:
            show_info("No unsaved changes in cells or rings layers.")
            return None

    saved_types = [
        "cells" if layer.name.endswith(cells_ext) else "rings"
        for layer in layers_to_save
    ]
    if len(saved_types) == 1:
        saved_message = f"Saved {saved_types[0]} layer."
    else:
        saved_message = f"Saved {' and '.join(saved_types)} layers."

    def on_saved(_) -> None:
        for layer in layers_to_save:
            layer.metadata["_is_dirty"] = False
        show_info(saved_message)

    return get_runner().submit(
        Job(
            "Saving",
            save_layers,
            args=(
                [LayerSnapshot.of(layer) for layer in layers_to_save],
                settings.get("project_directory"),
            ),
            tags={JobTag.SAVING},
        ),
        on_result=on_saved,
    )
