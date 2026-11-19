from typing import TYPE_CHECKING

import napari.layers

from napari_roxas_ai._job_runner import JobHandle, get_runner
from napari_roxas_ai._jobs import Job, JobTag

from ._operations import load_samples

if TYPE_CHECKING:
    import napari


def start_loading(
    viewer: "napari.viewer.Viewer",
    sample_stem_paths: list[str],
    project_directory: str,
) -> JobHandle:
    """
    Load the samples' layers into ``viewer`` in the background.

    Each layer is added as soon as it is read.
    """
    return get_runner().submit(
        Job(
            "Loading",
            load_samples,
            args=(sample_stem_paths, project_directory),
            tags={JobTag.LOADING},
        ),
        on_partial=lambda layer_data: _add_layer(viewer, layer_data),
    )


def _add_layer(viewer: "napari.viewer.Viewer", layer_data: tuple) -> None:
    data, add_kwargs, layer_type = layer_data
    layer = napari.layers.Layer.create(data, add_kwargs, layer_type)

    original_path = add_kwargs.get("metadata", {}).get("path")
    if original_path:
        print(f"[Loader] Setting file_path = {original_path}")
        layer.metadata["file_path"] = original_path
    else:
        print("[Loader] WARNING: no file_path in reader metadata")

    viewer.add_layer(layer)
