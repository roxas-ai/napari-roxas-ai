from dataclasses import dataclass
from pathlib import Path
from typing import Any

from napari_roxas_ai._jobs import Progress
from napari_roxas_ai._writer import write_single_layer


@dataclass(frozen=True)
class LayerSnapshot:
    """What saving needs from a layer, taken on the GUI thread."""

    name: str
    data: Any
    kwargs: dict

    @classmethod
    def of(cls, layer) -> "LayerSnapshot":
        data, kwargs, _ = layer.as_layer_data_tuple()
        return cls(name=layer.name, data=data, kwargs=kwargs)


def save_layers(snapshots: list[LayerSnapshot], project_dir: str | None):
    """
    Write the layers next to their sample in the project.
    """
    if not project_dir:
        raise ValueError("No project directory is configured")
    project_dir = Path(project_dir)

    total = len(snapshots)
    for i, snapshot in enumerate(snapshots):
        yield Progress(i, total, snapshot.name)
        out_path = _output_path(snapshot, project_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the metadata in sync with where the layer is written
        metadata = {
            **(snapshot.kwargs.get("metadata") or {}),
            "file_path": str(out_path),
        }
        write_single_layer(
            str(out_path),
            snapshot.data,
            {**snapshot.kwargs, "metadata": metadata},
            save_annotated=False,
        )
    yield Progress(total, total)
    return total


def _output_path(snapshot: LayerSnapshot, project_dir: Path) -> Path:
    """The layer's own file path, else its sample's, else the project root."""
    metadata = snapshot.kwargs.get("metadata") or {}
    file_path = metadata.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        return Path(file_path)
    stem = metadata.get("sample_stem_path")
    if isinstance(stem, str) and stem.strip():
        return project_dir / stem
    sample_name = metadata.get("sample_name") or snapshot.name
    return project_dir / str(sample_name)
