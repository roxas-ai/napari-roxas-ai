from pathlib import Path

from napari_roxas_ai._jobs import Partial, Progress
from napari_roxas_ai._reader import (
    read_cells_file,
    read_rings_file,
    read_scan_file,
)
from napari_roxas_ai._settings._settings_manager import SettingsManager

settings = SettingsManager()


def load_samples(sample_stem_paths: list[str], project_directory: str):
    """
    Read the scan, cells and rings files of the samples.
    Returns None, Yields its layers.
    """
    scan_file_extension = "".join(
        settings.get("file_extensions.scan_file_extension")
    )
    cells_file_extension = "".join(
        settings.get("file_extensions.cells_file_extension")
    )
    rings_file_extension = "".join(
        settings.get("file_extensions.rings_file_extension")
    )
    image_file_extensions = settings.get(
        "file_extensions.image_file_extensions"
    )
    scan_content_extension = settings.get(
        "file_extensions.scan_file_extension"
    )[0]

    project_dir = Path(project_directory).resolve()
    total = len(sample_stem_paths) * 3
    step = 0

    for sample_stem_path in sample_stem_paths:
        sample_path = Path(sample_stem_path).resolve()
        try:
            relative_stem = sample_path.relative_to(project_dir)
        except ValueError:
            relative_stem = sample_path

        # The scan may be saved with any supported image extension
        scan_file_path = f"{sample_stem_path}{scan_file_extension}"
        if not Path(scan_file_path).exists():
            for ext in image_file_extensions:
                alt_path = f"{sample_stem_path}{scan_content_extension}{ext}"
                if Path(alt_path).exists():
                    scan_file_path = alt_path
                    break

        for file_path, reader in (
            (scan_file_path, read_scan_file),
            (f"{sample_stem_path}{cells_file_extension}", read_cells_file),
            (f"{sample_stem_path}{rings_file_extension}", read_rings_file),
        ):
            if Path(file_path).exists():
                yield Progress(step, total, f"Loading {Path(file_path).name}")
                data, add_kwargs, layer_type = reader(file_path)
                add_kwargs["metadata"]["sample_stem_path"] = str(relative_stem)
                yield Partial((data, add_kwargs, layer_type))
            step += 1

    yield Progress(total, total)
