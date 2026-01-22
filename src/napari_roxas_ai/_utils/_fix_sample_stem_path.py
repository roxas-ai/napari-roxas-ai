
from __future__ import annotations

import json
from pathlib import Path

from napari_roxas_ai._settings import SettingsManager


def fix_sample_stem_paths_in_project(project_dir: str | Path) -> None:
    """
    Fix outdated 'sample_stem_path' entries in metadata files after folder renames.

    - Walks the project directory
    - For each *.metadata.json:
        derives the correct stem from the file location
        compares it to stored sample_stem_path
        updates it if mismatching
    - Prints ONLY when a change was applied
    """

    settings = SettingsManager()
    metadata_ext = "".join(settings.get("file_extensions.metadata_file_extension"))

    project_dir = Path(project_dir).resolve()
    if not project_dir.exists():
        return

    for meta_path in project_dir.rglob(f"*{metadata_ext}"):

        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        # --- derive correct values from filesystem ---
        # <project>/<subdirs>/<sample>.metadata.json
        sample_name = meta_path.name[: -len(metadata_ext)]
        parent_rel = meta_path.parent.relative_to(project_dir)

        if str(parent_rel) == ".":
            correct_stem = sample_name
        else:
            correct_stem = f"{parent_rel.as_posix()}/{sample_name}"

        old_stem = data.get("sample_stem_path")

        if old_stem != correct_stem:
            data["sample_stem_path"] = correct_stem

            meta_path.write_text(
                json.dumps(data, indent=4),
                encoding="utf-8",
            )

            print(
                "[fix_sample_stem_path]",
                f"{meta_path}",
                f"'{old_stem}' -> '{correct_stem}'",
            )
