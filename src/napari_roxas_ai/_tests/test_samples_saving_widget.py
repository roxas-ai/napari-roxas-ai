import numpy as np
import pytest

from napari_roxas_ai._job_runner import JobHandle, _runner
from napari_roxas_ai._jobs import Job, JobStatus, Progress, run_job_sync
from napari_roxas_ai._saving import (
    SamplesSavingWidget,
    _operations,
    _samples_saving_widget,
    start_saving,
)
from napari_roxas_ai._saving._operations import LayerSnapshot, save_layers

ACTIONS = "napari_roxas_ai._saving._actions"


@pytest.fixture
def saved(monkeypatch):
    """Replace the save job: record the names of the layers it would save."""
    names = []

    def fake_save_layers(snapshots, project_dir):
        names.extend(snapshot.name for snapshot in snapshots)
        yield Progress(len(snapshots), len(snapshots))
        return len(snapshots)

    monkeypatch.setattr(f"{ACTIONS}.save_layers", fake_save_layers)
    return names


@pytest.fixture
def messages(monkeypatch):
    shown = []
    monkeypatch.setattr(f"{ACTIONS}.show_info", shown.append)
    return shown


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


def add_labels(viewer, name, **metadata):
    return viewer.add_labels(
        np.zeros((10, 10), dtype=int), name=name, metadata=metadata
    )


# -- start_saving --


def test_only_cells_and_rings_layers_are_saved(viewer, sync_runner, saved):
    viewer.add_image(np.zeros((10, 10)), name="sample1.scan")
    add_labels(viewer, "sample1.cells")
    add_labels(viewer, "sample1.rings")
    viewer.add_shapes(name="sample1.vectorization_preview")

    start_saving(viewer.layers, only_unsaved=False)
    assert saved == ["sample1.cells", "sample1.rings"]


def test_nothing_to_save_is_reported(viewer, sync_runner, messages, saved):
    viewer.add_image(np.zeros((10, 10)), name="sample1.scan")

    assert start_saving(viewer.layers, only_unsaved=False) is None
    assert messages == ["No rings or cells layers to save"]
    assert saved == []


def test_only_unsaved_skips_saved_layers(viewer, sync_runner, messages, saved):
    add_labels(viewer, "sample1.cells", _is_dirty=False)
    add_labels(viewer, "sample1.rings", _is_dirty=True)

    start_saving(viewer.layers, only_unsaved=True)
    assert saved == ["sample1.rings"]


def test_no_unsaved_changes_is_reported(viewer, sync_runner, messages, saved):
    add_labels(viewer, "sample1.cells", _is_dirty=False)

    assert start_saving(viewer.layers, only_unsaved=True) is None
    assert messages == ["No unsaved changes in cells or rings layers."]
    assert saved == []


def test_saved_layers_are_marked_saved(viewer, sync_runner, messages, saved):
    # Layers without the flag count as modified
    cells = add_labels(viewer, "sample1.cells")
    rings = add_labels(viewer, "sample1.rings")

    handle = start_saving(viewer.layers, only_unsaved=True)
    assert handle.status is JobStatus.SUCCEEDED
    assert messages == ["Saved cells and rings layers."]
    assert cells.metadata["_is_dirty"] is False
    assert rings.metadata["_is_dirty"] is False


def test_failed_save_is_reported_and_layers_stay_unsaved(
    viewer, sync_runner, messages, monkeypatch
):
    rings = add_labels(viewer, "sample1.rings")

    def failing_save_layers(snapshots, project_dir):
        yield Progress(0, 1)
        raise OSError("disk full")

    monkeypatch.setattr(f"{ACTIONS}.save_layers", failing_save_layers)
    errors = []
    monkeypatch.setattr(_runner, "show_error", errors.append)

    handle = start_saving(viewer.layers, only_unsaved=True)
    assert handle.status is JobStatus.FAILED
    assert errors == ["Saving failed: disk full"]
    assert messages == []  # no "Saved ..." message
    assert rings.metadata.get("_is_dirty", True)


# -- The saving job --


def test_save_skips_the_annotated_image(viewer, monkeypatch, tmp_path):
    rings = add_labels(viewer, "sample1.rings", sample_name="sample1")
    calls = []
    monkeypatch.setattr(
        _operations,
        "write_single_layer",
        lambda path, data, kwargs, **options: calls.append(options),
    )

    outcome = run_job_sync(
        Job(
            "save",
            save_layers,
            args=([LayerSnapshot.of(rings)], str(tmp_path)),
        )
    )
    assert outcome.status is JobStatus.SUCCEEDED
    assert calls == [{"save_annotated": False}]


def test_save_without_project_directory_fails():
    outcome = run_job_sync(Job("save", save_layers, args=([], None)))
    assert outcome.status is JobStatus.FAILED


# -- The widget --


def test_other_save_button_is_locked_while_saving(viewer, monkeypatch):
    save = JobHandle(Job("Saving", lambda: None))  # finished by hand below
    monkeypatch.setattr(
        _samples_saving_widget, "start_saving", lambda *a, **k: save
    )
    widget = SamplesSavingWidget(viewer)

    widget._save_all_layers_button._button.native.click()
    assert not widget._save_selected_layers_button.enabled

    save._on_succeeded(1)
    save._on_finished()
    assert widget._save_selected_layers_button.enabled
    assert widget._save_all_layers_button.enabled
