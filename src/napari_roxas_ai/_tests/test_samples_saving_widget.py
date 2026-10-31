from unittest.mock import MagicMock
import pytest
import napari
import numpy as np

from napari_roxas_ai._saving._samples_saving_widget import SamplesSavingWidget, Worker


def test_samples_saving_widget_layer_filtering(make_napari_viewer, monkeypatch, tmp_path):
    viewer = make_napari_viewer()
    
    # Add various layers: scan, cells, rings, and temporary/other
    scan_layer = viewer.add_image(np.zeros((10, 10)), name="sample1.scan")
    cells_layer = viewer.add_labels(np.zeros((10, 10), dtype=int), name="sample1.cells")
    rings_layer = viewer.add_labels(np.zeros((10, 10), dtype=int), name="sample1.rings")
    temp_layer = viewer.add_shapes(name="sample1.vectorization_preview")

    widget = SamplesSavingWidget(viewer)

    saved_worker_layers = []

    class MockWorker:
        def __init__(self, layers):
            nonlocal saved_worker_layers
            saved_worker_layers = list(layers)
            self.finished = MagicMock()
            self.progress = MagicMock()
            self.run = MagicMock()
            self.deleteLater = MagicMock()
        def moveToThread(self, thread):
            pass

    monkeypatch.setattr("napari_roxas_ai._saving._samples_saving_widget.Worker", MockWorker)
    monkeypatch.setattr("napari_roxas_ai._saving._samples_saving_widget.QThread", MagicMock)

    # Test "Save all layers": should only include cells and rings (max 2 for this sample), not scan or temp
    widget._save_layers(how="all")
    assert len(saved_worker_layers) == 2
    assert cells_layer in saved_worker_layers
    assert rings_layer in saved_worker_layers
    assert scan_layer not in saved_worker_layers
    assert temp_layer not in saved_worker_layers

    # Test "Save selected layers" when only scan and temp are selected
    saved_worker_layers.clear()
    viewer.layers.selection.clear()
    viewer.layers.selection.add(scan_layer)
    viewer.layers.selection.add(temp_layer)
    widget._save_layers(how="selected")
    # No valid layers to save -> worker not created
    assert len(saved_worker_layers) == 0

    # Test "Save selected layers" when cells and scan are selected
    saved_worker_layers.clear()
    viewer.layers.selection.clear()
    viewer.layers.selection.add(scan_layer)
    viewer.layers.selection.add(cells_layer)
    widget._save_layers(how="selected")
    assert len(saved_worker_layers) == 1
    assert saved_worker_layers == [cells_layer]


def test_progress_bar_update(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SamplesSavingWidget(viewer)

    # Initially progress bar should have min 0, max 100
    assert widget._progress_bar.min == 0
    assert widget._progress_bar.max == 100
    assert not widget._progress_bar.native.isVisible()

    # 1 out of 2 layers saved -> 50%
    widget._update_progress(1, 2)
    assert widget._progress_bar.value == 50
    assert not widget._progress_bar.native.isHidden()

    # 2 out of 2 layers saved -> 100%
    widget._update_progress(2, 2)
    assert widget._progress_bar.value == 100
    assert not widget._progress_bar.native.isHidden()


def test_save_buttons_report_nothing_to_save(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_image(np.zeros((10, 10)), name="sample1.scan")
    widget = SamplesSavingWidget(viewer)

    messages = []
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.show_info", messages.append
    )

    widget._save_layers(how="all")
    assert messages == ["No rings or cells layers to save"]
    assert widget.worker is None
    assert widget.worker_thread is None


def _mock_save_run(monkeypatch):
    """Replace the save worker and thread by mocks, return (worker, thread)."""
    worker = MagicMock()
    thread = MagicMock()

    def make_worker(layers):
        worker.layers = layers
        return worker

    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.Worker", make_worker
    )
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.QThread", lambda: thread
    )
    return worker, thread


def _finish_save_run(thread):
    """Run the callbacks connected to the end of the save thread."""
    for call in thread.finished.connect.call_args_list:
        call.args[0]()


def test_save_all_skips_clean_layers(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_labels(
        np.zeros((10, 10), dtype=int),
        name="sample1.cells",
        metadata={"_is_dirty": False},
    )
    viewer.add_labels(
        np.zeros((10, 10), dtype=int),
        name="sample1.rings",
        metadata={"_is_dirty": False},
    )
    widget = SamplesSavingWidget(viewer)

    messages = []
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.show_info", messages.append
    )
    _mock_save_run(monkeypatch)

    widget._save_layers(how="all")
    assert messages == ["No unsaved changes in cells or rings layers."]
    assert widget.worker is None
    assert widget.worker_thread is None
    assert not widget._progress_bar.visible


def test_save_all_saves_only_the_modified_layer(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    viewer.add_labels(
        np.zeros((10, 10), dtype=int),
        name="sample1.cells",
        metadata={"_is_dirty": False},
    )
    rings_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=int),
        name="sample1.rings",
        metadata={"_is_dirty": True},
    )
    widget = SamplesSavingWidget(viewer)

    messages = []
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.show_info", messages.append
    )
    worker, thread = _mock_save_run(monkeypatch)

    widget._save_layers(how="all")
    assert worker.layers == [rings_layer]
    assert messages == []

    _finish_save_run(thread)
    assert messages == ["Saved rings layer."]
    assert rings_layer.metadata["_is_dirty"] is False


def test_save_buttons_report_saved_layers(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    cells_layer = viewer.add_labels(np.zeros((10, 10), dtype=int), name="sample1.cells")
    rings_layer = viewer.add_labels(np.zeros((10, 10), dtype=int), name="sample1.rings")
    widget = SamplesSavingWidget(viewer)

    messages = []
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.show_info", messages.append
    )
    worker, thread = _mock_save_run(monkeypatch)

    # Layers without the flag count as modified
    widget._save_layers(how="all")
    assert worker.layers == [cells_layer, rings_layer]
    assert messages == []

    _finish_save_run(thread)
    assert messages == ["Saved cells and rings layers."]
    assert cells_layer.metadata["_is_dirty"] is False
    assert rings_layer.metadata["_is_dirty"] is False


def test_worker_skips_the_annotated_image(make_napari_viewer, monkeypatch, tmp_path):
    viewer = make_napari_viewer()
    rings_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=int),
        name="sample1.rings",
        metadata={"sample_name": "sample1"},
    )

    from napari_roxas_ai._saving import _samples_saving_widget

    saving_settings = _samples_saving_widget.settings
    original_get = saving_settings.get
    monkeypatch.setattr(
        saving_settings,
        "get",
        lambda key: (
            str(tmp_path) if key == "project_directory" else original_get(key)
        ),
    )
    calls = []
    monkeypatch.setattr(
        _samples_saving_widget,
        "write_single_layer",
        lambda path, data, kwargs, **options: calls.append(options),
    )

    Worker([rings_layer]).run()
    assert len(calls) == 1
    assert calls[0]["save_annotated"] is False
