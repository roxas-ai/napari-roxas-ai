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
