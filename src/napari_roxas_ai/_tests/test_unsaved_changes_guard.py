import numpy as np
import pytest
from qtpy.QtWidgets import QMessageBox

from napari_roxas_ai._saving import _unsaved_changes_guard as guard
from napari_roxas_ai._saving import install_unsaved_changes_guard
from napari_roxas_ai._settings import SettingsManager


@pytest.fixture
def cells_name():
    ext = SettingsManager().get("file_extensions.cells_file_extension")[0]
    return f"sample{ext}"


@pytest.fixture
def answer(monkeypatch):
    """Answer the dialog with the button whose text is set in the returned dict."""
    state = {"button": None, "shown": 0}

    def fake_exec(box):
        state["shown"] += 1
        state["texts"] = [b.text() for b in box.buttons()]
        state["clicked"] = next(
            (b for b in box.buttons() if state["button"] in b.text()), None
        )

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    monkeypatch.setattr(QMessageBox, "clickedButton", lambda box: state["clicked"])
    return state


def _viewer_with_cells(make_napari_viewer, name, dirty):
    viewer = make_napari_viewer()
    layer = viewer.add_labels(np.zeros((5, 5), dtype=np.uint8), name=name)
    layer.metadata["_is_dirty"] = dirty
    install_unsaved_changes_guard(viewer)
    install_unsaved_changes_guard(viewer)  # idempotent
    viewer.layers.selection = {layer}
    return viewer


def test_clean_layers_removed_without_dialog(make_napari_viewer, cells_name, answer):
    viewer = _viewer_with_cells(make_napari_viewer, cells_name, dirty=False)
    viewer.layers.remove_selected()
    assert answer["shown"] == 0
    assert len(viewer.layers) == 0


@pytest.mark.parametrize("button, removed", [("Cancel", False), ("Discard", True)])
def test_dirty_layer_cancel_or_discard(make_napari_viewer, cells_name, answer, button, removed):
    viewer = _viewer_with_cells(make_napari_viewer, cells_name, dirty=True)
    answer["button"] = button
    viewer.layers.remove_selected()
    assert answer["shown"] == 1
    assert (cells_name not in viewer.layers) is removed


def test_dirty_layer_save_then_remove(make_napari_viewer, cells_name, answer, monkeypatch):
    saved = []

    class FakeWorker:
        def __init__(self, layers):
            self.layers = layers

        def run(self):
            saved.extend(l.name for l in self.layers)

    monkeypatch.setattr(guard, "Worker", FakeWorker)
    viewer = _viewer_with_cells(make_napari_viewer, cells_name, dirty=True)
    answer["button"] = "Save"
    viewer.layers.remove_selected()
    assert saved == [cells_name]
    assert cells_name not in viewer.layers


def test_edit_session_without_edits_is_ignored(make_napari_viewer, cells_name, answer):
    viewer = _viewer_with_cells(make_napari_viewer, cells_name, dirty=False)
    session = viewer.add_labels(np.zeros((5, 5), dtype=np.uint8), name="Cells Modification")
    session.metadata["_is_dirty"] = False
    viewer.layers.selection = {viewer.layers[cells_name]}
    viewer.layers.remove_selected()
    assert answer["shown"] == 0
    assert cells_name not in viewer.layers


def test_active_edit_session_offers_no_save(make_napari_viewer, cells_name, answer):
    viewer = _viewer_with_cells(make_napari_viewer, cells_name, dirty=True)
    session = viewer.add_labels(np.zeros((5, 5), dtype=np.uint8), name="Cells Modification")
    session.metadata["_is_dirty"] = True
    viewer.layers.selection = {viewer.layers[cells_name]}
    answer["button"] = "Cancel"
    viewer.layers.remove_selected()
    assert not any("Save" in t for t in answer["texts"])
    assert cells_name in viewer.layers
