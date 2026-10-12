import numpy as np
import pandas as pd
import pytest
from napari.layers import Shapes

from napari_roxas_ai._edition import _rings_layer_editor
from napari_roxas_ai._edition._rings_layer_editor import (
    DEFAULT_RINGS_MODEL,
    RingsLayerEditorWidget,
)


@pytest.fixture
def models_path(tmp_path, monkeypatch):
    """Provide a local rings models folder and prevent any asset download."""
    path = tmp_path / "_rings"
    path.mkdir()
    (path / "linear_rings_model.pth").write_bytes(b"weights")
    (path / DEFAULT_RINGS_MODEL).write_bytes(b"weights")
    (path / "manifest.json").write_text("{}")
    monkeypatch.setattr(_rings_layer_editor, "RINGS_MODELS_PATH", path)
    monkeypatch.setattr(
        _rings_layer_editor, "check_assets_and_download", lambda *a: None
    )
    return path


def _add_rings_layer(viewer):
    """Add a rings layer with three horizontal ring boundaries."""
    shape = (60, 50)
    features = pd.DataFrame(
        {
            "RBXY": [
                [[10.0, 0.0], [10.0, 25.0], [10.0, 50.0]],
                [[30.0, 0.0], [30.0, 25.0], [30.0, 50.0]],
                [[50.0, 0.0], [50.0, 25.0], [50.0, 50.0]],
            ],
            "YEAR": [2000, 2001, 2002],
            "enabled": [False, True, True],
        }
    )
    return viewer.add_labels(
        np.zeros(shape, dtype=np.int32),
        name="sample.rings",
        features=features,
        metadata={"rings_outmost_complete_year": 2002},
    )


def test_model_choices_filtered_and_default_selected(
    make_napari_viewer, models_path
):
    viewer = make_napari_viewer()
    widget = RingsLayerEditorWidget(viewer)

    # The manifest must not be offered as a model
    assert set(widget._rings_model_weights_file.choices) == {
        "linear_rings_model.pth",
        DEFAULT_RINGS_MODEL,
    }
    assert widget._rings_model_weights_file.value == DEFAULT_RINGS_MODEL


def test_ui_visibility_idle_and_editing(make_napari_viewer, models_path):
    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    widget = RingsLayerEditorWidget(viewer)
    widget.show()

    # Idle mode
    assert widget._edit_rings_geometries_button.visible
    assert widget._last_year_row.visible
    assert widget._save_all_layers_button.visible
    assert widget._visual_crossdating_button.visible
    assert not widget._apply_rings_geometries_button.visible
    assert not widget._undo_button.visible
    assert not widget._rerun_model_button.visible

    # Editing mode
    widget._edit_rings_geometries()
    assert not widget._edit_rings_geometries_button.visible
    assert not widget._save_all_layers_button.visible
    assert not widget._visual_crossdating_button.visible
    assert widget._cancel_rings_geometries_button.visible
    assert widget._apply_rings_geometries_button.visible
    assert widget._undo_button.visible
    assert not widget._undo_button.enabled  # nothing to undo yet
    assert widget._lasso_container.visible
    assert widget._rerun_params_row.visible
    assert widget._rerun_model_button.visible

    # Back to idle mode
    widget._cancel_rings_geometries()
    assert widget._save_all_layers_button.visible
    assert widget._visual_crossdating_button.visible
    assert not widget._undo_button.visible


def test_undo_lasso_deletion(make_napari_viewer, qtbot, models_path):
    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    widget = RingsLayerEditorWidget(viewer)
    widget.show()
    widget._edit_rings_geometries()

    edit_layer = viewer.layers["Rings Modification"]
    assert isinstance(edit_layer, Shapes)
    original_data = [d.copy() for d in edit_layer.data]

    # Delete all vertices of the middle boundary with a lasso
    widget._lasso_selection_checkbox.value = True
    viewer.layers["Lasso Selection"].data = [
        np.array([[25, -5], [25, 55], [35, 55], [35, -5]], dtype=float)
    ]
    widget._execute_lasso_deletion()

    assert len(edit_layer.data) == 2
    assert len(widget._undo_stack) == 1
    assert widget._undo_button.enabled

    widget._undo_last_change()

    assert len(edit_layer.data) == 3
    for restored, original in zip(edit_layer.data, original_data):
        np.testing.assert_array_equal(restored, original)
    assert edit_layer.features["YEAR"].tolist() == [2000, 2001, 2002]
    assert not widget._undo_button.enabled


def test_undo_restores_user_edits_in_reverse_order(
    make_napari_viewer, models_path
):
    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    widget = RingsLayerEditorWidget(viewer)
    widget._edit_rings_geometries()
    edit_layer = viewer.layers["Rings Modification"]
    original_last_boundary = edit_layer.data[2].copy()

    # Two completed user edits: remove a boundary, then add one
    edit_layer.selected_data = {2}
    edit_layer.remove_selected()
    assert len(edit_layer.data) == 2
    edit_layer.add(
        [np.array([[40.0, 0.0], [40.0, 50.0]])], shape_type="path"
    )
    assert len(edit_layer.data) == 3
    assert len(widget._undo_stack) == 2

    widget._undo_last_change()
    assert len(edit_layer.data) == 2

    widget._undo_last_change()
    assert len(edit_layer.data) == 3
    np.testing.assert_array_equal(edit_layer.data[2], original_last_boundary)
    assert not widget._undo_stack

    # Undo history is cleared at the end of the edit session
    widget._apply_rings_geometries()
    assert widget._undo_current is None


def test_open_visual_crossdating_floats_widget(
    make_napari_viewer, models_path, monkeypatch
):
    viewer = make_napari_viewer()
    widget = RingsLayerEditorWidget(viewer)

    calls = []
    dock = object()
    monkeypatch.setattr(
        viewer.window,
        "add_plugin_dock_widget",
        lambda plugin, name: calls.append((plugin, name)) or (dock, None),
    )
    from napari_roxas_ai._crossdating import CrossDatingPlotterWidget

    configured = []
    monkeypatch.setattr(
        CrossDatingPlotterWidget,
        "_configure_floating_dock",
        lambda d, v: configured.append(d),
    )

    widget._visual_crossdating_button.clicked.emit()

    assert calls == [("napari-roxas-ai", "7 - Visual crossdating")]
    assert configured == [dock]


def test_save_all_layers_without_layers_reenables_buttons(
    make_napari_viewer, models_path, monkeypatch
):
    viewer = make_napari_viewer()
    widget = RingsLayerEditorWidget(viewer)

    messages = []
    monkeypatch.setattr(_rings_layer_editor, "show_info", messages.append)

    widget._save_all_layers()

    assert widget._save_all_layers_button.enabled
    assert widget._edit_rings_geometries_button.enabled
    assert messages == ["No rings or cells layers to save"]


class _FakeRingsModel:
    """Stand-in for the packaged rings model, returning one new boundary."""

    def __init__(self):
        self.received_start_boundary = None

    def to(self, device):
        return self

    def infer(self, img, start_boundary=None):
        self.received_start_boundary = start_boundary
        return None, [[[40.0, 0.0], [40.0, 25.0], [40.0, 50.0]]]


def test_rerun_runs_in_worker_and_is_undoable(
    make_napari_viewer, qtbot, models_path, monkeypatch
):
    import threading

    import torch.package

    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    viewer.add_image(np.zeros((60, 50), dtype=np.uint8), name="sample.scan")
    widget = RingsLayerEditorWidget(viewer)
    widget._edit_rings_geometries()
    edit_layer = viewer.layers["Rings Modification"]

    fake_model = _FakeRingsModel()
    infer_threads = []
    original_infer = fake_model.infer

    def tracking_infer(img, start_boundary=None):
        infer_threads.append(threading.current_thread())
        return original_infer(img, start_boundary=start_boundary)

    fake_model.infer = tracking_infer

    class _FakeImporter:
        def __init__(self, path):
            pass

        def load_pickle(self, package, resource):
            return fake_model

    monkeypatch.setattr(torch.package, "PackageImporter", _FakeImporter)

    # Rerun from the first complete year: the boundaries after it are replaced
    widget._rerun_model_year_spinbox.value = 2001
    widget._rerun_model_from_year()

    # The edit session controls are locked and a spinner is shown while the worker runs
    assert not widget._rerun_model_button.enabled
    assert not widget._apply_rings_geometries_button.enabled
    assert widget._rerun_model_button.text.startswith("Rerunning Ring Segmentation")
    assert widget._spinner_timer.isActive()

    qtbot.wait_until(lambda: widget._rerun_model_button.enabled, timeout=5000)

    assert infer_threads and infer_threads[0] is not threading.main_thread()
    assert fake_model.received_start_boundary is not None
    assert widget._apply_rings_geometries_button.enabled
    assert widget._cancel_rings_geometries_button.enabled
    assert widget._rerun_model_button.text == "Rerun Ring Segmentation"
    assert not widget._spinner_timer.isActive()
    # 2000 and 2001 kept, 2002 replaced by the new boundary
    assert len(edit_layer.data) == 3
    np.testing.assert_array_equal(edit_layer.data[2][:, 0], 40.0)

    # The whole rerun is a single undo step
    assert len(widget._undo_stack) == 1
    widget._undo_last_change()
    np.testing.assert_array_equal(edit_layer.data[2][:, 0], 50.0)


def test_save_all_layers_shows_hint_when_completed(
    make_napari_viewer, qtbot, models_path, monkeypatch, tmp_path
):
    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    widget = RingsLayerEditorWidget(viewer)

    written = []
    monkeypatch.setattr(
        "napari_roxas_ai._saving._samples_saving_widget.write_single_layer",
        lambda path, data, kwargs: written.append(path),
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
    messages = []
    monkeypatch.setattr(_rings_layer_editor, "show_info", messages.append)

    widget._save_all_layers()
    qtbot.wait_until(
        lambda: widget._save_all_layers_button.enabled, timeout=5000
    )

    assert len(written) == 1
    assert messages == ["1 layer(s) saved successfully"]
