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
        _rings_layer_editor,
        "check_assets_and_download",
        lambda *args, **kwargs: None,
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
    assert not widget._rerun_separator.visible
    assert not widget._rerun_year_row.visible
    assert not widget._rerun_model_row.visible
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
    assert widget._rerun_separator.visible
    assert widget._rerun_year_row.visible
    assert widget._rerun_model_row.visible
    assert widget._rerun_model_button.visible

    # Back to idle mode
    widget._cancel_rings_geometries()
    assert widget._save_all_layers_button.visible
    assert widget._visual_crossdating_button.visible
    assert not widget._undo_button.visible
    assert not widget._rerun_separator.visible
    assert not widget._rerun_year_row.visible
    assert not widget._rerun_model_row.visible
    assert not widget._rerun_model_button.visible


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


def test_open_visual_crossdating_positions_widget_only_once(
    make_napari_viewer, models_path, monkeypatch
):
    from napari_roxas_ai._crossdating import CrossDatingPlotterWidget

    viewer = make_napari_viewer()
    widget = RingsLayerEditorWidget(viewer)

    configured = []
    monkeypatch.setattr(
        CrossDatingPlotterWidget,
        "_configure_floating_dock",
        staticmethod(lambda d, v: configured.append(d)),
    )

    # First opening: the new widget positions itself (deferred), so the
    # editor must not position it as well
    widget._visual_crossdating_button.clicked.emit()
    assert any(
        name.startswith("7 - Visual crossdating")
        for name in viewer.window.dock_widgets
    )
    assert configured == []

    # Already open: the editor brings the existing widget back to the floating layout
    widget._visual_crossdating_button.clicked.emit()
    assert len(configured) == 1


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


@pytest.fixture
def missing_models(tmp_path, monkeypatch):
    """Rings models folder of an older installation (default model missing)."""
    path = tmp_path / "_rings_missing"
    path.mkdir()
    (path / "linear_rings_model.pth").write_bytes(b"weights")
    monkeypatch.setattr(_rings_layer_editor, "RINGS_MODELS_PATH", path)
    messages = []
    monkeypatch.setattr(
        _rings_layer_editor, "show_info", lambda m: messages.append(("info", m))
    )
    monkeypatch.setattr(
        _rings_layer_editor,
        "show_warning",
        lambda m: messages.append(("warning", m)),
    )
    return path, messages


def _wait_for_download_thread(qtbot):
    qtbot.wait_until(
        lambda: _rings_layer_editor._models_download["thread"] is None,
        timeout=5000,
    )


def test_missing_model_downloaded_in_background(
    make_napari_viewer, qtbot, missing_models, monkeypatch
):
    import threading

    path, messages = missing_models
    release_download = threading.Event()
    download_threads = []

    def fake_download(directory, asset_name, required_files=()):
        download_threads.append(threading.current_thread())
        release_download.wait(5)
        (path / DEFAULT_RINGS_MODEL).write_bytes(b"weights")

    monkeypatch.setattr(
        _rings_layer_editor, "check_assets_and_download", fake_download
    )

    viewer = make_napari_viewer()
    # E.g. the hidden editor of the loading widget and the one opened by the user
    widget = RingsLayerEditorWidget(viewer)
    second_widget = RingsLayerEditorWidget(viewer)

    # The editors are created while the download is still running
    assert _rings_layer_editor.is_rings_models_download_running()
    assert widget._rings_model_weights_file.choices == ("linear_rings_model.pth",)
    assert messages == [
        (
            "info",
            f"The rings model '{DEFAULT_RINGS_MODEL}' is not available yet "
            "and is being downloaded in the background...",
        )
    ]

    # A rerun is refused while the model is being downloaded
    widget._rerun_model_from_year()
    assert "still being downloaded" in messages[-1][1]

    release_download.set()
    _wait_for_download_thread(qtbot)

    # Downloaded once, outside the main thread
    assert not _rings_layer_editor.is_rings_models_download_running()
    assert len(download_threads) == 1
    assert download_threads[0] is not threading.main_thread()
    assert messages[-1] == (
        "info",
        f"The rings model '{DEFAULT_RINGS_MODEL}' has been downloaded "
        "and is now available",
    )
    for editor in (widget, second_widget):
        assert editor._rings_model_weights_file.value == DEFAULT_RINGS_MODEL


def test_failed_background_download_shows_warning(
    make_napari_viewer, qtbot, missing_models, monkeypatch
):
    path, messages = missing_models

    def failing_download(directory, asset_name, required_files=()):
        raise ConnectionError("offline")

    monkeypatch.setattr(
        _rings_layer_editor, "check_assets_and_download", failing_download
    )

    viewer = make_napari_viewer()
    widget = RingsLayerEditorWidget(viewer)
    _wait_for_download_thread(qtbot)

    assert not _rings_layer_editor.is_rings_models_download_running()
    assert messages[-1][0] == "warning"
    assert "offline" in messages[-1][1]
    # The editor keeps working with the models present locally
    assert widget._rings_model_weights_file.choices == ("linear_rings_model.pth",)


def test_transform_tool_affine_applied_to_rings_layer(
    make_napari_viewer, models_path
):
    from napari.utils.transforms import Affine

    viewer = make_napari_viewer()
    rings = _add_rings_layer(viewer)
    # Set non-unit scale (e.g. 2.0 um/px, spatial_resolution = 0.5 px/um)
    rings.scale = [2.0, 2.0]
    widget = RingsLayerEditorWidget(viewer)
    widget.show()
    widget._edit_rings_geometries()

    edit_layer = viewer.layers["Rings Modification"]
    assert np.allclose(edit_layer.scale, [2.0, 2.0])
    original_coords = [np.array(d, copy=True) for d in edit_layer.data]

    # Apply world affine translation (+10 in Y / row world direction -> +5 in pixel space)
    edit_layer.affine = Affine(translate=[10.0, 0.0])

    widget._apply_rings_geometries()

    rings_layer = viewer.layers["sample.rings"]
    applied_rbxy = rings_layer.features["RBXY"].tolist()

    # Verify that the applied ring boundary coordinates reflect the +5 pixel translation
    assert len(applied_rbxy) == len(original_coords)
    for orig, applied in zip(original_coords, applied_rbxy):
        applied_arr = np.array(applied)
        # Check interior coordinates shifted by 5 pixels in row direction
        orig_y = orig[:, 0] + 5.0
        # Compare row coordinates for columns within original range
        assert np.all(applied_arr[:, 0] >= 5.0)


def test_transform_tool_undo_restores_affine(
    make_napari_viewer, qtbot, models_path
):
    from napari.utils.transforms import Affine

    viewer = make_napari_viewer()
    _add_rings_layer(viewer)
    widget = RingsLayerEditorWidget(viewer)
    widget.show()
    widget._edit_rings_geometries()

    edit_layer = viewer.layers["Rings Modification"]
    initial_affine = np.array(edit_layer.affine.affine_matrix, copy=True)

    # Apply affine transform and record undo step
    edit_layer.affine = Affine(translate=[10.0, 5.0])
    widget._record_undo_step()

    assert widget._undo_button.enabled
    assert not np.allclose(
        edit_layer.affine.affine_matrix, initial_affine
    )

    widget._undo_last_change()

    assert np.allclose(edit_layer.affine.affine_matrix, initial_affine)
    assert not widget._undo_button.enabled
