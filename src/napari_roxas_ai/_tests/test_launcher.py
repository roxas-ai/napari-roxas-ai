import napari

from napari_roxas_ai import _launcher
from napari_roxas_ai.shortcuts import has_shortcuts_applied, is_widget_already_present
from napari_roxas_ai._edition._rings_layer_editor import RingsLayerEditorWidget


def test_launcher_installs_shortcuts_and_docks_group(make_napari_viewer, monkeypatch):
    viewer = make_napari_viewer()
    monkeypatch.setattr(napari, "Viewer", lambda: viewer)
    monkeypatch.setattr(napari, "run", lambda: None)

    _launcher.main()

    assert has_shortcuts_applied(viewer)
    assert is_widget_already_present(viewer, RingsLayerEditorWidget)
