import napari
import pytest
from napari_roxas_ai.shortcuts import (
    close_all_dock_widgets,
    has_shortcuts_applied,
    install_wasd_shortcuts,
    is_widget_already_present,
    load_batch_operations_group,
    load_manual_editing_group,
)
from napari_roxas_ai._loading._samples_loading_widget import SamplesLoadingWidget
from napari_roxas_ai._settings._settings_widget import SettingsWidget
from qtpy.QtWidgets import QApplication
import yaml
import os
import napari_roxas_ai




def test_prevent_duplicate_individual_widget_dock(make_napari_viewer):
    """Verify that attempting to dock a widget already inside an active group is rejected."""
    viewer = make_napari_viewer()
    group = load_manual_editing_group(viewer)
    viewer.window.add_dock_widget(group, name="Group - Manual Editing")
    assert "Group - Manual Editing" in viewer.window.dock_widgets

    # Instantiate and try adding SamplesLoadingWidget
    w = SamplesLoadingWidget(viewer)
    viewer.window.add_dock_widget(w, name="2/4 - Load image(s)")
    app = QApplication.instance()
    if app:
        app.processEvents()

    # The duplicate dock should have been removed
    assert "2/4 - Load image(s)" not in viewer.window.dock_widgets
    assert "Group - Manual Editing" in viewer.window.dock_widgets


def test_lazy_imports():
    """Verify that widget group functions are exported and resolvable from the package root."""
    assert callable(napari_roxas_ai.load_manual_editing_group)
    assert callable(napari_roxas_ai.load_batch_operations_group)
    assert callable(napari_roxas_ai.close_all_dock_widgets)


def test_load_manual_editing_group(make_napari_viewer):
    """Verify loading the manual editing group returns the unified container and attaches."""
    viewer = make_napari_viewer()
    group = load_manual_editing_group(viewer)
    viewer.window.add_dock_widget(group, name="Group - Manual Editing")

    dock_keys = list(viewer.window.dock_widgets.keys())
    assert "Group - Manual Editing" in dock_keys
    assert has_shortcuts_applied(viewer)
    assert is_widget_already_present(viewer, SamplesLoadingWidget)
    assert not is_widget_already_present(viewer, SettingsWidget)


def test_load_batch_operations_group(make_napari_viewer):
    """Verify loading the batch operations group closes prior docks and attaches the unified batch dock."""
    viewer = make_napari_viewer()
    group_m = load_manual_editing_group(viewer)
    viewer.window.add_dock_widget(group_m, name="Group - Manual Editing")
    assert "Group - Manual Editing" in viewer.window.dock_widgets

    group_b = load_batch_operations_group(viewer)
    viewer.window.add_dock_widget(group_b, name="Group - Batch Operations")
    dock_keys = list(viewer.window.dock_widgets.keys())
    assert "Group - Batch Operations" in dock_keys
    assert "Group - Manual Editing" not in dock_keys


def test_close_all_dock_widgets(make_napari_viewer):
    """Verify close_all_dock_widgets closes all open plugin dock widgets."""
    viewer = make_napari_viewer()
    group = load_manual_editing_group(viewer)
    viewer.window.add_dock_widget(group, name="Group - Manual Editing")
    assert len(viewer.window.dock_widgets) == 1

    close_all_dock_widgets(viewer)
    assert len(viewer.window.dock_widgets) == 0


def test_widget_group_shortcuts_registered(make_napari_viewer):
    """Verify that Ctrl+Shift+M, Ctrl+Shift+B, and Ctrl+Shift+W are bound."""
    viewer = make_napari_viewer()
    install_wasd_shortcuts(viewer)

    bound_keys = [str(k) for k in viewer.keymap.keys()]
    # Check that keybindings for M, B, W are present
    assert any("M" in k for k in bound_keys)
    assert any("B" in k for k in bound_keys)
    assert any("W" in k for k in bound_keys)
