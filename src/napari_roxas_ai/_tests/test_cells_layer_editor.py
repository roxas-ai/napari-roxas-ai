from unittest.mock import MagicMock
from types import SimpleNamespace

import numpy as np
from napari.layers import Shapes, Labels

from napari_roxas_ai._edition import CellsLayerEditorWidget
from napari.components._viewer_mouse_bindings import dims_scroll

from napari_roxas_ai._edition._cells_layer_editor import (
    BRUSH_SHORTCUT_MODIFIER,
    MAX_BRUSH_SIZE,
    UNDO_HISTORY_LIMIT,
)


def test_cells_layer_editor_lasso_selection_and_deletion(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()

    # Create dummy cells label data: 2 separated boxes (cells)
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1   # Cell 1 (top-left)
    data[30:40, 30:40] = 1 # Cell 2 (bottom-right)

    cells_layer = viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()

    # Initially lasso container should be hidden
    assert not widget._lasso_container.visible
    assert not widget._lasso_selection_checkbox.visible
    assert not widget._delete_lasso_cells_button.visible

    # Set mode to Vector and enter edit mode
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()

    # Wait for the background vectorization worker to complete
    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers and isinstance(viewer.layers["Cells Modification"], Shapes),
        timeout=5000,
    )

    # In vector edit mode, lasso container is visible with checkbox unchecked
    assert widget._lasso_container.visible
    assert widget._lasso_selection_checkbox.visible
    assert not widget._lasso_selection_checkbox.value
    assert not widget._delete_lasso_cells_button.visible
    assert "Cells Modification" in viewer.layers
    edit_layer = viewer.layers["Cells Modification"]
    assert isinstance(edit_layer, Shapes)
    assert len(edit_layer.data) == 2

    # Toggle lasso select ON
    widget._lasso_selection_checkbox.value = True
    assert widget._delete_lasso_cells_button.visible
    assert "Lasso Selection" in viewer.layers
    lasso_layer = viewer.layers["Lasso Selection"]
    assert lasso_layer.mode == "add_polygon_lasso"
    assert viewer.layers.selection.active == lasso_layer

    # Draw a lasso polygon covering Cell 1 (around [5:15, 5:15])
    lasso_poly = np.array([
        [0, 0],
        [0, 20],
        [20, 20],
        [20, 0]
    ], dtype=np.float32)
    lasso_layer.data = [lasso_poly]

    # Execute deletion
    widget._execute_lasso_deletion()

    # Check that Cell 1 was deleted and only Cell 2 remains
    assert len(edit_layer.data) == 1
    # Checkbox should be reset to False
    assert not widget._lasso_selection_checkbox.value
    assert not widget._delete_lasso_cells_button.visible

    # Apply changes
    widget._apply_cells_geometries()
    assert not widget._lasso_container.visible
    assert "Cells Modification" not in viewer.layers
    assert np.any(cells_layer.data[30:40, 30:40] > 0)
    assert np.all(cells_layer.data[5:15, 5:15] == 0)


def test_lasso_delete_vertex_criteria(make_napari_viewer, qtbot):
    """
    Test specific vertex criteria:
    1. Cell completely inside lasso -> deleted
    2. Cell completely outside lasso -> retained
    3. One cell vertex inside lasso -> deleted
    4. Lasso crosses cell edges but no cell vertex inside -> retained
    5. Cell vertex on lasso boundary -> deleted
    6. Multiple cells inside/outside
    7. Lasso near image boundary
    """
    viewer = make_napari_viewer()
    data = np.zeros((100, 100), dtype=np.uint8)
    data[0:5, 0:5] = 1 # dummy layer
    viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()

    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )

    edit_layer = viewer.layers["Cells Modification"]
    widget._lasso_selection_checkbox.value = True
    lasso_layer = viewer.layers["Lasso Selection"]

    # Define cells:
    # Cell 1: Completely inside lasso [20, 20] to [40, 40] -> polygon [25, 25], [25, 35], [35, 35], [35, 25]
    cell_inside = np.array([[25, 25], [25, 35], [35, 35], [35, 25]], dtype=np.float32)

    # Cell 2: Completely outside lasso -> polygon [70, 70], [70, 80], [80, 80], [80, 70]
    cell_outside = np.array([[70, 70], [70, 80], [80, 80], [80, 70]], dtype=np.float32)

    # Cell 3: One vertex inside lasso -> polygon [38, 38] (inside), [50, 60], [60, 60], [60, 50]
    cell_one_vertex_inside = np.array([[38, 38], [50, 60], [60, 60], [60, 50]], dtype=np.float32)

    # Cell 4: Large cell spanning [10, 28] to [50, 32] (lasso [20, 20] to [40, 40] crosses it, but vertices are at row 10 and row 50, outside lasso)
    cell_crossing_no_vertex = np.array([[10, 28], [10, 32], [50, 32], [50, 28]], dtype=np.float32)

    # Cell 5: Exactly on boundary vertex -> polygon [20, 30] (on boundary of lasso row 20), [15, 25], [15, 35]
    cell_on_boundary = np.array([[20, 30], [15, 25], [15, 35]], dtype=np.float32)

    edit_layer.data = [
        cell_inside,
        cell_outside,
        cell_one_vertex_inside,
        cell_crossing_no_vertex,
        cell_on_boundary,
    ]

    # Lasso polygon from [20, 20] to [40, 40]
    lasso_poly = np.array([
        [20, 20],
        [20, 40],
        [40, 40],
        [40, 20]
    ], dtype=np.float32)
    lasso_layer.data = [lasso_poly]

    widget._execute_lasso_deletion()

    # cell_inside -> deleted
    # cell_outside -> retained
    # cell_one_vertex_inside -> deleted
    # cell_crossing_no_vertex -> retained
    # cell_on_boundary -> deleted
    remaining = edit_layer.data
    assert len(remaining) == 2

    # Verify retained cells are cell_outside and cell_crossing_no_vertex
    assert np.allclose(remaining[0], cell_outside)
    assert np.allclose(remaining[1], cell_crossing_no_vertex)


def test_lasso_near_boundary(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((100, 100), dtype=np.uint8)
    data[0:5, 0:5] = 1
    viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()

    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )

    edit_layer = viewer.layers["Cells Modification"]
    widget._lasso_selection_checkbox.value = True
    lasso_layer = viewer.layers["Lasso Selection"]

    # Lasso at the top-left boundary including (0, 0)
    lasso_poly = np.array([
        [0, 0],
        [0, 10],
        [10, 10],
        [10, 0]
    ], dtype=np.float32)
    lasso_layer.data = [lasso_poly]

    cell_near_0 = np.array([[0, 0], [0, 5], [5, 5], [5, 0]], dtype=np.float32)
    cell_away = np.array([[50, 50], [50, 60], [60, 60], [60, 50]], dtype=np.float32)
    edit_layer.data = [cell_near_0, cell_away]

    widget._execute_lasso_deletion()

    assert len(edit_layer.data) == 1
    assert np.allclose(edit_layer.data[0], cell_away)


def test_overlapping_lassos_deletion(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((100, 100), dtype=np.uint8)
    data[0:5, 0:5] = 1
    viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()

    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )

    edit_layer = viewer.layers["Cells Modification"]
    widget._lasso_selection_checkbox.value = True
    lasso_layer = viewer.layers["Lasso Selection"]

    # Two overlapping lasso polygons:
    # Lasso 1: [10, 10] to [40, 40]
    lasso_1 = np.array([
        [10, 10],
        [10, 40],
        [40, 40],
        [40, 10]
    ], dtype=np.float32)

    # Lasso 2: [30, 30] to [60, 60] (overlaps with lasso_1 in [30, 30]..[40, 40])
    lasso_2 = np.array([
        [30, 30],
        [30, 60],
        [60, 60],
        [60, 30]
    ], dtype=np.float32)

    lasso_layer.data = [lasso_1, lasso_2]

    # Cell in lasso 1 only: [15, 15]..[25, 25]
    cell_lasso1_only = np.array([[15, 15], [15, 25], [25, 25], [25, 15]], dtype=np.float32)
    # Cell in lasso 2 only: [45, 45]..[55, 55]
    cell_lasso2_only = np.array([[45, 45], [45, 55], [55, 55], [55, 45]], dtype=np.float32)
    # Cell in the intersection: [32, 32]..[38, 38]
    cell_intersection = np.array([[32, 32], [32, 38], [38, 38], [38, 32]], dtype=np.float32)
    # Cell outside both: [80, 80]..[90, 90]
    cell_outside = np.array([[80, 80], [80, 90], [90, 90], [90, 80]], dtype=np.float32)

    edit_layer.data = [cell_lasso1_only, cell_lasso2_only, cell_intersection, cell_outside]

    widget._execute_lasso_deletion()

    # All three cells in lasso1, lasso2, and their intersection should be deleted
    assert len(edit_layer.data) == 1
    assert np.allclose(edit_layer.data[0], cell_outside)


def test_layer_visibility_management_on_edit_and_cancel(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    viewer.add_labels(data, name="sample.cells")

    rings_data = np.zeros((50, 50), dtype=np.uint8)
    rings_layer = viewer.add_labels(rings_data, name="sample.rings", visible=True)
    rings_years_layer = viewer.add_points([[10, 10]], name="Rings Years", visible=True)
    other_layer = viewer.add_labels(data.copy(), name="sample.other", visible=True)

    widget = CellsLayerEditorWidget(viewer)
    widget.show()

    assert rings_layer.visible
    assert rings_years_layer.visible
    assert other_layer.visible

    # Enter edit mode
    widget._edit_cells_geometries()

    # .rings and Rings Years should be hidden; other layers remain untouched
    assert not rings_layer.visible
    assert not rings_years_layer.visible
    assert other_layer.visible

    # Cancel editing
    widget._cancel_cells_geometries()

    # Visibility states should be restored
    assert rings_layer.visible
    assert rings_years_layer.visible
    assert other_layer.visible


def test_layer_visibility_management_on_apply_and_initially_hidden(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    viewer.add_labels(data, name="sample.cells")

    rings_data = np.zeros((50, 50), dtype=np.uint8)
    rings_layer = viewer.add_labels(rings_data, name="sample.rings", visible=True)
    # rings years layer is initially hidden
    rings_years_layer = viewer.add_points([[10, 10]], name="Rings Years", visible=False)

    widget = CellsLayerEditorWidget(viewer)
    widget.show()

    # Enter edit mode
    widget._edit_cells_geometries()
    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )
    assert not rings_layer.visible
    assert not rings_years_layer.visible

    # Apply editing
    widget._apply_cells_geometries()

    # .rings restored to True, rings years restored to its original False
    assert rings_layer.visible
    assert not rings_years_layer.visible


def test_delete_selected_cells_button_enabled_disabled_state(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()

    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )

    # Initially hidden and enabled
    assert not widget._delete_lasso_cells_button.visible
    assert widget._delete_lasso_cells_button.enabled

    # Toggle lasso on -> button visible and enabled
    widget._lasso_selection_checkbox.value = True
    assert widget._delete_lasso_cells_button.visible
    assert widget._delete_lasso_cells_button.enabled

    # Add a lasso shape
    lasso_layer = viewer.layers["Lasso Selection"]
    lasso_layer.data = [np.array([[0, 0], [0, 20], [20, 20], [20, 0]], dtype=np.float32)]

    # Mock or wrap _execute_lasso_deletion to check intermediate disabled state if desired, or test direct execution
    btn_disabled_during_exec = False
    original_shapes = viewer.layers["Cells Modification"].data

    def check_state_during_exec():
        nonlocal btn_disabled_during_exec
        btn_disabled_during_exec = not widget._delete_lasso_cells_button.enabled

    # Execute deletion
    widget._execute_lasso_deletion()

    # After deletion, lasso selection checkbox is reset to False, button is hidden and re-enabled
    assert not widget._delete_lasso_cells_button.visible
    assert widget._delete_lasso_cells_button.enabled

    # When lasso is enabled again, button should be visible and enabled
    widget._lasso_selection_checkbox.value = True
    assert widget._delete_lasso_cells_button.visible
    assert widget._delete_lasso_cells_button.enabled


def test_save_all_layers_button_lifecycle_and_callback(make_napari_viewer, qtbot, monkeypatch):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    viewer.add_labels(data, name="sample.cells")

    widget = CellsLayerEditorWidget(viewer)
    widget.show()

    # 1. Start view checks
    assert hasattr(widget, "_save_all_layers_button")
    assert widget._save_all_layers_button.text == "Save All Layers"
    assert widget._save_all_layers_button.visible
    assert hasattr(widget, "_progress_bar")
    assert not widget._progress_bar.visible
    assert widget._progress_bar.label == "Progress"
    assert widget._edition_mode_combo.label == "Editing Mode"
    assert tuple(widget._edition_mode_combo.choices) == ("Vector", "Raster")
    assert widget._edition_mode_combo.value == "Vector"

    # Layout order: save button is after edition mode row, progress bar is after save button
    widget_list = list(widget)
    combo_idx = widget_list.index(widget._edition_mode_row)
    save_btn_idx = widget_list.index(widget._save_all_layers_button)
    prog_bar_idx = widget_list.index(widget._progress_bar)
    assert save_btn_idx == combo_idx + 1
    assert prog_bar_idx == save_btn_idx + 1

    # 2. Check delegate callback when clicked and disabled controls
    assert widget._save_all_layers_button.enabled
    assert widget._edit_cells_geometries_button.enabled
    assert widget._edition_mode_combo.enabled
    assert widget._edition_mode_row.enabled

    saved_how = None
    worker_thread = MagicMock()

    def fake_save_layers(how):
        nonlocal saved_how
        saved_how = how
        widget._saving_widget.worker = MagicMock()
        widget._saving_widget.worker_thread = worker_thread

    monkeypatch.setattr(widget._saving_widget, "_save_layers", fake_save_layers)
    widget._save_all_layers_button.native.click()
    assert saved_how == "all"
    assert not widget._save_all_layers_button.enabled
    assert not widget._edit_cells_geometries_button.enabled
    assert not widget._edition_mode_combo.enabled
    assert not widget._edition_mode_row.enabled

    # Simulate worker completion: the controls are enabled again
    for call in worker_thread.finished.connect.call_args_list:
        call.args[0]()
    assert widget._save_all_layers_button.enabled
    assert widget._edit_cells_geometries_button.enabled
    assert widget._edition_mode_combo.enabled
    assert widget._edition_mode_row.enabled
    assert not widget._progress_bar.visible

    # Progress bar updates
    widget._update_progress(1, 2)
    assert widget._progress_bar.value == 50
    assert widget._progress_bar.visible

    # 3. Enter raster edit mode -> button should be hidden
    widget._edition_mode_combo.value = "Raster"
    widget._edit_cells_geometries()
    assert not widget._save_all_layers_button.visible
    assert not widget._progress_bar.visible

    # Cancel -> button visible again
    widget._cancel_cells_geometries()
    assert widget._save_all_layers_button.visible

    # 4. Enter vector edit mode -> button should be hidden
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()
    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )
    assert not widget._save_all_layers_button.visible

    # Apply -> button visible again
    widget._apply_cells_geometries()
    assert widget._save_all_layers_button.visible


def _start_vector_editing(viewer, qtbot, data):
    viewer.add_labels(data, name="sample.cells")
    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Vector"
    widget._edit_cells_geometries()
    qtbot.wait_until(
        lambda: "Cells Modification" in viewer.layers,
        timeout=5000,
    )
    return widget, viewer.layers["Cells Modification"]


def _start_raster_editing(viewer, data):
    cells_layer = viewer.add_labels(data, name="sample.cells")
    widget = CellsLayerEditorWidget(viewer)
    widget.show()
    widget._edition_mode_combo.value = "Raster"
    widget._edit_cells_geometries()
    return widget, cells_layer, viewer.layers["Cells Modification"]


def test_backspace_triggers_lasso_deletion(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    data[30:40, 30:40] = 1
    widget, edit_layer = _start_vector_editing(viewer, qtbot, data)

    widget._lasso_selection_checkbox.value = True
    lasso_layer = viewer.layers["Lasso Selection"]
    lasso_layer.data = [np.array([[0, 0], [0, 20], [20, 20], [20, 0]], dtype=np.float32)]

    # Both Delete and Backspace (macOS) are bound to the lasso deletion
    keymap = {str(key): func for key, func in viewer.keymap.items()}
    assert "Delete" in keymap
    assert "Backspace" in keymap
    keymap["Backspace"](viewer)

    assert len(edit_layer.data) == 1
    assert not widget._lasso_selection_checkbox.value


def test_undo_button_lifecycle_and_vector_undo(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    data[30:40, 30:40] = 1

    widget = CellsLayerEditorWidget(viewer)
    assert not widget._undo_button.visible

    widget, edit_layer = _start_vector_editing(viewer, qtbot, data)
    assert widget._undo_button.visible
    assert not widget._undo_button.enabled  # nothing to undo yet
    original_shapes = [np.array(d, copy=True) for d in edit_layer.data]

    # Lasso deletion is a single undo step
    widget._lasso_selection_checkbox.value = True
    viewer.layers["Lasso Selection"].data = [
        np.array([[0, 0], [0, 20], [20, 20], [20, 0]], dtype=np.float32)
    ]
    widget._execute_lasso_deletion()
    assert len(edit_layer.data) == 1

    # A newly drawn polygon is another undo step
    edit_layer.add_polygons([np.array([[20, 20], [20, 25], [25, 25], [25, 20]])])
    assert len(edit_layer.data) == 2
    assert len(widget._undo_stack) == 2
    assert widget._undo_button.enabled

    # Undo restores the states in reverse order
    widget._undo_last_change()
    assert len(edit_layer.data) == 1
    widget._undo_last_change()
    assert len(edit_layer.data) == 2
    assert all(np.array_equal(a, b) for a, b in zip(edit_layer.data, original_shapes))
    assert not widget._undo_button.enabled

    # The undo button is hidden and the history cleared after the edit session
    widget._cancel_cells_geometries()
    assert not widget._undo_button.visible
    assert not widget._undo_stack
    assert widget._undo_current is None


def test_raster_undo_uses_labels_paint_history(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    widget, cells_layer, edit_layer = _start_raster_editing(viewer, data)

    assert widget._undo_button.visible
    assert not widget._undo_button.enabled

    edit_layer.brush_size = 3
    edit_layer.paint((30, 30), 1)
    assert edit_layer.data[30, 30] == 1
    assert widget._undo_button.enabled

    widget._undo_last_change()
    assert edit_layer.data[30, 30] == 0
    assert not widget._undo_button.enabled

    widget._apply_cells_geometries()
    assert not widget._undo_button.visible
    assert cells_layer.data[30, 30] == 0


def test_overlapping_polygons_are_combined_on_apply(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((100, 100), dtype=np.uint8)
    data[0:5, 0:5] = 1
    widget, edit_layer = _start_vector_editing(viewer, qtbot, data)
    cells_layer = viewer.layers["sample.cells"]

    # Two overlapping polygons, overlapping in [30, 40] x [30, 40]
    edit_layer.data = [
        np.array([[10, 10], [10, 40], [40, 40], [40, 10]], dtype=np.float32),
        np.array([[30, 30], [30, 60], [60, 60], [60, 30]], dtype=np.float32),
    ]

    widget._apply_cells_geometries()

    # The overlapping area is preserved (union) instead of excluded (XOR)
    assert np.all(cells_layer.data[31:40, 31:40] == 1)
    assert np.all(cells_layer.data[15:25, 15:25] == 1)
    assert np.all(cells_layer.data[45:55, 45:55] == 1)


def test_ctrl_mouse_wheel_adjusts_brush_size(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    widget, _, edit_layer = _start_raster_editing(viewer, data)

    controls = viewer.window._qt_viewer.controls.widgets[edit_layer]
    slider = controls._brush_size_slider_control.brush_size_slider
    assert slider.maximum() == MAX_BRUSH_SIZE

    def wheel(dy, modifiers=(BRUSH_SHORTCUT_MODIFIER,)):
        event = SimpleNamespace(modifiers=modifiers, delta=(0, dy), handled=False)
        widget._on_raster_mouse_wheel(edit_layer, event)
        return event

    edit_layer.mode = "paint"
    edit_layer.brush_size = 10

    # Without CTRL, the wheel does not change the brush size
    assert not wheel(1, modifiers=()).handled
    assert edit_layer.brush_size == 10

    event = wheel(1)
    assert event.handled
    assert edit_layer.brush_size > 10

    # The brush size is clamped between 1 and MAX_BRUSH_SIZE
    for _ in range(100):
        wheel(1)
    assert edit_layer.brush_size == MAX_BRUSH_SIZE
    assert slider.value() == MAX_BRUSH_SIZE
    for _ in range(100):
        wheel(-1)
    assert edit_layer.brush_size == 1

    # Small trackpad deltas accumulate
    for _ in range(10):
        wheel(0.1)
    assert edit_layer.brush_size == 2

    # Works with the eraser too, but not in other modes
    edit_layer.mode = "erase"
    wheel(1)
    assert edit_layer.brush_size == 3
    edit_layer.mode = "pan_zoom"
    assert not wheel(1).handled
    assert edit_layer.brush_size == 3


def test_ctrl_double_click_toggles_eraser_and_brush(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    data[5:15, 5:15] = 1
    widget, _, edit_layer = _start_raster_editing(viewer, data)
    viewer.layers.selection.active = edit_layer

    edit_layer.mode = "erase"
    assert widget._on_raster_double_click in edit_layer.mouse_double_click_callbacks

    # A double-click without CTRL keeps the eraser
    widget._on_raster_double_click(
        edit_layer, SimpleNamespace(modifiers=(), handled=False)
    )
    assert edit_layer.mode == "erase"

    widget._on_raster_double_click(
        edit_layer, SimpleNamespace(modifiers=(BRUSH_SHORTCUT_MODIFIER,), handled=False)
    )
    assert edit_layer.mode == "paint"

    # And back from the brush to the eraser
    widget._on_raster_double_click(
        edit_layer, SimpleNamespace(modifiers=(BRUSH_SHORTCUT_MODIFIER,), handled=False)
    )
    assert edit_layer.mode == "erase"

    # Other tools are not affected
    edit_layer.mode = "fill"
    widget._on_raster_double_click(
        edit_layer, SimpleNamespace(modifiers=(BRUSH_SHORTCUT_MODIFIER,), handled=False)
    )
    assert edit_layer.mode == "fill"
    edit_layer.mode = "paint"

    # CTRL+clicks do not paint/erase, plain clicks still do
    drag_callback = edit_layer.mouse_drag_callbacks[-1]
    ctrl_event = SimpleNamespace(modifiers=(BRUSH_SHORTCUT_MODIFIER,), type="mouse_press")
    gen = drag_callback(edit_layer, ctrl_event)
    assert list(gen) == []


def test_vector_undo_history_keeps_only_last_edits(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((200, 200), dtype=np.uint8)
    data[5:15, 5:15] = 1
    widget, edit_layer = _start_vector_editing(viewer, qtbot, data)

    for i in range(UNDO_HISTORY_LIMIT + 5):
        edit_layer.add_polygons([np.array([[20, i], [20, i + 2], [22, i + 2], [22, i]])])
    assert len(edit_layer.data) == UNDO_HISTORY_LIMIT + 6

    # The oldest edits are dropped from the history
    assert len(widget._undo_stack) == UNDO_HISTORY_LIMIT
    for _ in range(UNDO_HISTORY_LIMIT):
        widget._undo_last_change()
    assert len(edit_layer.data) == 6
    assert not widget._undo_button.enabled


def test_only_ctrl_triggers_raster_shortcuts(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    widget, _, edit_layer = _start_raster_editing(viewer, data)

    # The Cmd key on macOS ("Control") or the Windows key ("Meta") do not trigger the shortcuts
    other_modifier = "Control" if BRUSH_SHORTCUT_MODIFIER == "Meta" else "Meta"
    edit_layer.mode = "erase"
    edit_layer.brush_size = 10
    event = SimpleNamespace(modifiers=(other_modifier,), delta=(0, 1), handled=False)
    widget._on_raster_mouse_wheel(edit_layer, event)
    assert edit_layer.brush_size == 10
    widget._on_raster_double_click(
        edit_layer, SimpleNamespace(modifiers=(other_modifier,), handled=False)
    )
    assert edit_layer.mode == "erase"


def test_dims_scroll_disabled_during_raster_editing(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((50, 50), dtype=np.uint8)
    assert dims_scroll in viewer.mouse_wheel_callbacks

    widget, _, _ = _start_raster_editing(viewer, data)
    assert dims_scroll not in viewer.mouse_wheel_callbacks

    widget._cancel_cells_geometries()
    assert viewer.mouse_wheel_callbacks.count(dims_scroll) == 1

    widget._edit_cells_geometries()
    assert dims_scroll not in viewer.mouse_wheel_callbacks
    widget._apply_cells_geometries()
    assert viewer.mouse_wheel_callbacks.count(dims_scroll) == 1
