import numpy as np
from napari.layers import Shapes, Labels

from napari_roxas_ai._edition import CellsLayerEditorWidget


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

    # Set mode to Edit As Vector and enter edit mode
    widget._edition_mode_combo.value = "Edit As Vector"
    widget._edit_cells_geometries()

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
    widget._edition_mode_combo.value = "Edit As Vector"
    widget._edit_cells_geometries()

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
    widget._edition_mode_combo.value = "Edit As Vector"
    widget._edit_cells_geometries()

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
    assert not rings_layer.visible
    assert not rings_years_layer.visible

    # Apply editing
    widget._apply_cells_geometries()

    # .rings restored to True, rings years restored to its original False
    assert rings_layer.visible
    assert not rings_years_layer.visible
