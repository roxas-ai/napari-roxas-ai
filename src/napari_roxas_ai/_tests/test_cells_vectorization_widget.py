import numpy as np
from napari.layers import Shapes

from napari_roxas_ai._conversion import cells_vectorization_widget


def test_cells_vectorization_widget(make_napari_viewer, qtbot):
    # Create a napari viewer
    viewer = make_napari_viewer()

    # Create a simple label layer with some labeled regions
    label_data = np.array(
        [[0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [1, 0, 0, 1, 1], [1, 1, 0, 1, 1]],
        dtype=int,
    )
    label_layer = viewer.add_labels(label_data)

    # Instantiate the widget
    widget = cells_vectorization_widget()

    # Execute widget function (triggers worker)
    widget(label_layer, 0.5, 0.5)

    # Wait for the worker to finish
    qtbot.wait_until(
        lambda: any(isinstance(layer, Shapes) for layer in viewer.layers),
        timeout=5000,
    )

    # Get the newly added Shapes layer
    shapes_layer = next(
        layer for layer in viewer.layers if isinstance(layer, Shapes)
    )

    # Ensure output is a Shapes layer
    assert isinstance(shapes_layer, Shapes), "Output should be a Shapes layer"

    # Ensure some shapes were detected and created
    assert (
        len(shapes_layer.data) > 0
    ), "Shapes layer should contain at least one shape"

    # Ensure shapes have correct properties
    for shape in shapes_layer.data:
        assert (
            shape.shape[1] == 2
        ), "Each shape should contain (x, y) coordinates"

    # Ensure assigned face color is a valid vispy color
    assert shapes_layer.face_color is not None, "Face color should be assigned"

    print("All tests passed!")
