"""
This module contains a napari widget using a thread worker to perform vectorization.

References:
- Widget specification: https://napari.org/stable/plugins/guides.html?#widgets
- magicgui docs: https://pyapp-kit.github.io/magicgui/

"""

import random
from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory
from napari.layers import Shapes
from skimage.measure import approximate_polygon, find_contours
from vispy.color import get_color_names

if TYPE_CHECKING:
    import napari

import napari
from napari.qt.threading import thread_worker


@thread_worker
def vectorizer_worker(label_data, contours_tolerance, polygon_tolerance):
    """Thread worker for vectorization."""
    padding_size = 1
    contours = [
        np.maximum(contour - 1, 0)
        for contour in find_contours(
            np.pad(label_data, padding_size), contours_tolerance
        )
    ]
    polygons = [
        approximate_polygon(contour, polygon_tolerance) for contour in contours
    ]
    return polygons


@magic_factory(
    contours_tolerance={
        "widget_type": "FloatSpinBox",
        "value": 0.5,
        "min": 0.00,
        "max": 0.99,
    },
    polygon_tolerance={
        "widget_type": "FloatSpinBox",
        "value": 0.5,
        "min": 0.00,
    },
)
def cells_vectorization_widget(
    label_layer: "napari.layers.Labels",
    contours_tolerance: "float",
    polygon_tolerance: "float",
) -> "napari.layers.Shapes":
    """Widget function to vectorize label layer asynchronously."""

    def update_shapes(polygons):
        """Callback function to update shapes layer."""
        viewer = napari.current_viewer()
        shape_layer = Shapes(
            polygons,
            shape_type="polygon",
            face_color=random.choice(get_color_names()),
        )
        viewer.add_layer(shape_layer)

    worker = vectorizer_worker(
        label_layer.data, contours_tolerance, polygon_tolerance
    )
    worker.returned.connect(update_shapes)
    worker.start()

    return None
