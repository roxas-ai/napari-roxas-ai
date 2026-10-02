import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from napari_roxas_ai._measurements import SingleSampleMeasurementsWidget


def test_single_sample_measurements_widget_structure(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    # Ensure no sample dropdown / combo box exists
    assert not hasattr(widget, "_input_sample_combo")

    # Ensure measurement option spinboxes do not exist on the widget
    assert not hasattr(widget, "_cluster_dbl_cwt_threshold")
    assert not hasattr(widget, "_smoothing_kernel_size")
    assert not hasattr(widget, "_relwidth_cwt_integration")
    assert not hasattr(widget, "_cells_measurements_settings")

    # Ensure the expected widgets are in the container
    assert widget[0] == widget._measure_cells_checkbox
    assert widget[1] == widget._measure_rings_checkbox
    assert widget[2] == widget._run_analysis_button
    assert len(widget) == 3


def test_single_sample_measurements_layer_resolution(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    # Initially no layers
    assert widget._cells_layer is None
    assert widget._rings_layer is None

    # Add cells layer and rings layer
    cells_data = np.zeros((20, 20), dtype=np.uint8)
    rings_data = np.zeros((20, 20), dtype=np.uint8)

    cells_layer = viewer.add_labels(cells_data, name="sample_1.cells")
    rings_layer = viewer.add_labels(rings_data, name="sample_1.rings")

    assert widget._cells_layer == cells_layer
    assert widget._rings_layer == rings_layer


def test_single_sample_measurements_run_analysis_missing_layers(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    with patch("napari_roxas_ai._measurements._single_sample_measurements.show_info") as mock_show_info:
        widget._run_analysis()
        mock_show_info.assert_called_with(
            "Cells layer not found in the viewer. Please load the sample first or disable cells processing."
        )
