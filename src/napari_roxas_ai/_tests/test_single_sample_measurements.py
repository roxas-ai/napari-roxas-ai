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
    assert widget._scan_layer is None

    # Add cells, rings, and scan layers
    cells_data = np.zeros((20, 20), dtype=np.uint8)
    rings_data = np.zeros((20, 20), dtype=np.uint8)
    scan_data = np.zeros((20, 20), dtype=np.uint8)

    cells_layer = viewer.add_labels(cells_data, name="sample_1.cells")
    rings_layer = viewer.add_labels(rings_data, name="sample_1.rings")
    scan_layer = viewer.add_image(scan_data, name="sample_1.scan")

    assert widget._cells_layer == cells_layer
    assert widget._rings_layer == rings_layer
    assert widget._scan_layer == scan_layer


def test_single_sample_measurements_add_result_layers_exports_annotated_image(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    scan_file = tmp_path / "sample_1.scan.jpg"
    scan_file.write_bytes(b"dummy")

    scan_layer = viewer.add_image(
        np.zeros((10, 10)),
        name="sample_1.scan",
        metadata={"file_path": str(scan_file), "sample_name": "sample_1"},
    )
    rings_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.rings",
        metadata={
            "file_path": str(tmp_path / "sample_1.rings.png"),
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    widget._rings_input_layer = rings_layer
    widget._run_config = {}

    rings_table = pd.DataFrame(
        {"RBXY": [[[0, 0], [0, 9]]], "YEAR": [2020]}
    )

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.write_single_layer"
    ), patch(
        "napari_roxas_ai._measurements._single_sample_measurements.save_annotated_scan_image"
    ) as mock_save_annotated, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.settings"
    ) as mock_settings:
        mock_settings.get.side_effect = lambda key: {
            "project_directory": str(tmp_path),
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.cells_file_extension": [".cells"],
        }.get(key, "")

        widget._add_result_layers(cells_table=pd.DataFrame(), rings_table=rings_table)

        mock_save_annotated.assert_called_once()
        call_kwargs = mock_save_annotated.call_args[1]
        assert call_kwargs["scan_path"] == str(scan_file)
        assert call_kwargs["annotated_path"] == str(tmp_path / "sample_1_annotated.jpg")


def test_single_sample_measurements_add_result_layers_skips_annotated_if_rings_missing(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    scan_file = tmp_path / "sample_1.scan.jpg"
    scan_file.write_bytes(b"dummy")

    viewer.add_image(
        np.zeros((10, 10)),
        name="sample_1.scan",
        metadata={"file_path": str(scan_file), "sample_name": "sample_1"},
    )
    widget._rings_input_layer = None
    widget._run_config = {}

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.save_annotated_scan_image"
    ) as mock_save_annotated, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.settings"
    ) as mock_settings:
        mock_settings.get.side_effect = lambda key: {
            "project_directory": str(tmp_path),
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.cells_file_extension": [".cells"],
        }.get(key, "")

        widget._add_result_layers(cells_table=pd.DataFrame(), rings_table=pd.DataFrame())

        mock_save_annotated.assert_not_called()


def test_single_sample_measurements_cells_only_passes_rings_when_available(
    make_napari_viewer, tmp_path
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    cells_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.cells",
        metadata={
            "file_path": str(tmp_path / "sample_1.cells.png"),
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    rings_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.rings",
        metadata={
            "file_path": str(tmp_path / "sample_1.rings.png"),
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    rings_df = pd.DataFrame({"RBXY": [[[0, 0], [0, 9]]], "YEAR": [2020]})
    rings_layer.features = rings_df

    widget._measure_cells_checkbox.value = True
    widget._measure_rings_checkbox.value = False

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.Worker"
    ) as mock_worker_cls, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.QThread"
    ):
        widget._run_analysis()

        mock_worker_cls.assert_called_once()
        args = mock_worker_cls.call_args[0]
        # args: config, cells_array, rings_table, cells_table, measurement
        assert args[4] == "cells"
        pd.testing.assert_frame_equal(args[2], rings_df)


def test_single_sample_measurements_run_analysis_missing_layers(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    with patch("napari_roxas_ai._measurements._single_sample_measurements.show_info") as mock_show_info:
        widget._run_analysis()
        mock_show_info.assert_called_with(
            "Cells layer not found in the viewer. Please load the sample first or disable cells processing."
        )
