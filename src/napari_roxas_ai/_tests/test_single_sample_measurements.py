from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from napari_roxas_ai._job_runner import _runner
from napari_roxas_ai._jobs import Progress
from napari_roxas_ai._measurements import (
    SingleSampleMeasurementsWidget,
    _single_sample_measurements,
)


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
    assert widget[2] == widget._check_settings_button
    assert widget[3] == widget._run_analysis_button
    assert widget._check_settings_button.text == "Check Settings"
    assert widget._run_analysis_button.text == "Measure"
    assert len(widget) == 4


def test_single_sample_measurements_open_settings(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    # Test the actual _open_settings method adds/shows the dock widget and sets it floating
    widget._check_settings_button.changed(True)
    assert "ZZ – Settings" in viewer.window._dock_widgets or "ZZ – Settings" in getattr(viewer.window, "dock_widgets", {})


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


def test_single_sample_measurements_run_analysis_missing_layers(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    with patch("napari_roxas_ai._measurements._single_sample_measurements.show_info") as mock_show_info:
        widget._run_analysis()
        mock_show_info.assert_called_with(
            "Cells layer not found in the viewer. Please load the cells layer first or uncheck 'Measure Cells'."
        )


def test_single_sample_measurements_asks_for_sample_if_several_loaded(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    viewer.add_labels(np.zeros((10, 10), dtype=np.uint8), name="sample_1.cells")
    viewer.add_labels(np.zeros((10, 10), dtype=np.uint8), name="sample_1.rings")
    cells_2 = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8), name="sample_2.cells"
    )
    rings_2 = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8), name="sample_2.rings"
    )
    scan_2 = viewer.add_image(np.zeros((10, 10)), name="sample_2.scan")

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.QInputDialog.getItem",
        return_value=("sample_2", True),
    ) as mock_get_item:
        assert widget._choose_sample()

    assert mock_get_item.call_args[0][3] == ["sample_1", "sample_2"]
    assert widget._cells_layer is cells_2
    assert widget._rings_layer is rings_2
    assert widget._scan_layer is scan_2



# --- Measuring ---

OPERATIONS = "napari_roxas_ai._measurements._operations"
WIDGET = "napari_roxas_ai._measurements._single_sample_measurements"
RINGS_TABLE = pd.DataFrame({"RBXY": [[[0, 0], [0, 9]]], "YEAR": [2020]})
CELLS_TABLE = pd.DataFrame({"CA": [12.0]})


@pytest.fixture
def project(tmp_path, monkeypatch):
    """Point the project directory (settings singleton) to a temp dir."""
    settings = _single_sample_measurements.settings
    original_get = settings.get
    monkeypatch.setattr(
        settings,
        "get",
        lambda key, *args: (
            str(tmp_path) if key == "project_directory" else original_get(key, *args)
        ),
    )
    return tmp_path


@pytest.fixture
def measuring(monkeypatch):
    """Replace the analysis and the file writing; record what they get."""
    calls = {"measure": [], "saved": [], "annotated": []}

    def fake_measure_sample(config, measurement, cells_array, rings_table=None):
        calls["measure"].append(measurement)
        cells = CELLS_TABLE if measurement in ("both", "cells") else pd.DataFrame()
        rings = RINGS_TABLE if measurement in ("both", "rings") else pd.DataFrame()
        cleaned = cells_array if measurement != "rings" else None
        return cells, rings, cleaned

    def fake_save_layers(snapshots, project_dir):
        calls["saved"].extend(snapshot.name for snapshot in snapshots)
        yield Progress(1, 1)

    def fake_annotated(sample_path, rings_features):
        calls["annotated"].append(sample_path)
        return f"{sample_path}_annotated.jpg"

    monkeypatch.setattr(f"{OPERATIONS}.measure_sample", fake_measure_sample)
    monkeypatch.setattr(f"{OPERATIONS}.save_layers", fake_save_layers)
    monkeypatch.setattr(f"{OPERATIONS}.save_sample_annotated_image", fake_annotated)
    return calls


def add_sample_layer(viewer, kind, **features):
    layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name=f"sample_1.{kind}",
        metadata={
            "sample_name": "sample_1",
            "sample_stem_path": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    if features:
        layer.features = pd.DataFrame(features)
    return layer


def test_measured_layers_are_written_and_marked_saved(
    make_napari_viewer, sync_runner, project, measuring
):
    viewer = make_napari_viewer()
    cells = add_sample_layer(viewer, "cells")
    rings = add_sample_layer(viewer, "rings", RBXY=[[[0, 0], [0, 9]]], YEAR=[2019])
    widget = SingleSampleMeasurementsWidget(viewer)

    widget._run_analysis_button.press()

    assert measuring["saved"] == ["sample_1.cells", "sample_1.rings"]
    pd.testing.assert_frame_equal(cells.features, CELLS_TABLE)
    pd.testing.assert_frame_equal(rings.features, RINGS_TABLE)
    assert cells.metadata["_is_dirty"] is False
    assert rings.metadata["_is_dirty"] is False
    assert "meas_created_at" in rings.metadata
    # One annotated image, next to the sample
    assert len(measuring["annotated"]) == 1
    assert widget._run_analysis_button.enabled


def test_no_annotated_image_without_rings(
    make_napari_viewer, sync_runner, project, measuring
):
    viewer = make_napari_viewer()
    add_sample_layer(viewer, "cells")
    widget = SingleSampleMeasurementsWidget(viewer)
    widget._measure_rings_checkbox.value = False

    widget._run_analysis_button.press()

    assert measuring["saved"] == ["sample_1.cells"]
    assert measuring["annotated"] == []


def test_cells_only_draws_the_annotated_image_with_the_loaded_rings(
    make_napari_viewer, sync_runner, project, measuring
):
    viewer = make_napari_viewer()
    add_sample_layer(viewer, "cells")
    add_sample_layer(viewer, "rings", RBXY=[[[0, 0], [0, 9]]], YEAR=[2019])
    widget = SingleSampleMeasurementsWidget(viewer)
    widget._measure_rings_checkbox.value = False

    widget._run_analysis_button.press()

    assert measuring["measure"] == ["cells"]
    # Only the cells are written, the annotated image still shows the rings
    assert measuring["saved"] == ["sample_1.cells"]
    assert len(measuring["annotated"]) == 1


def test_failed_measurement_reports_and_leaves_the_widget_usable(
    make_napari_viewer, sync_runner, project, monkeypatch
):
    viewer = make_napari_viewer()
    cells = add_sample_layer(viewer, "cells")
    add_sample_layer(viewer, "rings", RBXY=[[[0, 0], [0, 9]]], YEAR=[2019])
    widget = SingleSampleMeasurementsWidget(viewer)

    def failing_measure_sample(*args):
        raise ValueError("no rings found")

    monkeypatch.setattr(f"{OPERATIONS}.measure_sample", failing_measure_sample)
    errors = []
    monkeypatch.setattr(_runner, "show_error", errors.append)

    widget._run_analysis_button.press()

    assert errors == ["Measuring failed: no rings found"]
    assert cells.metadata.get("_is_dirty", True)  # nothing was written
    assert widget._run_analysis_button.enabled


def test_cancelled_sample_choice_does_nothing(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    viewer.add_labels(np.zeros((10, 10), dtype=np.uint8), name="sample_1.cells")
    viewer.add_labels(np.zeros((10, 10), dtype=np.uint8), name="sample_2.cells")

    with patch(
        f"{WIDGET}.QInputDialog.getItem", return_value=("", False)
    ), patch(f"{WIDGET}.start_measuring") as mock_start:
        widget._run_analysis_button.press()

    mock_start.assert_not_called()
    assert widget._run_analysis_button.enabled
<<<<<<< HEAD
=======
    assert widget._run_analysis_button.text == "Measure"


def test_single_sample_measurements_add_result_layers_project_dir_none_uses_fallback(
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
    rings_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.rings",
        metadata={
            "file_path": str(tmp_path / "sample_1.rings.png"),
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
            "_is_dirty": True,
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
            "project_directory": None,
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.cells_file_extension": [".cells"],
        }.get(key, "")

        widget._add_result_layers(cells_table=pd.DataFrame(), rings_table=rings_table)

        mock_save_annotated.assert_called_once()
        call_kwargs = mock_save_annotated.call_args[1]
        assert call_kwargs["scan_path"] == str(scan_file)
        assert call_kwargs["annotated_path"] == str(tmp_path / "sample_1_annotated.jpg")
        assert rings_layer.metadata["_is_dirty"] is False


def test_single_sample_measurements_add_result_layers_project_dir_none_no_fallback(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    cells_layer = viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.cells",
        metadata={
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    widget._cells_input_layer = cells_layer
    widget._run_config = {}

    cells_table = pd.DataFrame({"id": [1], "area": [50.0]})

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.show_info"
    ) as mock_show_info, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.settings"
    ) as mock_settings:
        mock_settings.get.side_effect = lambda key: {
            "project_directory": None,
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.cells_file_extension": [".cells"],
        }.get(key, "")

        widget._add_result_layers(cells_table=cells_table, rings_table=pd.DataFrame())

        mock_show_info.assert_called_once()
        assert "Project directory is not configured" in mock_show_info.call_args[0][0]


def test_single_sample_measurements_run_analysis_warns_when_project_dir_none_and_no_file_path(
    make_napari_viewer,
):
    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    viewer.add_labels(
        np.zeros((10, 10), dtype=np.uint8),
        name="sample_1.cells",
        metadata={
            "sample_name": "sample_1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    widget._measure_cells_checkbox.value = True
    widget._measure_rings_checkbox.value = False

    with patch(
        "napari_roxas_ai._measurements._single_sample_measurements.show_info"
    ) as mock_show_info, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.Worker"
    ) as mock_worker_cls, patch(
        "napari_roxas_ai._measurements._single_sample_measurements.settings"
    ) as mock_settings:
        mock_settings.get.side_effect = lambda key: {
            "project_directory": None,
            "file_extensions.scan_file_extension": [".scan"],
            "file_extensions.rings_file_extension": [".rings"],
            "file_extensions.cells_file_extension": [".cells"],
        }.get(key, "")

        widget._run_analysis()

        mock_worker_cls.assert_not_called()
        mock_show_info.assert_called_once()
        assert "Project directory is not configured" in mock_show_info.call_args[0][0]
        assert widget._run_analysis_button.enabled
        assert widget._run_analysis_button.text == "Measure"
>>>>>>> develop
