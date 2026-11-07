import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from napari_roxas_ai._reader import read_cells_file, read_rings_file
from napari_roxas_ai._measurements._sample_measurer import (
    SampleAnalyzer,
    save_cwt_visualization_json,
    load_cwt_visualization_json,
)
from napari_roxas_ai._settings import SettingsManager


@pytest.fixture
def default_config():
    settings = SettingsManager()
    return {
        "pixels_per_um": 2.267574,
        "cluster_dbl_cwt_threshold": settings.get("measurements.cluster_dbl_cwt_threshold"),
        "smoothing_kernel_size": settings.get("measurements.cells_smoothing_kernel_size"),
        "relwidth_cwt_integration": settings.get("measurements.relwidth_cwt_integration"),
        "tangential_angle": settings.get("measurements.cells_tangential_angle"),
        "lower_limit_cwt_iqr_multiplier": settings.get("measurements.lower_limit_cwt_iqr_multiplier"),
        "upper_limit_cwt_iqr_multiplier": settings.get("measurements.upper_limit_cwt_iqr_multiplier"),
        "opposite_cwt_ratio_limit": settings.get("measurements.opposite_cwt_ratio_limit"),
        "opposite_cwt_percentile_exception": settings.get("measurements.opposite_cwt_percentile_exception"),
        "adjacent_cwt_ratio_limit": settings.get("measurements.adjacent_cwt_ratio_limit"),
        "sample_type": "conifer",
        "sample_interval": 25,
        "sample_name": "conifer_sample",
    }


def test_cwt_sampling_and_geometry_conifer_end_to_end(default_config):
    sample_dir = (
        Path(__file__).resolve().parent.parent
        / "_sample_data"
        / "sample_data"
        / "conifer_sample"
    )
    cells_path = sample_dir / "conifer_sample.cells.png"
    rings_path = sample_dir / "conifer_sample.rings.tif"

    cells_data, cells_meta, _ = read_cells_file(str(cells_path))
    rings_data, rings_meta, _ = read_rings_file(str(rings_path))

    config = default_config.copy()
    config["pixels_per_um"] = cells_meta["metadata"]["spatial_resolution"]
    config["sample_type"] = cells_meta["metadata"].get("sample_type", "conifer")
    config["sample_interval"] = 25

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=cells_data.astype("uint8") * 255,
        rings_table=rings_meta["features"].copy(),
    )
    cells_df, rings_df = analyzer.analyze_sample()

    cwt_data = analyzer.get_cwt_visualization_data("conifer_sample")
    assert isinstance(cwt_data, dict)
    assert cwt_data["sample_name"] == "conifer_sample"
    assert cwt_data["sample_interval"] == 25
    assert cwt_data["total_cells"] == len(cells_df)
    assert cwt_data["sampled_cells_count"] > 0
    assert isinstance(cwt_data["circles"], list)
    assert len(cwt_data["circles"]) > 0

    valid_walls = {"pith", "bark", "left", "right"}
    for circle in cwt_data["circles"]:
        assert isinstance(circle["cell_id"], int)
        assert circle["cell_id"] % 25 == 0
        assert circle["wall"] in valid_walls
        assert len(circle["center_yx"]) == 2
        assert isinstance(circle["center_yx"][0], (int, float))
        assert isinstance(circle["center_yx"][1], (int, float))
        assert circle["radius_px"] > 0
        assert circle["radius_um"] > 0
        assert circle["status"] in {"included", "excluded"}
        if circle["status"] == "included":
            assert circle["reason"] == ""
        else:
            assert isinstance(circle["reason"], str)
            assert len(circle["reason"]) > 0


def test_cwt_outlier_filtering_reason_mapping(default_config):
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "id": np.arange(n),
        "CWT_pith": np.random.uniform(2.0, 3.0, n),
        "CWT_bark": np.random.uniform(2.0, 3.0, n),
        "CWT_left": np.random.uniform(2.0, 3.0, n),
        "CWT_right": np.random.uniform(2.0, 3.0, n),
    })

    # Hard limit outlier for cell 0
    df.loc[0, "CWT_pith"] = 50.0

    # Opposite side ratio outlier for cell 10 (bark > 1.5 * pith, within IQR)
    df.loc[10, "CWT_bark"] = 3.5
    df.loc[10, "CWT_pith"] = 2.0

    # Adjacent side ratio outlier for cell 20 (left > 1.4 * avg(pith, bark), opposite left/right < 1.5, within IQR)
    df.loc[20, "CWT_left"] = 3.5
    df.loc[20, "CWT_right"] = 3.4
    df.loc[20, "CWT_pith"] = 2.0
    df.loc[20, "CWT_bark"] = 2.0

    config = default_config.copy()
    config["sample_interval"] = 10  # will sample 0, 10, 20
    config["adjacent_cwt_ratio_limit"] = 1.4

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
        cells_table=df,
    )

    # Populate synthetic circles metadata for sampled cells
    for cell_id in [0, 10, 20]:
        analyzer._cwt_circles_metadata[cell_id] = {
            "pith": {
                "cell_id": cell_id,
                "wall": "pith",
                "center_yx": [10.0, 10.0],
                "radius_px": float(df.loc[cell_id, "CWT_pith"] * analyzer.pixels_per_um),
                "radius_um": float(df.loc[cell_id, "CWT_pith"]),
                "status": "included",
                "reason": "",
            },
            "bark": {
                "cell_id": cell_id,
                "wall": "bark",
                "center_yx": [20.0, 10.0],
                "radius_px": float(df.loc[cell_id, "CWT_bark"] * analyzer.pixels_per_um),
                "radius_um": float(df.loc[cell_id, "CWT_bark"]),
                "status": "included",
                "reason": "",
            },
            "left": {
                "cell_id": cell_id,
                "wall": "left",
                "center_yx": [15.0, 5.0],
                "radius_px": float(df.loc[cell_id, "CWT_left"] * analyzer.pixels_per_um),
                "radius_um": float(df.loc[cell_id, "CWT_left"]),
                "status": "included",
                "reason": "",
            },
            "right": {
                "cell_id": cell_id,
                "wall": "right",
                "center_yx": [15.0, 15.0],
                "radius_px": float(df.loc[cell_id, "CWT_right"] * analyzer.pixels_per_um),
                "radius_um": float(df.loc[cell_id, "CWT_right"]),
                "status": "included",
                "reason": "",
            },
        }

    analyzer._apply_cwt_filters()
    cwt_data = analyzer._build_cwt_visualization_data()

    # Check cell 0 pith (hard limit)
    cell_0_pith = [c for c in cwt_data["circles"] if c["cell_id"] == 0 and c["wall"] == "pith"][0]
    assert cell_0_pith["status"] == "excluded"
    assert "Tukey IQR" in cell_0_pith["reason"] or "limit exceeded" in cell_0_pith["reason"]

    # Check cell 10 bark (opposite wall ratio)
    cell_10_bark = [c for c in cwt_data["circles"] if c["cell_id"] == 10 and c["wall"] == "bark"][0]
    assert cell_10_bark["status"] == "excluded"
    assert "Opposite wall ratio exceeded" in cell_10_bark["reason"]

    # Check cell 20 left (adjacent wall ratio)
    cell_20_left = [c for c in cwt_data["circles"] if c["cell_id"] == 20 and c["wall"] == "left"][0]
    assert cell_20_left["status"] == "excluded"
    assert "Adjacent wall ratio exceeded" in cell_20_left["reason"]


def test_cwt_json_save_load_roundtrip(tmp_path):
    test_data = {
        "sample_name": "Sample_01",
        "sample_interval": 250,
        "total_cells": 12500,
        "sampled_cells_count": 2,
        "circles": [
            {
                "cell_id": 250,
                "wall": "pith",
                "center_yx": [1240.5, 830.2],
                "radius_px": 5.4,
                "radius_um": 2.16,
                "status": "included",
                "reason": "",
            },
            {
                "cell_id": 250,
                "wall": "bark",
                "center_yx": [1275.1, 832.0],
                "radius_px": 14.8,
                "radius_um": 5.92,
                "status": "excluded",
                "reason": "Opposite wall ratio exceeded (2.1x > 1.8x limit)",
            },
        ],
    }

    file_path = tmp_path / "Sample_01.cwt.json"
    save_cwt_visualization_json(file_path, test_data)
    assert file_path.exists()

    loaded_data = load_cwt_visualization_json(file_path)
    assert loaded_data == test_data

    with pytest.raises(FileNotFoundError):
        load_cwt_visualization_json(tmp_path / "non_existent.cwt.json")


def test_single_sample_measurements_cwt_layer_and_json(make_napari_viewer, tmp_path, monkeypatch):
    from copy import deepcopy
    from napari_roxas_ai._measurements import _single_sample_measurements as single_module
    from napari_roxas_ai._measurements._single_sample_measurements import (
        SingleSampleMeasurementsWidget,
        on_cwt_mouse_move,
    )
    from napari_roxas_ai._settings import DEFAULT_SETTINGS
    from napari_roxas_ai._utils._metadata_keys import MEASUREMENT_PARAMETER_KEYS
    import napari.layers

    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    settings_dict = deepcopy(DEFAULT_SETTINGS)
    settings_dict["project_directory"] = str(tmp_path)
    settings_dict["cwt_visualization"]["enabled"] = True
    settings_dict["cwt_visualization"]["sample_interval"] = 100
    settings_dict["cwt_visualization"]["included_color"] = "black"
    settings_dict["cwt_visualization"]["excluded_color"] = "red"
    settings_dict["cwt_visualization"]["edge_width"] = 2.0

    def mock_get(self, key, default=None):
        parts = key.split(".")
        val = settings_dict
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    monkeypatch.setattr(single_module.SettingsManager, "get", mock_get)

    cells_file = tmp_path / "tree1.cells.png"
    cells_file.touch()

    cells_layer = viewer.add_labels(
        np.zeros((50, 50), dtype=np.uint8),
        name="tree1.cells",
        metadata={
            "file_path": str(cells_file),
            "sample_name": "tree1",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    widget._cells_input_layer = cells_layer
    widget._run_config = {key: 1.0 for key in MEASUREMENT_PARAMETER_KEYS}
    widget._run_config.update({
        "spatial_resolution": 1.0,
        "sample_type": "conifer",
        "sample_interval": 100,
    })

    cells_table = pd.DataFrame({
        "id": [1, 2],
        "CWT_pith": [2.0, 3.0],
        "CWT_bark": [2.1, 3.1],
        "CWT_left": [2.2, 3.2],
        "CWT_right": [2.3, 3.3],
    })

    cwt_vis_data = {
        "sample_name": "tree1",
        "sample_interval": 100,
        "total_cells": 2,
        "sampled_cells_count": 1,
        "circles": [
            {
                "cell_id": 1,
                "wall": "pith",
                "center_yx": [10.0, 10.0],
                "radius_px": 5.0,
                "radius_um": 5.0,
                "status": "included",
                "reason": "",
            },
            {
                "cell_id": 1,
                "wall": "bark",
                "center_yx": [20.0, 10.0],
                "radius_px": 8.0,
                "radius_um": 8.0,
                "status": "excluded",
                "reason": "Outlier",
            },
        ],
    }

    widget._add_result_layers(
        cells_table=cells_table,
        rings_table=pd.DataFrame(),
        cleaned_cells_array=None,
        cwt_vis_data=cwt_vis_data,
    )

    # Verify JSON was saved
    cwt_json_path = tmp_path / "tree1.cwt.json"
    assert cwt_json_path.exists()
    loaded_data = load_cwt_visualization_json(cwt_json_path)
    assert loaded_data["sample_name"] == "tree1"
    assert len(loaded_data["circles"]) == 2

    # Verify Shapes layer was added to viewer
    layer_name = "tree1.cwt_circles"
    assert layer_name in viewer.layers
    shapes_layer = viewer.layers[layer_name]
    assert isinstance(shapes_layer, napari.layers.Shapes)
    assert shapes_layer.shape_type == ["ellipse", "ellipse"]
    assert len(shapes_layer.data) == 2
    assert shapes_layer.edge_width[0] == 2.0
    # Check features DataFrame
    assert hasattr(shapes_layer, "features")
    assert len(shapes_layer.features) == 2
    assert "cell_id" in shapes_layer.features.columns
    assert "wall" in shapes_layer.features.columns
    assert "cwt_um" in shapes_layer.features.columns
    # Check unfilled face color and full solid opacity
    assert np.allclose(shapes_layer.face_color[:, 3], 0.0)
    assert shapes_layer.opacity == 1.0
    assert on_cwt_mouse_move in viewer.mouse_move_callbacks

    # Test re-measuring / changing settings (in-place layer update)
    settings_dict["cwt_visualization"]["edge_width"] = 4.0
    settings_dict["cwt_visualization"]["included_color"] = "blue"

    widget._add_result_layers(
        cells_table=cells_table,
        rings_table=pd.DataFrame(),
        cleaned_cells_array=None,
        cwt_vis_data=cwt_vis_data,
    )

    # Must update layer in-place (same layer instance)
    updated_layer = viewer.layers[layer_name]
    assert updated_layer is shapes_layer
    assert updated_layer.edge_width[0] == 4.0


def test_single_sample_measurements_cwt_layer_disabled(make_napari_viewer, tmp_path, monkeypatch):
    from copy import deepcopy
    from napari_roxas_ai._measurements import _single_sample_measurements as single_module
    from napari_roxas_ai._measurements._single_sample_measurements import SingleSampleMeasurementsWidget
    from napari_roxas_ai._settings import DEFAULT_SETTINGS
    from napari_roxas_ai._utils._metadata_keys import MEASUREMENT_PARAMETER_KEYS

    viewer = make_napari_viewer()
    widget = SingleSampleMeasurementsWidget(viewer)

    settings_dict = deepcopy(DEFAULT_SETTINGS)
    settings_dict["project_directory"] = str(tmp_path)
    settings_dict["cwt_visualization"]["enabled"] = False

    def mock_get(self, key, default=None):
        parts = key.split(".")
        val = settings_dict
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    monkeypatch.setattr(single_module.SettingsManager, "get", mock_get)

    cells_file = tmp_path / "tree2.cells.png"
    cells_file.touch()

    cells_layer = viewer.add_labels(
        np.zeros((50, 50), dtype=np.uint8),
        name="tree2.cells",
        metadata={
            "file_path": str(cells_file),
            "sample_name": "tree2",
            "spatial_resolution": 1.0,
            "sample_type": "conifer",
        },
    )
    widget._cells_input_layer = cells_layer
    widget._run_config = {key: 1.0 for key in MEASUREMENT_PARAMETER_KEYS}
    widget._run_config.update({
        "spatial_resolution": 1.0,
        "sample_type": "conifer",
        "sample_interval": 100,
    })

    cwt_vis_data = {
        "sample_name": "tree2",
        "sample_interval": 100,
        "total_cells": 1,
        "sampled_cells_count": 1,
        "circles": [
            {
                "cell_id": 0,
                "wall": "pith",
                "center_yx": [10.0, 10.0],
                "radius_px": 5.0,
                "radius_um": 5.0,
                "status": "included",
                "reason": "",
            }
        ],
    }

    widget._add_result_layers(
        cells_table=pd.DataFrame({"id": [0]}),
        rings_table=pd.DataFrame(),
        cleaned_cells_array=None,
        cwt_vis_data=cwt_vis_data,
    )

    # JSON should still be saved
    assert (tmp_path / "tree2.cwt.json").exists()
    # But shapes layer is not added
    assert "tree2.cwt_circles" not in viewer.layers


def test_batch_sample_measurements_cwt_json_saved(tmp_path, monkeypatch):
    from napari_roxas_ai._measurements._batch_sample_measurements import Worker as BatchWorker
    from napari_roxas_ai._writer import write_single_layer

    sample_name = "batch_sample_01"
    cells_file = tmp_path / f"{sample_name}.cells.png"
    rings_file = tmp_path / f"{sample_name}.rings.tif"

    # Create synthetic cells mask (a small rectangle)
    cells_img = np.zeros((30, 30), dtype=np.uint8)
    cells_img[5:25, 5:25] = 1

    cells_meta = {
        "name": f"{sample_name}.cells",
        "metadata": {
            "spatial_resolution": 1.0,
            "sample_name": sample_name,
            "sample_type": "conifer",
        },
        "features": pd.DataFrame(),
    }
    rings_meta = {
        "name": f"{sample_name}.rings",
        "metadata": {
            "spatial_resolution": 1.0,
            "sample_name": sample_name,
            "sample_type": "conifer",
        },
        "features": pd.DataFrame({"RBXY": [[[0, 0], [0, 29]]], "YEAR": [2020]}),
    }

    write_single_layer(str(cells_file), cells_img, cells_meta)
    write_single_layer(str(rings_file), np.zeros((30, 30), dtype=np.uint8), rings_meta)

    config = {
        "cluster_dbl_cwt_threshold": 3.0,
        "smoothing_kernel_size": 1,
        "relwidth_cwt_integration": 0.75,
        "tangential_angle": 0.0,
        "lower_limit_cwt_iqr_multiplier": 1.5,
        "upper_limit_cwt_iqr_multiplier": 3.0,
        "opposite_cwt_ratio_limit": 1.5,
        "opposite_cwt_percentile_exception": 0.5,
        "adjacent_cwt_ratio_limit": 3.0,
        "sample_interval": 1,
    }

    worker = BatchWorker(
        config=config,
        samples={
            str(tmp_path / sample_name): {
                "cells": str(cells_file),
                "rings": str(rings_file),
            }
        },
        measure_cells=True,
        measure_rings=True,
        update_annotated_images=False,
        update_reference_plots=False,
    )

    worker.run()

    cwt_path = tmp_path / f"{sample_name}.cwt.json"
    assert cwt_path.exists()
    loaded_data = load_cwt_visualization_json(cwt_path)
    assert loaded_data["sample_name"] == sample_name
    assert loaded_data["total_cells"] >= 1
    assert "circles" in loaded_data


def test_single_sample_measurements_worker_emits_cwt_data(default_config):
    from napari_roxas_ai._measurements._single_sample_measurements import Worker

    cells_img = np.zeros((30, 30), dtype=np.uint8)
    cells_img[5:25, 5:25] = 255

    worker = Worker(
        config=default_config,
        cells_array=cells_img,
        rings_table=pd.DataFrame(),
        cells_table=pd.DataFrame(),
        measurement="cells",
    )

    emitted_results = []
    worker.result_ready.connect(lambda *args: emitted_results.append(args))
    worker.run()

    assert len(emitted_results) == 1
    cells_table, rings_table, cleaned_cells_array, cwt_vis_data = emitted_results[0]
    assert not cells_table.empty
    assert isinstance(cwt_vis_data, dict)
    assert "circles" in cwt_vis_data
    assert cwt_vis_data["total_cells"] == len(cells_table)


def test_format_cwt_tooltip():
    from napari_roxas_ai._measurements._single_sample_measurements import (
        format_cwt_tooltip,
    )

    # 1. Included circle
    inc_circle = {
        "cell_id": 250,
        "wall": "pith",
        "radius_um": 2.164,
        "status": "included",
        "reason": "",
    }
    inc_html = format_cwt_tooltip(inc_circle)
    assert "<b>Cell 250</b> (Pith wall)" in inc_html
    assert "<b>CWT:</b> 2.16 µm" in inc_html
    assert "<b>Included</b>" in inc_html
    assert "#28a745" in inc_html
    assert "px" not in inc_html
    assert "Reason:" not in inc_html

    # 2. Excluded circle
    exc_circle = {
        "cell_id": 250,
        "wall": "bark",
        "radius_um": 5.92,
        "status": "excluded",
        "reason": "Opposite wall ratio exceeded (2.1x > 1.8x limit)",
    }
    exc_html = format_cwt_tooltip(exc_circle)
    assert "<b>Cell 250</b> (Bark wall)" in exc_html
    assert "<b>CWT:</b> 5.92 µm" in exc_html
    assert "<b>Excluded</b>" in exc_html
    assert "#d9534f" in exc_html
    assert (
        "<b>Reason:</b> Opposite wall ratio exceeded (2.1x > 1.8x limit)"
        in exc_html
    )
    assert "px" not in exc_html


def test_calculate_cwt_tooltip_pos():
    from qtpy.QtCore import QPoint
    from napari_roxas_ai._measurements._single_sample_measurements import (
        calculate_cwt_tooltip_pos,
        format_cwt_tooltip,
    )

    circle = {
        "cell_id": 100,
        "wall": "pith",
        "radius_um": 2.5,
        "status": "included",
    }
    tooltip_html = format_cwt_tooltip(circle)
    center_pos = QPoint(500, 500)
    radius_screen = 15.0
    margin = 12

    # Pith: entire box placed above circle (top of circle = 500 - 15 = 485)
    pos_pith = calculate_cwt_tooltip_pos(
        center_pos, radius_screen, "pith", tooltip_html, margin=margin
    )
    assert pos_pith.y() < center_pos.y() - radius_screen
    assert pos_pith.x() < center_pos.x()

    # Bark: entire box placed below circle (bottom of circle = 500 + 15 = 515)
    pos_bark = calculate_cwt_tooltip_pos(
        center_pos, radius_screen, "bark", tooltip_html, margin=margin
    )
    assert pos_bark.y() == center_pos.y() + int(radius_screen) + margin
    assert pos_bark.x() == pos_pith.x()  # horizontally centered identical to pith

    # Left: entire box placed to the left of circle, centered vertically
    pos_left = calculate_cwt_tooltip_pos(
        center_pos, radius_screen, "left", tooltip_html, margin=margin
    )
    assert pos_left.x() < center_pos.x() - radius_screen
    assert pos_left.y() < center_pos.y()

    # Right: entire box placed to the right of circle, centered vertically
    pos_right = calculate_cwt_tooltip_pos(
        center_pos, radius_screen, "right", tooltip_html, margin=margin
    )
    assert pos_right.x() == center_pos.x() + int(radius_screen) + margin
    assert pos_right.y() == pos_left.y()  # vertically centered identical to left

    # Excluded circle with long HTML tags: should strip tags and compute reasonable box width
    exc_circle = {
        "cell_id": 250,
        "wall": "left",
        "radius_um": 5.92,
        "status": "excluded",
        "reason": "Opposite wall ratio exceeded (2.1x > 1.8x limit)",
    }
    exc_html = format_cwt_tooltip(exc_circle)
    pos_exc_left = calculate_cwt_tooltip_pos(
        center_pos, radius_screen, "left", exc_html, margin=margin
    )
    assert pos_exc_left.x() < center_pos.x() - radius_screen
    # Box should be reasonably sized (< 400px wide) despite HTML formatting
    assert center_pos.x() - pos_exc_left.x() < 400


def test_on_cwt_mouse_move(make_napari_viewer):
    from unittest.mock import patch
    from qtpy.QtCore import QPoint
    from qtpy.QtWidgets import QToolTip
    from napari_roxas_ai._measurements._single_sample_measurements import (
        on_cwt_mouse_move,
    )

    viewer = make_napari_viewer()
    cwt_data = {
        "sample_name": "sample_01",
        "circles": [
            {
                "cell_id": 50,
                "wall": "pith",
                "center_yx": [100.0, 100.0],
                "radius_px": 10.0,
                "radius_um": 4.5,
                "status": "included",
                "reason": "",
            }
        ],
    }

    shapes_layer = viewer.add_shapes(
        [np.array([[90, 90], [90, 110], [110, 110], [110, 90]])],
        shape_type="ellipse",
        name="sample_01.cwt_circles",
        metadata={"cwt_data": cwt_data},
    )

    class FakeEvent:
        def __init__(self, position):
            self.position = position

    with patch.object(QToolTip, "showText") as mock_show, patch.object(
        QToolTip, "hideText"
    ) as mock_hide:
        # Hover inside circle (100, 100)
        on_cwt_mouse_move(viewer, FakeEvent(position=(100.0, 100.0)))
        assert mock_show.called
        assert not mock_hide.called
        call_args = mock_show.call_args[0]
        assert isinstance(call_args[0], QPoint)
        assert "Cell 50" in call_args[1]

        # Hover again inside same circle -> debounced (no redundant showText call)
        mock_show.reset_mock()
        mock_hide.reset_mock()
        on_cwt_mouse_move(viewer, FakeEvent(position=(101.0, 101.0)))
        assert not mock_show.called
        assert not mock_hide.called

        # Reset mocks and hover far away (200, 200) -> hide once
        mock_show.reset_mock()
        mock_hide.reset_mock()
        on_cwt_mouse_move(viewer, FakeEvent(position=(200.0, 200.0)))
        assert mock_hide.called
        assert not mock_show.called

        # Hover again in empty area -> debounced (no redundant hideText call)
        mock_show.reset_mock()
        mock_hide.reset_mock()
        on_cwt_mouse_move(viewer, FakeEvent(position=(210.0, 210.0)))
        assert not mock_hide.called
        assert not mock_show.called

        # Reset and test with invisible layer
        mock_show.reset_mock()
        mock_hide.reset_mock()
        shapes_layer.visible = False
        on_cwt_mouse_move(viewer, FakeEvent(position=(100.0, 100.0)))
        assert not mock_show.called


def test_samples_loading_widget_cwt_circles_layer(make_napari_viewer, tmp_path, monkeypatch):
    from copy import deepcopy
    from napari_roxas_ai._loading import _samples_loading_widget as loader_module
    from napari_roxas_ai._loading._samples_loading_widget import SamplesLoadingWidget
    from napari_roxas_ai._measurements import save_cwt_visualization_json
    from napari_roxas_ai._settings import DEFAULT_SETTINGS

    viewer = make_napari_viewer()
    loader = SamplesLoadingWidget(viewer)
    loader.project_directory = str(tmp_path)

    sample_name = "sample_test"
    cwt_file = tmp_path / f"{sample_name}.cwt.json"
    cwt_data = {
        "sample_name": sample_name,
        "sample_interval": 50,
        "total_cells": 10,
        "sampled_cells_count": 1,
        "circles": [
            {
                "cell_id": 1,
                "wall": "pith",
                "center_yx": [10.0, 20.0],
                "radius_px": 5.0,
                "radius_um": 2.5,
                "status": "included",
                "reason": "",
            }
        ],
    }
    save_cwt_visualization_json(cwt_file, cwt_data)

    settings_dict = deepcopy(DEFAULT_SETTINGS)
    settings_dict["cwt_visualization"]["enabled"] = True

    def mock_get(self, key, default=None):
        parts = key.split(".")
        val = settings_dict
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    monkeypatch.setattr(loader_module.SettingsManager, "get", mock_get)

    # Add a cells layer to match
    viewer.add_labels(
        np.zeros((50, 50), dtype=np.uint8),
        name=f"{sample_name}.cells",
        metadata={"sample_name": sample_name},
    )

    class FakeSelect:
        value = [sample_name]

    loader._sample_select_widget = FakeSelect()

    # Trigger loading finished
    loader._on_sample_loading_finished()

    cwt_layer_name = f"{sample_name}.cwt_circles"
    assert cwt_layer_name in viewer.layers
    cwt_layer = viewer.layers[cwt_layer_name]
    assert len(cwt_layer.data) == 1
    assert cwt_layer.metadata["sample_name"] == sample_name

    # Test when disabled in settings: layer should not be loaded
    viewer.layers.remove(cwt_layer)
    settings_dict["cwt_visualization"]["enabled"] = False
    loader._on_sample_loading_finished()
    assert cwt_layer_name not in viewer.layers


def test_cells_editor_cwt_circles_lifecycle(make_napari_viewer, tmp_path):
    from napari_roxas_ai._edition._cells_layer_editor import CellsLayerEditorWidget

    viewer = make_napari_viewer()
    sample_name = "sample_test"

    cells_layer = viewer.add_labels(
        np.zeros((50, 50), dtype=np.uint8),
        name=f"{sample_name}.cells",
        metadata={"sample_name": sample_name},
    )
    cwt_layer = viewer.add_shapes(
        [np.array([[5, 5], [5, 15], [15, 15], [15, 5]])],
        shape_type="ellipse",
        name=f"{sample_name}.cwt_circles",
        metadata={"sample_name": sample_name},
    )

    editor = CellsLayerEditorWidget(viewer)
    editor._edition_mode_combo.value = "Raster"

    # 1. Edit mode: CWT circles should be hidden
    editor._edit_cells_geometries()
    assert not cwt_layer.visible

    # 2. Cancel: CWT circles visibility should be restored
    editor._cancel_cells_geometries()
    assert cwt_layer.visible

    # 3. Edit again and Apply: CWT circles layer should be removed from viewer
    editor._edit_cells_geometries()
    editor._apply_cells_geometries()
    assert f"{sample_name}.cwt_circles" not in viewer.layers


def test_rings_editor_cwt_circles_lifecycle(make_napari_viewer):
    from napari_roxas_ai._edition._rings_layer_editor import RingsLayerEditorWidget

    viewer = make_napari_viewer()
    sample_name = "sample_test"

    rings_layer = viewer.add_labels(
        np.zeros((50, 50), dtype=np.uint8),
        name=f"{sample_name}.rings",
        metadata={"sample_name": sample_name, "rings_outmost_complete_year": 2020},
        features=pd.DataFrame({
            "RBXY": [[[0.0, 0.0], [0.0, 49.0]], [[25.0, 0.0], [25.0, 49.0]]],
            "YEAR": [2019, 2020],
            "enabled": [True, True],
        }),
    )
    cwt_layer = viewer.add_shapes(
        [np.array([[5, 5], [5, 15], [15, 15], [15, 5]])],
        shape_type="ellipse",
        name=f"{sample_name}.cwt_circles",
        metadata={"sample_name": sample_name},
    )

    editor = RingsLayerEditorWidget(viewer)

    # 1. Edit mode: CWT circles should be hidden
    editor._edit_rings_geometries()
    assert not cwt_layer.visible

    # 2. Cancel: CWT circles visibility should be restored
    editor._cancel_rings_geometries()
    editor._restore_original_visibility()
    assert cwt_layer.visible


def test_write_cells_file_deletes_stale_cwt_json(tmp_path, monkeypatch):
    from copy import deepcopy
    from napari_roxas_ai._writer import _writer as writer_module
    from napari_roxas_ai._writer._writer import write_cells_file
    from napari_roxas_ai._measurements import save_cwt_visualization_json
    from napari_roxas_ai._settings import DEFAULT_SETTINGS

    settings_dict = deepcopy(DEFAULT_SETTINGS)
    settings_dict["project_directory"] = str(tmp_path)

    def mock_get(self, key, default=None):
        parts = key.split(".")
        val = settings_dict
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    monkeypatch.setattr(writer_module.SettingsManager, "get", mock_get)

    sample_name = "sample_01"
    sample_stem_path = sample_name
    cwt_file = tmp_path / f"{sample_name}.cwt.json"
    save_cwt_visualization_json(cwt_file, {"sample_name": sample_name, "circles": []})
    assert cwt_file.exists()

    data = np.zeros((20, 20), dtype=np.uint8)

    # 1. Write cells with empty features (e.g. after manual editing without measurements) -> should delete stale .cwt.json
    meta_empty = {
        "metadata": {
            "sample_name": sample_name,
            "sample_stem_path": sample_stem_path,
        },
        "features": pd.DataFrame(),
    }
    write_cells_file(str(tmp_path / f"{sample_name}.cells.png"), data, meta_empty)
    assert not cwt_file.exists()

    # 2. Write cells with valid features -> should preserve .cwt.json if present
    save_cwt_visualization_json(cwt_file, {"sample_name": sample_name, "circles": []})
    assert cwt_file.exists()
    meta_with_features = {
        "metadata": {
            "sample_name": sample_name,
            "sample_stem_path": sample_stem_path,
        },
        "features": pd.DataFrame({"id": [1], "area": [10.0]}),
    }
    write_cells_file(str(tmp_path / f"{sample_name}.cells.png"), data, meta_with_features)
    assert cwt_file.exists()
