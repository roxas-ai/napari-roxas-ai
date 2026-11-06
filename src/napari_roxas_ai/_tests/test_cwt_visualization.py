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
        "adjacent_cwt_ratio_limit": settings.get("measurements.adjacent_cwt_ratio_limit"),
        "sample_type": "conifer",
        "sample_interval": 50,
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
    config["sample_interval"] = 50

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=cells_data.astype("uint8") * 255,
        rings_table=rings_meta["features"].copy(),
    )
    cells_df, rings_df = analyzer.analyze_sample()

    cwt_data = analyzer.get_cwt_visualization_data("conifer_sample")
    assert isinstance(cwt_data, dict)
    assert cwt_data["sample_name"] == "conifer_sample"
    assert cwt_data["sample_interval"] == 50
    assert cwt_data["total_cells"] == len(cells_df)
    assert cwt_data["sampled_cells_count"] > 0
    assert isinstance(cwt_data["circles"], list)
    assert len(cwt_data["circles"]) > 0

    valid_walls = {"pith", "bark", "left", "right"}
    for circle in cwt_data["circles"]:
        assert isinstance(circle["cell_id"], int)
        assert circle["cell_id"] % 50 == 0
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
