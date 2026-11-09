import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from napari_roxas_ai._reader import read_cells_file, read_rings_file
from napari_roxas_ai._measurements._sample_measurer import SampleAnalyzer
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
    }


def test_sample_analyzer_conifer_end_to_end(default_config):
    sample_dir = Path(__file__).resolve().parent.parent / "_sample_data" / "sample_data" / "conifer_sample"
    cells_path = sample_dir / "conifer_sample.cells.png"
    rings_path = sample_dir / "conifer_sample.rings.tif"

    cells_data, cells_meta, _ = read_cells_file(str(cells_path))
    rings_data, rings_meta, _ = read_rings_file(str(rings_path))

    config = default_config.copy()
    config["pixels_per_um"] = cells_meta["metadata"]["spatial_resolution"]
    config["sample_type"] = cells_meta["metadata"].get("sample_type", "conifer")

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=cells_data.astype("uint8") * 255,
        rings_table=rings_meta["features"].copy(),
    )
    cells_df, rings_df = analyzer.analyze_sample()

    assert isinstance(cells_df, pd.DataFrame)
    assert isinstance(rings_df, pd.DataFrame)
    assert len(cells_df) == 244
    assert len(rings_df) == 3

    # Check key columns in cells_df
    expected_cell_cols = [
        "id", "XPIX", "YPIX", "lumen_area", "lumen_peri",
        "CWT_pith", "CWT_bark", "CWT_left", "CWT_right", "CWTTAN", "CWTRAD", "CWTALL",
        "CWA", "RWD", "bot_ring_id"
    ]
    for col in expected_cell_cols:
        assert col in cells_df.columns, f"Missing column {col} in cells_df"

    # Check key columns in rings_df
    expected_ring_cols = [
        "id", "MRW", "CWA", "CWTTAN", "CWTRAD", "CWTALL", "RWD", "CNO", "CD", "CTA", "RCTA", "MLA", "MINLA", "MAXLA", "KH", "KS"
    ]
    for col in expected_ring_cols:
        assert col in rings_df.columns, f"Missing column {col} in rings_df"

    # Check valid numeric values
    valid_cwt = cells_df["CWTALL"].dropna()
    assert len(valid_cwt) > 0
    assert (valid_cwt > 0).all()
    assert (cells_df["lumen_area"] > 0).all()


def test_cwt_outlier_filtering(default_config):
    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "id": np.arange(n),
        "CWT_pith": np.random.uniform(2.0, 3.0, n),
        "CWT_bark": np.random.uniform(2.0, 3.0, n),
        "CWT_left": np.random.uniform(2.0, 3.0, n),
        "CWT_right": np.random.uniform(2.0, 3.0, n),
    })

    # Hard limit outlier
    df.loc[5, "CWT_pith"] = 50.0

    # Opposite side ratio outlier (bark > 1.5 * pith)
    df.loc[10, "CWT_bark"] = 8.0
    df.loc[10, "CWT_pith"] = 2.0

    # Adjacent side ratio outlier (left > 3.0 * avg(pith, bark))
    df.loc[15, "CWT_left"] = 12.0
    df.loc[15, "CWT_pith"] = 2.0
    df.loc[15, "CWT_bark"] = 2.0

    analyzer = SampleAnalyzer(
        config=default_config,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
        cells_table=df,
    )
    analyzer._apply_cwt_filters()

    filtered_df = analyzer.cells_table

    # Extreme value filtered by hard limit
    assert np.isnan(filtered_df.loc[5, "CWT_pith"])

    # Opposite side ratio filtered
    assert np.isnan(filtered_df.loc[10, "CWT_bark"])

    # Adjacent side ratio filtered
    assert np.isnan(filtered_df.loc[15, "CWT_left"])


def test_opposite_cwt_percentile_exception(default_config):
    # Setup 100 cells uniformly distributed from 1.0 to 5.0 µm
    # 50th percentile (median) will be ~3.0 µm
    np.random.seed(123)
    n = 100
    df = pd.DataFrame({
        "id": np.arange(n),
        "CWT_pith": np.linspace(1.0, 5.0, n),
        "CWT_bark": np.linspace(1.0, 5.0, n),
        "CWT_left": np.linspace(1.0, 5.0, n),
        "CWT_right": np.linspace(1.0, 5.0, n),
    })

    # Cell 0: Earlywood thin wall (pith=1.0, bark=1.7). Ratio = 1.7 > 1.5.
    # Bark (1.7) <= median (~3.0) -> must be KEPT by percentile exception!
    df.loc[0, "CWT_pith"] = 1.0
    df.loc[0, "CWT_bark"] = 1.7
    df.loc[0, "CWT_left"] = 1.7
    df.loc[0, "CWT_right"] = 1.7

    # Cell 1: Latewood thick wall (pith=2.0, bark=3.6). Ratio = 1.8 > 1.5.
    # Bark (3.6) > median (~3.0) -> must be EXCLUDED (NaN)!
    df.loc[1, "CWT_pith"] = 2.0
    df.loc[1, "CWT_bark"] = 3.6
    df.loc[1, "CWT_left"] = 3.6
    df.loc[1, "CWT_right"] = 3.6

    # Cell 2: Radial earlywood thin wall (left=1.0, right=1.7). Ratio = 1.7 > 1.5.
    # Right (1.7) <= median (~3.0) -> must be KEPT!
    df.loc[2, "CWT_pith"] = 1.7
    df.loc[2, "CWT_bark"] = 1.7
    df.loc[2, "CWT_left"] = 1.0
    df.loc[2, "CWT_right"] = 1.7

    # Cell 3: Radial latewood thick wall (left=2.0, right=3.6). Ratio = 1.8 > 1.5.
    # Right (3.6) > median (~3.0) -> must be EXCLUDED (NaN)!
    df.loc[3, "CWT_pith"] = 3.6
    df.loc[3, "CWT_bark"] = 3.6
    df.loc[3, "CWT_left"] = 2.0
    df.loc[3, "CWT_right"] = 3.6

    config = default_config.copy()
    config["opposite_cwt_percentile_exception"] = 0.50

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
        cells_table=df.copy(),
    )
    analyzer._apply_cwt_filters()
    res = analyzer.cells_table

    # Thin walls kept
    assert not np.isnan(res.loc[0, "CWT_bark"])
    assert np.isclose(res.loc[0, "CWT_bark"], 1.7)
    assert not np.isnan(res.loc[2, "CWT_right"])
    assert np.isclose(res.loc[2, "CWT_right"], 1.7)

    # Thick walls excluded
    assert np.isnan(res.loc[1, "CWT_bark"])
    assert np.isnan(res.loc[3, "CWT_right"])

    # Test shifting threshold to 60% (p60 ~ 3.4) vs 30% (p30 ~ 2.2)
    # With 0.60: a wall of 3.2 (pith=2.0, bark=3.2, ratio=1.6) is <= p60 (3.4) -> KEPT
    df60 = df.copy()
    df60.loc[4, "CWT_pith"] = 2.0
    df60.loc[4, "CWT_bark"] = 3.2
    df60.loc[4, "CWT_left"] = 3.2
    df60.loc[4, "CWT_right"] = 3.2

    config60 = default_config.copy()
    config60["opposite_cwt_percentile_exception"] = 0.60
    analyzer60 = SampleAnalyzer(
        config=config60,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
        cells_table=df60,
    )
    analyzer60._apply_cwt_filters()
    assert not np.isnan(analyzer60.cells_table.loc[4, "CWT_bark"])

    # With 0.50 (median ~3.0): bark=3.2 > 3.0 -> EXCLUDED
    config50 = default_config.copy()
    config50["opposite_cwt_percentile_exception"] = 0.50
    analyzer50 = SampleAnalyzer(
        config=config50,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
        cells_table=df60.copy(),
    )
    analyzer50._apply_cwt_filters()
    assert np.isnan(analyzer50.cells_table.loc[4, "CWT_bark"])

    # Test rounding of float artifact (0.7500000000000001 -> 0.75)
    config_artifact = default_config.copy()
    config_artifact["opposite_cwt_percentile_exception"] = 0.7500000000000001
    analyzer_artifact = SampleAnalyzer(
        config=config_artifact,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=pd.DataFrame(),
    )
    assert analyzer_artifact.opposite_cwt_percentile_exception == 0.75


def test_cluster_cells_and_vessel_grouping_angiosperm(default_config):
    config = default_config.copy()
    config["sample_type"] = "angiosperm"

    # Setup cells table with clusters and rings table
    cells_df = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "cluster": [1, 1, 2],
            "bot_ring_id": [0, 0, 0],
            "lumen_area": [100.0, 150.0, 300.0],
        },
        index=[0, 1, 2],
    )
    rings_df = pd.DataFrame(
        {
            "id": [0],
            "CTA": [10000.0],
        },
        index=[0],
    )

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=np.zeros((10, 10), dtype=np.uint8),
        rings_table=rings_df,
        cells_table=cells_df,
    )

    # 1. Test cell-level cluster sizes (NBRNO, NBRID)
    analyzer._compute_cluster_sizes()
    assert "NBRNO" in analyzer.cells_table.columns
    assert "NBRID" in analyzer.cells_table.columns
    # Cells 0 and 1 belong to group of size 2; Cell 2 is solitary (size 1)
    assert analyzer.cells_table.loc[0, "NBRNO"] == 2
    assert analyzer.cells_table.loc[1, "NBRNO"] == 2
    assert analyzer.cells_table.loc[2, "NBRNO"] == 1
    assert analyzer.cells_table.loc[0, "NBRID"] == [0, 1]
    assert pd.isna(analyzer.cells_table.loc[2, "NBRID"])

    # 2. Test ring-level vessel grouping metrics (RVGI, RVSF, RGSGV)
    analyzer._compute_vessel_grouping_metrics()
    res = analyzer.rings_table
    assert "RVGI" in res.columns
    assert "RVSF" in res.columns
    assert "RGSGV" in res.columns

    # Verify grouping math:
    # 3 total cells in ring 0, 2 clusters -> RVGI = 3 / 2 = 1.5
    # 1 solitary cell out of 3 -> RVSF = (1/3)*100 = 33.333%
    # 1 group with size > 1 (size 2) -> RGSGV = 2.0
    assert np.isclose(res.loc[0, "RVGI"], 1.5)
    assert np.isclose(res.loc[0, "RVSF"], 100.0 / 3.0)
    assert np.isclose(res.loc[0, "RGSGV"], 2.0)


def test_small_object_filtering_and_thin_cell_preservation(default_config):
    # Test shape-aware filtering rule:
    # Exclude if area < 12 or (area <= 16 and max_dim < 10)
    #
    # 1. Noise speck 2x2 (4 px) -> excluded (area < 12)
    # 2. Noise speck 3x3 (9 px) -> excluded (area < 12)
    # 3. 1x10 line (10 px, max_dim=10) -> excluded (area < 12)
    # 4. 2x6 bar (12 px, max_dim=6) -> excluded (area <= 16 and max_dim < 10)
    # 5. Noise speck 4x4 (16 px, max_dim=4) -> excluded (area <= 16 and max_dim < 10)
    # 6. Flat 1x12 line (12 px, max_dim=12) -> preserved (area >= 12 and max_dim >= 10)
    # 7. Thin elongated cell 2x15 (30 px) -> preserved (area > 16)
    # 8. Normal cell 6x6 (36 px) -> preserved (area > 16)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[5:7, 5:7] = 255          # 2x2 (4 px)
    mask[15:18, 5:8] = 255        # 3x3 (9 px)
    mask[25:26, 5:15] = 255       # 1x10 (10 px)
    mask[30:32, 5:11] = 255       # 2x6 (12 px, max_dim=6)
    mask[35:39, 5:9] = 255        # 4x4 (16 px, max_dim=4)
    mask[45:46, 5:17] = 255       # 1x12 (12 px, max_dim=12) -> preserved
    mask[55:57, 10:25] = 255      # 2x15 (30 px) -> preserved
    mask[65:71, 10:16] = 255      # 6x6 (36 px) -> preserved

    config = default_config.copy()
    config["smoothing_kernel_size"] = 1

    analyzer = SampleAnalyzer(
        config=config,
        cells_array=mask.copy(),
        rings_table=pd.DataFrame(),
    )
    cells_df = analyzer.analyze_cells()

    # Exactly the 3 valid objects should be detected
    assert len(cells_df) == 3
    # Cleaned cells array should only retain the valid cells
    assert analyzer.cells_array[5:7, 5:7].sum() == 0
    assert analyzer.cells_array[15:18, 5:8].sum() == 0
    assert analyzer.cells_array[25:26, 5:15].sum() == 0
    assert analyzer.cells_array[30:32, 5:11].sum() == 0
    assert analyzer.cells_array[35:39, 5:9].sum() == 0
    assert analyzer.cells_array[45:46, 5:17].sum() > 0
    assert analyzer.cells_array[55:57, 10:25].sum() > 0
    assert analyzer.cells_array[65:71, 10:16].sum() > 0
