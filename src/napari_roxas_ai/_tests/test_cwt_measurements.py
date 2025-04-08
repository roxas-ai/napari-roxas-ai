import numpy as np

from napari_roxas_ai._measurements._cwt_measurements import CellAnalyzer
from napari_roxas_ai._sample_data import load_conifer_sample_data


def test_cell_analyzer():
    # Load sample data (cells image)
    sample_data = load_conifer_sample_data()
    cells_image, metadata, layer_type = sample_data[
        [s[1]["name"] == "conifer_sample.cells" for s in sample_data].index(
            True
        )
    ]  # Cells input

    # Ensure data is correctly loaded
    assert isinstance(
        cells_image, np.ndarray
    ), "Cells data should be a NumPy array"
    assert cells_image.size > 0, "Cells data should not be empty"
    assert layer_type == "labels", "Cells data should have layer type 'labels'"

    # Define a sample configuration
    config = {
        "pixels_per_um": 2.2675,
        "cluster_separation_threshold": 3,  # µm
        "smoothing_kernel_size": 5,
        "integration_interval": 0.75,
        "tangential_angle": 0,
    }

    # Initialize the CellAnalyzer with the configuration
    analyzer = CellAnalyzer(config)
    analyzer.cells_array = cells_image.astype("uint8")

    # Run analysis steps
    analyzer._smooth_image()
    analyzer._find_contours()
    analyzer._analyze_lumina()
    analyzer._analyze_cell_walls()
    analyzer._cluster_cells()
    analyzer._get_results_df()

    print("CellAnalyzer tests passed!")
