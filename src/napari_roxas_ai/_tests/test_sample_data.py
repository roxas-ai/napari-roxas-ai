import numpy as np

from napari_roxas_ai._sample_data import load_sample_data


def test_load_sample_data():
    # Call the function to load sample data
    sample_data = load_sample_data()

    # Ensure we get a list with three elements
    assert isinstance(sample_data, list), "Sample data should be a list"
    assert len(sample_data) == 3, "Sample data should contain three elements"

    # Ensure each entry has the correct format: (data, metadata, layer_type)
    for entry in sample_data:
        assert (
            isinstance(entry, tuple) and len(entry) == 3
        ), "Each entry should be a tuple of (data, metadata, layer_type)"

        data, metadata, layer_type = entry

        # Check data is a NumPy array and not empty
        assert isinstance(
            data, np.ndarray
        ), "Image data should be a NumPy array"
        assert data.size > 0, "Image data should not be empty"

        # Check metadata is a dictionary with a "name" key
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert "name" in metadata, "Metadata should contain a 'name' key"

        # Ensure layer type is either "image" or "labels"
        assert layer_type in {
            "image",
            "labels",
        }, "Layer type should be 'image' or 'labels'"

    print("All sample data tests passed!")
