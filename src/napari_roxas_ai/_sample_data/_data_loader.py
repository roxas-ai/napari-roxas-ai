"""
This module provides sample data to experiment with the plugin
"""

from pathlib import Path

from PIL import Image

from napari_roxas_ai._reader import read_directory

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path(__file__).parent.absolute()


def load_sample_data():
    """
    Load the sample data for the conifer dataset.
    This function reads the sample data files from the specified directory
    and returns a list of layer data tuples.
    """
    # Read the sample data files from the sample_data directory
    layers = read_directory(str(BASE_DIR / "sample_data"))
    return layers
