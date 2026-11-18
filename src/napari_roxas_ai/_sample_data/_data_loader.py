"""
This module provides sample data to experiment with the plugin
"""

from pathlib import Path

from PIL import Image

from napari_roxas_ai._reader import read_directory

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None

BASE_DIR = Path(__file__).parent.absolute()


def _layer_order_key(layer_tuple):
    """Sort key to ensure layers are loaded from bottom to top: scan -> cells -> rings."""
    _, add_kwargs, layer_type = layer_tuple
    name = str(add_kwargs.get("name", "")).lower() if isinstance(add_kwargs, dict) else ""
    if layer_type == "image" or "scan" in name:
        return 0
    if "cells" in name:
        return 1
    if "rings" in name:
        return 2
    return 3


def load_sample_data():
    """
    Load the sample data for the conifer dataset.
    This function reads the sample data files from the specified directory
    and returns a list of layer data tuples ordered from bottom to top:
    input image (scan) -> cells -> rings.
    """
    # Read the sample data files from the sample_data directory
    layers = read_directory(str(BASE_DIR / "sample_data"))
    return sorted(layers, key=_layer_order_key)
