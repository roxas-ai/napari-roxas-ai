"""
This module provides sample data to experiment with the plugin
"""

import os

from napari_roxas_ai._reader._reader import read_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_conifer_sample_data():
    """
    Load the sample data for the conifer dataset.
    This function reads the sample data files from the specified directory
    and returns a list of layer data tuples.
    """
    # Read the sample data files from the conifer directory
    layers = read_directory(os.path.join(BASE_DIR, "conifer"))
    return layers
