"""
All Colormap Related Utils
"""

import colorsys
from collections import defaultdict

import numpy as np
from napari.utils.colormaps import DirectLabelColormap


def make_binary_labels_colormap():
    # Generate a bright, highly saturated random color
    h = np.random.rand()  # Random hue (0 to 1)
    s = 0.9  # High saturation
    v = 1.0  # Maximum brightness
    r, g, b = colorsys.hsv_to_rgb(h, s, v)  # Convert HSV to RGB

    # Define the color mapping for labels
    color_dict = defaultdict(
        lambda: np.array([0, 0, 0, 0])
    )  # Default: transparent
    color_dict[1] = np.array(
        [r, g, b, 1]
    )  # Label 1: bright visible color with full opacity

    # Create a DirectLabelColormap
    return DirectLabelColormap(color_dict=color_dict)
