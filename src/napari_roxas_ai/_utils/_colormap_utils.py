"""
All Colormap Related Utils
"""

import colorsys
from collections import defaultdict

import numpy as np
from napari.utils.colormaps import DirectLabelColormap

from napari_roxas_ai._settings import SettingsManager

settings = SettingsManager()


def make_binary_labels_colormap(create_random_color=False):

    # Define the color mapping for labels
    color_dict = defaultdict(
        lambda: np.array([0, 0, 0, 0])
    )  # Default: transparent

    if create_random_color:
        # Generate a bright, highly saturated random color
        h = np.random.rand()  # Random hue (0 to 1)
        s = 0.9  # High saturation
        v = 1.0  # Maximum brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)  # Convert HSV to RGB

        color_dict[1] = np.array(
            [r, g, b, 1]
        )  # Label 1: bright visible color with full opacity

    else:
        color_dict[1] = settings.get("rasterization.cells_color")

    # Create a DirectLabelColormap
    return DirectLabelColormap(color_dict=color_dict)


def make_rings_colormap(unique_rings_raster_values):

    qualitative_colors = settings.get("rasterization.rings_color_sequence")
    uncomplete_ring_color = settings.get("rasterization.uncomplete_ring_color")
    uncomplete_ring_value = settings.get("rasterization.uncomplete_ring_value")

    ring_years = unique_rings_raster_values[
        unique_rings_raster_values != uncomplete_ring_value
    ]

    # Create a colormap using string color names
    colormap = defaultdict(
        lambda: [0, 0, 0, 0]
    )  # Default: transparent for unsupported values
    colormap[uncomplete_ring_value] = (
        uncomplete_ring_color  # red for disabled rings
    )

    # Map each ring year to a color from the qualitative_colors list (cycling if needed)
    for i, year in enumerate(ring_years):
        color_idx = i % len(qualitative_colors)
        colormap[year] = qualitative_colors[color_idx]

    return colormap
