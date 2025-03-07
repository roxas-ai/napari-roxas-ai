"""
This module provides sample data to experiment with the plugin
"""

import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_conifer_sample_data():
    with Image.open(
        os.path.join(BASE_DIR, "_sample_data/conifer/cells.png")
    ) as img:
        cells = np.array(img)

    with Image.open(
        os.path.join(BASE_DIR, "_sample_data/conifer/rings.tiff")
    ) as img:
        rings = np.array(img)

    with Image.open(
        os.path.join(BASE_DIR, "_sample_data/conifer/thin_section.jpg")
    ) as img:
        thin_section = np.array(img)

    return [
        (thin_section, {"name": "Thin section"}, "image"),
        (rings, {"name": "Tree rings"}, "labels"),
        (cells, {"name": "Tree cells"}, "labels"),
    ]
