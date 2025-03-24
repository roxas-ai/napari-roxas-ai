"""
This module provides sample data to experiment with the plugin
"""

import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_conifer_sample_data():
    with Image.open(os.path.join(BASE_DIR, "conifer/cells.png")) as img:
        cells = np.divide(np.array(img), 255).astype("uint8")

    with Image.open(os.path.join(BASE_DIR, "conifer/rings.tiff")) as img:
        rings = np.array(img)

    with Image.open(os.path.join(BASE_DIR, "conifer/thin_section.jpg")) as img:
        thin_section = np.array(img)

    return [
        (thin_section, {"name": "Thin section"}, "image"),
        (rings, {"name": "Tree rings"}, "labels"),
        (cells, {"name": "Tree cells"}, "labels"),
    ]
