import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # remove cuda for tests

import numpy as np
import torch

from napari_roxas_ai._models._cells_model import CellsSegmentationModel


def test_cells_segmentation_model():

    # Initialize model
    model = CellsSegmentationModel()
    print(f"Model running on device : {model.available_device}")

    # Create a synthetic input image
    test_input = np.random.randint(
        low=0, high=256, size=(512, 512, 3), dtype="uint8"
    )

    # Run forward pass
    with torch.no_grad():
        output = model.infer(test_input)

    # Validate output shape
    assert (
        output.shape == test_input.shape[:2]
    ), f"Unexpected output shape: {output.shape}"

    # Check for NaNs
    assert ~np.isnan(output).any(), "Found NaNs in the output"

    # Ensure values are binaries
    assert np.isin(output, (0, 255)).all(), "Output is not a binary label"

    print("CellsSegmentationModel test passed!")


# Run the test
test_cells_segmentation_model()
