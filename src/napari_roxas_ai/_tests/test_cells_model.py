import numpy as np
import torch

from napari_roxas_ai._models._cells_model import CellsSegmentationModel


def test_cells_segmentation_model():

    # Initialize model
    model = CellsSegmentationModel()
    print(f"Model running on device : {model.available_device}")

    # Create a synthetic input image
    test_input = np.random.randn(1, 3, 1024, 1024)

    # Run forward pass (full image inference is too heavy for github actions environment)
    with torch.no_grad():
        output = (
            model(torch.tensor(test_input).to(model.available_device).float())
            .squeeze(0)
            .cpu()
            .numpy()
        )
    # Validate output shape
    assert (
        output.shape[-2:] == test_input.shape[-2:]
    ), f"Unexpected output shape: {output.shape}"

    # Check for NaNs
    assert ~np.isnan(output).any(), "Found NaNs in the output"

    # Ensure values are binaries
    assert np.isin(output, (0, 1)).all(), "Output is not a binary label"

    print("CellsSegmentationModel test passed!")
