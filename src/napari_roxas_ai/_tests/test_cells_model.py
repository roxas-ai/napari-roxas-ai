import sys

import numpy as np
import pytest
import torch

from napari_roxas_ai._segmentation._cells_model import CellsSegmentationModel


@pytest.mark.skipif(sys.platform == "darwin", reason="Skip on macOS")
def test_cells_segmentation_model():

    # Initialize model
    model = CellsSegmentationModel()
    print(f"Model running on device : {model.available_device}")

    # Create a synthetic input image
    # note: force float32 bc mps backend does not like float64
    test_input = np.random.randn(1, 3, 1024, 1024).astype(np.float32)

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


class _AllCellsNet(torch.nn.Module):
    """Stub network predicting class 1 everywhere, to test tiling cheaply."""

    def forward(self, x):
        out = torch.zeros((x.shape[0], 2, x.shape[2], x.shape[3]))
        out[:, 1] = 1
        return out


@pytest.mark.skipif(sys.platform == "darwin", reason="Skip on macOS")
@pytest.mark.parametrize(
    "shape",
    [
        (1376, 1376),  # padded size is a multiple of the stride
        (800, 1600),  # image size is a multiple of the stride
        (1477, 900),
        (300, 2000),
    ],
)
def test_cells_segmentation_model_tiling_covers_image(shape):

    model = CellsSegmentationModel()
    model.net = _AllCellsNet()
    model.available_device = "cpu"
    model.use_autocast = False

    img = np.random.randint(0, 256, (*shape, 3), dtype=np.uint8)
    labels = model.infer(img)

    assert labels.shape == shape
    assert (labels == 255).all(), "Some pixels were not covered by a patch"
