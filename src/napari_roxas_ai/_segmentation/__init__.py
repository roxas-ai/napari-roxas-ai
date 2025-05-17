from pathlib import Path

from napari_roxas_ai._assets_files import check_assets_and_download

from ._batch_sample_segmentation import BatchSampleSegmentationWidget
from ._single_sample_segmentation import SingleSampleSegmentationWidget

__all__ = [
    "SingleSampleSegmentationWidget",
    "BatchSampleSegmentationWidget",
]


BASE_DIR = Path(__file__).parent.absolute()
check_assets_and_download(
    str(BASE_DIR / "_models" / "_cells"), "cells_models.zip"
)
check_assets_and_download(
    str(BASE_DIR / "_models" / "_rings"), "rings_models.zip"
)
