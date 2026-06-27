from ._batch_sample_segmentation import BatchSampleSegmentationWidget
from ._single_sample_segmentation import SingleSampleSegmentationWidget

__all__ = [
    "SingleSampleSegmentationWidget",
    "BatchSampleSegmentationWidget",
]

# NOTE: model weights are downloaded lazily by each segmentation widget's
# __init__ (see check_assets_and_download calls there) instead of at import
# time, so importing this package never triggers network/filesystem I/O.
