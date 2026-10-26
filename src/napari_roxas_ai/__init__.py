"""napari-roxas-ai plugin package.

Public widget/contribution objects are imported lazily (PEP 562) so that
opening any single widget only pulls in that widget's own dependencies.

Previously every ``python_name`` in ``napari.yaml`` pointed at this package
root, so opening *any* widget executed this module and eagerly imported all
submodules -- including the heavy ML stack (torch, pytorch-lightning,
segmentation-models-pytorch) via ``._segmentation``. That made the first
widget of any kind take ~10 s. With lazy access the heavy stack is only
imported when a segmentation widget is actually opened.
"""

import importlib
import os
import warnings

# Suppress Albumentations update check
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# Suppress PyTorch TypedStorage deprecation warning (common when using torch.package)
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

# Suppress napari's GL_MAX_TEXTURE_SIZE warning for large scans: only the on-screen
# texture is downsampled, layer data stays at full resolution
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"data shape .* exceeds GL_MAX_TEXTURE_SIZE",
)

# The project directory "widget" only opens a folder dialog and returns None, which
# napari then fails to dock; nothing is meant to be docked, so the warning is noise
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"Processor .* failed to process result \(None, '0 - Define project directory",
)

# NumPy 2.0 Compatibility Patch
# Many older models/packages use np.Inf, np.NaN, etc. which were removed in NumPy 2.0.
try:
    import numpy as np
    if not hasattr(np, 'Inf'):
        np.Inf = np.inf
    if not hasattr(np, 'NaN'):
        np.NaN = np.nan
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'complex'):
        np.complex = complex
except ImportError:
    pass

__version__ = "0.2.1"

# Public attribute name -> submodule that defines it.
# Mirrors the ``python_name`` references in napari.yaml and the previous
# eager imports / ``__all__`` of this module.
_LAZY_IMPORTS = {
    "CrossDatingPlotterWidget": "._crossdating",
    "CellsLayerEditorWidget": "._edition",
    "RingsLayerEditorWidget": "._edition",
    "SamplesLoadingWidget": "._loading",
    "BatchSampleMeasurementsWidget": "._measurements",
    "SingleSampleMeasurementsWidget": "._measurements",
    "PreparationWidget": "._preparation",
    "open_project_directory_dialog": "._project_directory",
    "napari_get_reader": "._reader",
    "load_sample_data": "._sample_data",
    "SamplesSavingWidget": "._saving",
    "BatchSampleSegmentationWidget": "._segmentation",
    "SingleSampleSegmentationWidget": "._segmentation",
    "SettingsWidget": "._settings._settings_widget",
    "write_multiple_layers": "._writer",
    "write_single_layer": "._writer",
    "close_all_dock_widgets": ".shortcuts._shortcuts",
    "load_manual_editing_group": ".shortcuts._shortcuts",
    "load_batch_operations_group": ".shortcuts._shortcuts",
    "is_widget_already_present": ".shortcuts._shortcuts",
    "prevent_duplicate_dock_if_already_present": ".shortcuts._shortcuts",
}

__all__ = tuple(_LAZY_IMPORTS)


def __getattr__(name):
    """Lazily import public attributes on first access (PEP 562)."""
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    module = importlib.import_module(module_name, __name__)
    attr = getattr(module, name)
    # Cache on the package so subsequent lookups skip __getattr__ entirely.
    globals()[name] = attr
    return attr


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
