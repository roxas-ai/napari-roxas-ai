import numpy as np
from scipy import ndimage as ndi

def remove_border_touching_components(binary: np.ndarray) -> np.ndarray:
    """
    Remove connected components that touch the image border.
    Expects a 2D binary array (0/1 or 0/255).
    Returns a uint8 array with values 0/1.
    """
    if binary.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={binary.shape}")

    mask = (binary > 0)

    if not mask.any():
        return mask.astype("uint8")

    structure = np.ones((3, 3), dtype=bool)
    labeled, num = ndi.label(mask, structure=structure)

    if num == 0:
        return mask.astype("uint8")

    border_labels = np.unique(
        np.concatenate([labeled[0, :], labeled[-1, :], labeled[:, 0], labeled[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]

    if border_labels.size == 0:
        return mask.astype("uint8")

    mask[np.isin(labeled, border_labels)] = False
    return mask.astype("uint8")
