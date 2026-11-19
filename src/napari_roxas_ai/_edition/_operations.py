import cv2
import numpy as np

from napari_roxas_ai._jobs import Progress


def vectorize_cells(layer_data: np.ndarray, tolerance: float):
    """
    Extract polygons from binary cells mask.
    """
    yield Progress(0, 0, "Vectorizing cells…")
    contours, _ = cv2.findContours(
        layer_data.astype("uint8"),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cells_polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        if tolerance > 0:
            poly = cv2.approxPolyDP(contour, epsilon=tolerance, closed=True)
        else:
            poly = contour
        if poly.shape[0] > 2:
            cells_polygons.append(poly.squeeze(axis=1)[:, ::-1])
    return cells_polygons
