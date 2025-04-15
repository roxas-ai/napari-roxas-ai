"""
Module for analyzing segmented wood cell images to measure cell wall thickness and related metrics.
Based on code by github user triyan-b https://github.com/triyan-b
Refactored and adapted to large images by github user tha-santacruz https://github.com/tha-santacruz
"""

from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rasterio import features

# Disable DecompressionBomb warnings for large images
Image.MAX_IMAGE_PIXELS = None


class CellAnalyzer:
    """Main class for analyzing cell structures in segmented images."""

    def __init__(self, config: Dict):
        """
        Initialize analyzer with configuration parameters.

        Args:
            config: Dictionary containing analysis parameters:
                - pixels_per_um: Conversion factor from pixels to micrometers
                - cluster_separation_threshold: Minimum distance between clusters (µm)
                - smoothing_kernel_size: Size of morphological operation kernel
                - integration_interval: Fraction of wall used for thickness measurement
                - tangential_angle : Sample angle (degrees, clockwise)
        """
        self.config = config
        self.cells: Dict = {}
        self.cells_array = None
        self.centroids_map = None
        self.dist_transform = None

        # Derived parameters
        self.cluster_separation_px = (
            config["cluster_separation_threshold"] * config["pixels_per_um"]
        )
        self.kernel = np.ones(
            (config["smoothing_kernel_size"], config["smoothing_kernel_size"])
        )
        self.integration_margin = (1 - config["integration_interval"]) / 2
        self.radial_angle = config["tangential_angle"] - 90

    def _load_image(self, image_path: Path) -> None:
        """Load and preprocess the input image."""
        with Image.open(image_path) as img:
            self.cells_array = np.array(img).astype("uint8")

    def _smooth_image(self) -> None:
        """Apply morphological smoothing"""
        self.cells_array = cv2.dilate(
            cv2.erode(self.cells_array, self.kernel), self.kernel
        )

    def _find_contours(self) -> None:
        """Find lumen and cell wall contours."""
        # Find lumen contours
        self.lumen_contours, _ = cv2.findContours(
            self.cells_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Find cell wall contours using distance transform
        _, comps = cv2.distanceTransformWithLabels(
            cv2.bitwise_not(self.cells_array),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
            cv2.DIST_LABEL_CCOMP,
        )

        # Extract cell wall contours
        self.cell_walls_contours = []
        for shape, _ in features.shapes(comps):
            coords = np.array(shape["coordinates"][0]).astype("int32")
            self.cell_walls_contours.append(np.expand_dims(coords, 1))

    def _analyze_lumina(self) -> None:
        """Calculate lumen metrics and initialize cell entries."""
        for i, contour in enumerate(self.lumen_contours):
            cell = {"id": i}
            M = cv2.moments(contour)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cell.update(
                    {
                        "centroid": (cy, cx),
                        "lumen_area": M["m00"]
                        / self.config["pixels_per_um"] ** 2,
                        "lumen_peri": cv2.arcLength(contour, True)
                        / self.config["pixels_per_um"],
                    }
                )
            else:
                cell.update(
                    {
                        "centroid": (np.nan, np.nan),
                        "lumen_area": np.nan,
                        "lumen_peri": np.nan,
                    }
                )

            # Ellipse fitting for orientation metrics
            if len(contour) >= 5:
                self._calculate_ellipse_metrics(contour, cell)
            else:
                cell.update(
                    {
                        "lumen_aoma_rad": np.nan,
                        "lumen_diam_rad": np.nan,
                        "lumen_diam_tang": np.nan,
                    }
                )

            self.cells[i] = cell

    def _calculate_ellipse_metrics(
        self, contour: np.ndarray, cell: Dict
    ) -> None:
        """Calculate ellipse-based metrics for lumen contours."""
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        w, h = axes

        # TODO : Figure out what's up with the /2, and remove if appropriate
        if w >= h:
            a, b = w / 2, h / 2
            aoma_rad = angle - self.radial_angle
        else:
            a, b = h / 2, w / 2
            aoma_rad = angle - self.radial_angle - 90

        aoma_tang = aoma_rad + 90

        # Calculate diameters using polar form of ellipse equation
        lumen_diam_rad = (
            2
            * a
            * b
            / np.linalg.norm(
                [
                    b * np.cos(np.deg2rad(aoma_rad)),
                    a * np.sin(np.deg2rad(aoma_rad)),
                ]
            )
        )
        lumen_diam_tang = (
            2
            * a
            * b
            / np.linalg.norm(
                [
                    b * np.cos(np.deg2rad(aoma_tang)),
                    a * np.sin(np.deg2rad(aoma_tang)),
                ]
            )
        )

        cell.update(
            {
                "lumen_aoma_rad": aoma_rad,
                "lumen_diam_rad": lumen_diam_rad
                / self.config["pixels_per_um"],
                "lumen_diam_tang": lumen_diam_tang
                / self.config["pixels_per_um"],
            }
        )

    def _analyze_cell_walls(self) -> None:
        """Calculate cell wall metrics."""
        # Create centroids map for cell identification
        self.centroids_map = np.full_like(self.cells_array, -1, dtype="int32")
        for cell_id, data in self.cells.items():
            if not np.isnan(data["centroid"]).any():
                self.centroids_map[data["centroid"]] = cell_id

        # Precompute distance transform
        self.dist_transform = cv2.distanceTransform(
            cv2.bitwise_not(self.cells_array).astype("uint8"),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )

        for contour in self.cell_walls_contours:
            self._process_cell_wall_contour(contour)

    def _process_cell_wall_contour(self, contour: np.ndarray) -> None:
        """Process individual cell wall contour."""
        # Find associated cell through centroid containment
        cell_id = self._find_contained_cell(contour)
        if cell_id is None:
            return

        # Calculate basic cell metrics
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.cells[cell_id].update(
                {
                    "cell_area": M["m00"] / self.config["pixels_per_um"] ** 2,
                    "cw_peri": cv2.arcLength(contour, True)
                    / self.config["pixels_per_um"],
                }
            )

        # Calculate wall thickness measurements
        self._measure_wall_thickness(contour, cell_id)

    def _find_contained_cell(self, contour: np.ndarray) -> int:
        """Find cell ID contained within the wall contour."""
        x, y, w, h = cv2.boundingRect(contour)

        # Make sure it does not got out of the image (a 1 pixel offset is possible)
        h = h - 1 if (y + h) > self.cells_array.shape[0] else h
        w = w - 1 if (x + w) > self.cells_array.shape[1] else w

        centroids_crop = self.centroids_map[y : y + h, x : x + w]
        candidates = np.unique(centroids_crop[centroids_crop >= 0])
        for candidate in candidates:
            centroid = self.cells[candidate]["centroid"][::-1]  # (x,y) format
            if cv2.pointPolygonTest(contour, centroid, False) >= 0:
                return candidate
        return None

    def _measure_wall_thickness(
        self, contour: np.ndarray, cell_id: int
    ) -> None:
        """Measure wall thickness in different directions."""
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Make sure it does not got out of the image (a 1 pixel offset is possible)
        h = h - 1 if (y + h) > self.cells_array.shape[0] else h
        w = w - 1 if (x + w) > self.cells_array.shape[1] else w

        # Create labeling canvas with edge markers to create distinct 0 zones for labelling
        canvas = np.ones((h, w), dtype="uint8")
        canvas[0, w // 2] = 0  # Top center (pith)
        canvas[h // 2, 0] = 0  # Left center (left)
        canvas[-1, w // 2] = 0  # Bottom center (bark)
        canvas[h // 2, -1] = 0  # Right center (right)

        # Compute distance transform labels
        _, labels = cv2.distanceTransformWithLabels(
            canvas.astype("uint8"),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
            cv2.DIST_LABEL_CCOMP,
        )

        # Original label mapping
        label_map = {1: "pith", 2: "left", 3: "right", 4: "bark"}

        # Get precise cell wall contour pixels by drawing their filled contour and then finding contours again
        contour_points = contour.squeeze() - [x, y]
        mask = np.zeros_like(labels).astype("uint8")
        drawing = cv2.drawContours(mask, [contour_points], 0, 1, -1)
        refined_contours, _ = cv2.findContours(
            drawing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Update the contours pixels coordinates and get corresponding labels. NB: pixel coordinates are given in the positive order (counter-clockwise)
        contour_points = refined_contours[0].squeeze()
        contour_labels = labels[contour_points[:, 1], contour_points[:, 0]]

        # Just write nan if we can't find all walls or we have too many (may be error codes in the future)
        if len(np.unique(contour_labels)) != 4:
            for name in label_map.values():
                self.cells[cell_id][f"CWT_{name}"] = np.nan
            return

        # Check when the labels vector changes value (aka we change wall)
        value_changes = np.where(np.diff(contour_labels) != 0)[0]

        # If it changes more than 4 times, it means the shape is strange and concave and we need to find another way to process it. We write an nan instead (may be error codes in the future)
        if len(value_changes) > 4:
            for name in label_map.values():
                self.cells[cell_id][f"CWT_{name}"] = np.nan
            return

        # If one of the wall is split between the contour_pixels_ids' head and tail (aka if the value changes 4 times), we rearrange the tail by placing it before the head
        n_roll = contour_labels.shape - value_changes[-1] - 1
        contour_labels = np.roll(contour_labels, n_roll)
        contour_points = np.roll(
            contour_points,
            contour_labels.shape - value_changes[-1] - 1,
            axis=0,
        )
        contour = np.roll(contour, n_roll, axis=0)

        # Now we compute the distances
        dist_crop = self.dist_transform[y : y + h, x : x + w]

        # And we compute the thickness
        for label in label_map:
            # Get wall pixels where the label cooresponds
            wall_pixel_coords = contour_points[
                np.where(contour_labels == label)[0], :
            ]

            # Crop to keep the middle 75%
            lower_bound = np.ceil(
                self.integration_margin * wall_pixel_coords.shape[0]
            ).astype("int32")
            upper_bound = np.ceil(
                (1 - self.integration_margin) * wall_pixel_coords.shape[0]
            ).astype("int32")

            # Write the mean thickness in the cell dict (might be median in the future)
            avg_dist = dist_crop[
                wall_pixel_coords[lower_bound:upper_bound, 1],
                wall_pixel_coords[lower_bound:upper_bound, 0],
            ].mean()
            self.cells[cell_id].update(
                {
                    f"CWT_{label_map[label]}": avg_dist
                    * self.config["pixels_per_um"],
                }
            )

    def _cluster_cells(self) -> None:
        """Cluster cells based on proximity."""
        # Threshold distance transform for clustering
        _, dist_thresh = cv2.threshold(
            self.dist_transform / self.config["pixels_per_um"],
            self.config["cluster_separation_threshold"],
            255,
            cv2.THRESH_BINARY,
        )

        # Find connected components
        _, clusters = cv2.connectedComponents(
            cv2.bitwise_not(dist_thresh.astype("uint8")), connectivity=8
        )

        # Assign clusters to cells
        for cell_id, data in self.cells.items():
            if not np.isnan(data["centroid"]).any():
                self.cells[cell_id]["cluster"] = clusters[data["centroid"]]
            else:
                self.cells[cell_id]["cluster"] = np.nan

    def _get_results_df(self) -> pd.DataFrame:
        """Return results as pandas DataFrame."""
        self.df = pd.DataFrame(self.cells).T

    def _draw_contours(self) -> None:
        """Draw lumina and cell wall contours on an image"""
        self.lumen_contours_image = cv2.drawContours(
            np.zeros_like(self.cells_array), self.lumen_contours, -1, (255), 1
        )
        self.cell_walls_contours_image = cv2.drawContours(
            np.zeros_like(self.cells_array),
            self.cell_walls_contours,
            -1,
            (255),
            1,
        )


def measure_cells(
    config: Dict, input_path: Path, output_path: Path = None
) -> pd.DataFrame:
    """
    Main analysis workflow.

    Args:
        config: Analysis configuration parameters
        input_path: Path to input segmentation image
        output_path: Optional path to save results CSV

    Returns:
        DataFrame containing all measurements
    """
    analyzer = CellAnalyzer(config)
    analyzer._load_image(input_path)
    analyzer._smooth_image()
    analyzer._find_contours()
    analyzer._analyze_lumina()
    analyzer._analyze_cell_walls()
    analyzer._cluster_cells()
    analyzer._draw_contours()

    analyzer._get_results_df()

    if output_path:
        analyzer.df.to_csv(output_path, index_label="cell_id")

    return analyzer.df


if __name__ == "__main__":
    # Example configuration
    CONFIG = {
        "pixels_per_um": 2.2675,
        "cluster_separation_threshold": 3,  # µm
        "smoothing_kernel_size": 5,
        "integration_interval": 0.75,
        "tangential_angle": 0,  # Assuming vertical orientation
    }

    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze wood cell segmentation images"
    )
    parser.add_argument(
        "input", type=Path, help="Path to input segmentation image"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output CSV path")
    args = parser.parse_args()

    measure_cells(CONFIG, args.input, args.output)
