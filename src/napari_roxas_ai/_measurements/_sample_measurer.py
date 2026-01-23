"""
Module for analyzing segmented wood cells and rings images to measure cell wall thickness, rings width and related metrics.
Based on code by github user triyan-b https://github.com/triyan-b
Refactored and adapted to large images by github user tha-santacruz https://github.com/tha-santacruz
"""

import ast
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from rasterio import features


class SampleAnalyzer:
    """Main class for analyzing cell structures in segmented images."""

    def __init__(
        self,
        config: Dict,
        cells_array: np.ndarray,
        rings_table: pd.DataFrame,
        cells_table: Optional[pd.DataFrame] = None,
    ) -> None:
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
        self.cells_array = cells_array
        self.rings_table = rings_table
        self.cells_table = (
            pd.DataFrame() if cells_table is None else cells_table
        )
        self.cells: Dict = {}
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

    def _smooth_cells_array(self) -> None:
        """Apply morphological smoothing"""
        self.cells_array = cv2.dilate(
            cv2.erode(self.cells_array, self.kernel), self.kernel
        )

    def _find_cells_contours(self) -> None:
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

    def _compute_cells_lumina(self) -> None:
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
                        "XPIX": cx,
                        "YPIX": cy,
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
                        "XPIX": np.nan,
                        "YPIX": np.nan,
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
                        "ASP": np.nan,
                        "MAJAX": np.nan,
                        "KH": np.nan,
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
            angle_major = angle
        else:
            a, b = h / 2, w / 2
            aoma_rad = angle - self.radial_angle - 90
            angle_major = angle + 90

        # Normalize major axis angle to [0,180)
        angle_major = angle_major % 180

        # Compute MAJAX: deviation from the image vertical (90°)
        majax = abs(angle_major - 90)
        if majax > 90:
            majax = 180 - majax  # fold into [0,90]

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

        asp = a / b if b != 0 else np.nan
        LA = cell.get("lumen_area", np.nan)
        KH = self.compute_kh(
            lumen_area_um2=LA,
            major_radius_px=a,
            minor_radius_px=b,
            pixels_per_um=self.config["pixels_per_um"]
        )

        DH = self.compute_dh(
            lumen_area_um2=LA,
            aspect_ratio=asp
        )


        cell.update(
            {
                "lumen_aoma_rad": aoma_rad,
                "lumen_diam_rad": lumen_diam_rad
                / self.config["pixels_per_um"],
                "lumen_diam_tang": lumen_diam_tang
                / self.config["pixels_per_um"],
                "ASP": asp,
                "MAJAX": majax,
                "KH": KH,
                "DH": DH,
            }
        )

    def compute_kh(self, lumen_area_um2: float, major_radius_px: float, minor_radius_px: float, pixels_per_um: float) -> float:
        """
        Compute theoretical hydraulic conductance KH for an elliptical lumen.

        Parameters
        ----------
        lumen_area_um2 : Lumen area in µm².
        major_radius_px : Major semi-axis (a) in pixels.
        minor_radius_px : Minor semi-axis (b) in pixels.
        pixels_per_um : Conversion factor: pixels per micrometer.
        """
        if (
                lumen_area_um2 is None or np.isnan(lumen_area_um2) or
                major_radius_px is None or minor_radius_px is None or
                np.isnan(major_radius_px) or np.isnan(minor_radius_px) or
                major_radius_px <= 0 or minor_radius_px <= 0
        ):
            return np.nan

        # Convert radii to µm
        a_um = major_radius_px / pixels_per_um
        b_um = minor_radius_px / pixels_per_um

        a_um = a_um / 1000000
        b_um = b_um / 1000000

        if a_um <= 0 or b_um <= 0:
            return np.nan

        # calculate Eccentricity
        diff = a_um * a_um - b_um * b_um
        e = np.sqrt(max(diff, 0)) / a_um

        # calculate Lumen circumference C
        C = np.pi * (3 * (a_um + b_um) - np.sqrt((3 * a_um + b_um) * (a_um + 3 * b_um)))
        if C <= 0:
            return np.nan

        # calculate Mean hydraulic radius m
        m = (np.pi * a_um * b_um) / C

        # calculate Form factor k
        term = 1 - e ** 4
        if term < 0:
            return np.nan

        k = 4.0 / (1.0 + np.sqrt(term))
        if k <= 0:
            return np.nan

        # Convert LA from µm² → m²
        LA_m2 = lumen_area_um2 * 1e-12

        # calculate KH final formula
        nu = 1.002e-9  # viscosity of water (MPa·s)
        KH = (LA_m2 * (m * m)) / (nu * k)

        return KH

    def _compute_cell_walls(self) -> None:
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
        self._measure_cell_wall_thickness(contour, cell_id)

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

    def _measure_cell_wall_thickness(
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
                    / self.config["pixels_per_um"],
                }
            )
        self._compute_cwttan(cell_id)
        self._compute_cwtrad(cell_id)
        self._compute_cwtall(cell_id)
        self._compute_rtsr(cell_id)
        self._compute_ctsr(cell_id)
        self._compute_tb2(cell_id)
        self._compute_cwa(cell_id)
        self._compute_rwd(cell_id)

    def _compute_cwttan(self, cell_id: int) -> None:
        """Compute CWTTAN = tangential wall thickness = (CWT_pith + CWT_bark) / 2"""

        cwt_pith = self.cells[cell_id].get("CWT_pith", np.nan)
        cwt_bark = self.cells[cell_id].get("CWT_bark", np.nan)

        if (
                not np.isnan(cwt_pith) and cwt_pith > 0 and
                not np.isnan(cwt_bark) and cwt_bark > 0
        ):
            self.cells[cell_id]["CWTTAN"] = (cwt_pith + cwt_bark) / 2
        else:
            self.cells[cell_id]["CWTTAN"] = np.nan


    def _compute_cwtrad(self, cell_id: int) -> None:
        """Compute CWTRAD = Thickness of radial cell walls ([CWT_left+CWT_right]/2)"""

        cwt_left = self.cells[cell_id].get("CWT_left", np.nan)
        cwt_right = self.cells[cell_id].get("CWT_right", np.nan)

        if (
                not np.isnan(cwt_left) and cwt_left > 0 and
                not np.isnan(cwt_right) and cwt_right > 0
        ):
            self.cells[cell_id]["CWTRAD"] = (cwt_left + cwt_right) / 2
        else:
            self.cells[cell_id]["CWTRAD"] = np.nan


    def _compute_cwtall(self, cell_id: int) -> None:
        """Compute CWTALL = Thickness of all cell walls ([CWTRAD+CWTTAN]/2)"""

        CWTRAD = self.cells[cell_id].get("CWTRAD", np.nan)
        CWTTAN = self.cells[cell_id].get("CWTTAN", np.nan)

        if not np.isnan(CWTRAD)  and not np.isnan(CWTTAN):
            self.cells[cell_id]["CWTALL"] = (CWTRAD + CWTTAN) / 2
        else:
            self.cells[cell_id]["CWTALL"] = np.nan

    def _compute_rtsr(self, cell_id: int) -> None:
        """Compute RTSR = Radial Thickness-to-span ratio, Mork's index: ratio between 4x single cell wall
           thickness (CWTtan) and tracheid diameter (lumen_diam_rad) in radial direction (pith to bark)
        """

        cwttan = self.cells[cell_id].get("CWTTAN", np.nan)
        lumen_diam_rad = self.cells[cell_id].get("lumen_diam_rad", np.nan)

        if not np.isnan(lumen_diam_rad) and lumen_diam_rad > 0:
            self.cells[cell_id]["RTSR"] = (4 * cwttan) / lumen_diam_rad
        else:
            self.cells[cell_id]["RTSR"] = np.nan

    def _compute_ctsr(self, cell_id: int) -> None:
        """Compute CTSR = Circular Thickness-to-span ratio: ratio between 4x single cell wall
           thickness (CWTall) and tracheid diameter (assuming a circle area-equivalent to the lumen area)
        """

        cwtall = self.cells[cell_id].get("CWTALL", np.nan)
        la = self.cells[cell_id].get("lumen_area", np.nan)

        if (
                cwtall is None or np.isnan(cwtall) or cwtall <= 0 or
                la is None or np.isnan(la) or la <= 0
        ):
            self.cells[cell_id]["CTSR"] = np.nan
            return

        # Circle diameter from area-equivalent circle
        circle_diameter = 2.0 * np.sqrt(la / np.pi)

        self.cells[cell_id]["CTSR"] = (4.0 * cwtall) / circle_diameter

    def compute_dh(self, lumen_area_um2: float, aspect_ratio: float) -> float:
        """
        Compute hydraulic diameter Dh (µm) following Lewis & Boose (1995)
        for an elliptical conduit.

        Dh = sqrt( (2 a² b²) / (a² + b²) )

        where:
            a, b = semi-axes of an ellipse derived from
                   lumen area and aspect ratio (a / b).
        """

        if (
                lumen_area_um2 is None or np.isnan(lumen_area_um2) or lumen_area_um2 <= 0 or
                aspect_ratio is None or np.isnan(aspect_ratio) or aspect_ratio <= 0
        ):
            return np.nan

        a = 2.0 * np.sqrt(aspect_ratio * lumen_area_um2 / np.pi)
        b = a / aspect_ratio

        a2 = a * a
        b2 = b * b

        denom = a2 + b2
        if denom <= 0:
            return np.nan

        Dh = np.sqrt((2.0 * a2 * b2) / denom)
        return Dh

    def _compute_tb2(self, cell_id: int) -> None:
        """
        Compute TB2 = Cell wall reinforcement index (t/b)^2
        following Hacke et al. (2001).

        t = double cell wall thickness
            - radial: 2 * CWTRAD
            - tangential: 2 * CWTTAN

        b = lumen diameter in the same direction
            - radial:      DRAD  (lumen_diam_rad)
            - tangential: DTAN  (lumen_diam_tang)

        TB2 is the smaller of the radial or tangential value.
        """

        cwtrad = self.cells[cell_id].get("CWTRAD", np.nan)
        cwttan = self.cells[cell_id].get("CWTTAN", np.nan)
        drad = self.cells[cell_id].get("lumen_diam_rad", np.nan)
        dtan = self.cells[cell_id].get("lumen_diam_tang", np.nan)

        values = []

        # Radial TB2
        if (
                not np.isnan(cwtrad) and cwtrad > 0 and
                not np.isnan(drad) and drad > 0
        ):
            t_rad = 2.0 * cwtrad
            values.append((t_rad / drad) ** 2)

        # Tangential TB2
        if (
                not np.isnan(cwttan) and cwttan > 0 and
                not np.isnan(dtan) and dtan > 0
        ):
            t_tan = 2.0 * cwttan
            values.append((t_tan / dtan) ** 2)

        self.cells[cell_id]["TB2"] = min(values) if values else np.nan

    def _compute_cwa(self, cell_id: int) -> None:
        """Compute Cell wall area = cell_area - lumen_area
        """

        cell_area = self.cells[cell_id].get("cell_area", np.nan)
        lumen_area = self.cells[cell_id].get("lumen_area", np.nan)

        if (
                not np.isnan(cell_area) and cell_area > 0 and
                not np.isnan(lumen_area) and lumen_area > 0 and
                cell_area > lumen_area
        ):
            CWA = cell_area - lumen_area
        else:
            CWA = np.nan


        self.cells[cell_id]["CWA"] = CWA

    def _compute_rwd(self, cell_id: int) -> None:
        """Compute RWD = Relative anatomical cell density = CWA / (CWA + LA)"""

        cwa = self.cells[cell_id].get("CWA", np.nan)
        la = self.cells[cell_id].get("lumen_area", np.nan)

        if (
                not np.isnan(cwa) and cwa > 0 and
                not np.isnan(la) and la > 0 and
                (cwa + la) > 0
        ):
            self.cells[cell_id]["RWD"] = cwa / (cwa + la)
        else:
            self.cells[cell_id]["RWD"] = np.nan

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

    def _compute_cluster_sizes(self):
        """Compute NBRNO (cluster size) and NBRID (cluster ID) """

        if "cluster" not in self.cells_table.columns:
            return

        clusters = self.cells_table["cluster"]

        # only add NBRNO and NBRID when sample_type is not conifer
        if self.config["sample_type"] != "conifer":
            # Count cells per cluster ID
            cluster_sizes = (
                clusters
                .value_counts(dropna=False)  # count all cluster IDs
                .rename("NBRNO")
            )
            # Map cluster size to each cell
            self.cells_table["NBRNO"] = clusters.map(cluster_sizes)

            cluster_members = (
                self.cells_table
                .groupby("cluster")
                .apply(lambda df: df.index.tolist())
            )

            # Map each cell's cluster to its member list
            self.cells_table["NBRID"] = clusters.map(cluster_members)

            # 3) Solitary cells → NBRID = NA
            solitary_mask = self.cells_table["NBRNO"] == 1
            self.cells_table.loc[solitary_mask, "NBRID"] = pd.NA

        else:
            # Conifer: always set NA
            self.cells_table["NBRNO"] = np.nan
            self.cells_table["NBRID"] = np.nan


    def _get_cells_table(self) -> pd.DataFrame:
        """Return results as pandas DataFrame."""
        self.cells_table = pd.DataFrame(self.cells).T.set_index("id")

    # TODO review needed!
    def _apply_cwt_filters(self) -> None:
        """
        Automatic filtering of cell wall thickness (CWT) measurements.

        Implements the exact workflow from the provided screenshot:

        Pre-selection of candidates
          1) Compute Median, Q1 and Q3 for:
             i) tangential: CWT_pith & CWT_bark combined
             ii) radial:    CWT_left & CWT_right combined
          2) Any CWT measurement within [Q1..Q3] is confirmed.
             Only measurements outside [Q1..Q3] are "candidates" and are subjected
             to the following outlier/context filtering.

        Hard limit filtering (applied only to candidates)
          3) IQR = Q3 - Q1 separately for tangential and radial combined sets.
          4) Lower hard limit:
               LL = Q1 - ll_scaling * IQR
               LL = max(LL, 1/pixels_per_um)   (sub-pixel values are implausible)
          5) Upper hard limit:
               UL = Q3 + ul_scaling * IQR
          6) Candidate measurements outside [LL..UL] are set to NA.

        Context filtering (applied only to candidates; keep order!)
          7) Opposite sides of lumen:
               a) CWT_bark  > opp_scaling * CWT_pith  -> remove CWT_bark
               b) CWT_pith  > opp_scaling * CWT_bark  -> remove CWT_pith
               c) CWT_left  > opp_scaling * CWT_right -> remove CWT_left
               d) CWT_right > opp_scaling * CWT_left  -> remove CWT_right
          8) Adjacent sides of lumen (keep this order!):
               a) CWT_bark  > adj_scaling * ave(CWT_left,  CWT_right) -> remove CWT_bark
               b) CWT_pith  > adj_scaling * ave(CWT_left,  CWT_right) -> remove CWT_pith
               c) CWT_left  > adj_scaling * ave(CWT_bark,  CWT_pith)  -> remove CWT_left
               d) CWT_right > adj_scaling * ave(CWT_bark,  CWT_pith)  -> remove CWT_right

        Notes
        -----
        - This function modifies self.cells_table in-place.
        - It also recomputes dependent metrics (CWTTAN, CWTRAD, CWTALL, RTSR, CTSR, TB2)
          to stay consistent with filtered base wall values.
        """

        if self.cells_table is None or self.cells_table.empty:
            return

        required = ["CWT_pith", "CWT_bark", "CWT_left", "CWT_right"]
        for col in required:
            if col not in self.cells_table.columns:
                return

        # --- Settings (colored parameters in the screenshot) ---
        ll_scaling = float(self.config.get("ll_scaling", 1.5))
        ul_scaling = float(self.config.get("ul_scaling", 3.0))
        opp_scaling = float(self.config.get("opp_scaling", 1.5))
        adj_scaling = float(self.config.get("adj_scaling", 2.5))

        px_per_um = float(self.config["pixels_per_um"])
        min_plausible = 1.0 / px_per_um  # 1 pixel in µm (sub-pixel range is implausible)

        df = self.cells_table

        # --- Pull columns as numeric ---
        cwt_pi = pd.to_numeric(df["CWT_pith"], errors="coerce")
        cwt_ba = pd.to_numeric(df["CWT_bark"], errors="coerce")
        cwt_le = pd.to_numeric(df["CWT_left"], errors="coerce")
        cwt_ri = pd.to_numeric(df["CWT_right"], errors="coerce")

        # --- Step 1: combined quantiles (tangential + radial) ---
        tan_all = pd.concat([cwt_pi, cwt_ba], ignore_index=True).dropna()
        rad_all = pd.concat([cwt_le, cwt_ri], ignore_index=True).dropna()

        if tan_all.empty or rad_all.empty:
            return

        # Medians computed for completeness (per screenshot), not used downstream
        _median_tan = float(tan_all.median())
        _median_rad = float(rad_all.median())

        q1_tan = float(tan_all.quantile(0.25))
        q3_tan = float(tan_all.quantile(0.75))
        q1_rad = float(rad_all.quantile(0.25))
        q3_rad = float(rad_all.quantile(0.75))

        # --- Step 2: confirmed (inside IQR) vs candidate (outside IQR) ---
        def _inside_iqr(x: pd.Series, q1: float, q3: float) -> pd.Series:
            return x.notna() & (x >= q1) & (x <= q3)

        confirmed_pi = _inside_iqr(cwt_pi, q1_tan, q3_tan)
        confirmed_ba = _inside_iqr(cwt_ba, q1_tan, q3_tan)
        confirmed_le = _inside_iqr(cwt_le, q1_rad, q3_rad)
        confirmed_ri = _inside_iqr(cwt_ri, q1_rad, q3_rad)

        cand_pi = cwt_pi.notna() & ~confirmed_pi
        cand_ba = cwt_ba.notna() & ~confirmed_ba
        cand_le = cwt_le.notna() & ~confirmed_le
        cand_ri = cwt_ri.notna() & ~confirmed_ri

        # --- Steps 3-6: hard limits (apply only to candidates) ---
        iqr_tan = q3_tan - q1_tan
        iqr_rad = q3_rad - q1_rad

        # Guard against pathological cases
        if not np.isfinite(iqr_tan) or iqr_tan < 0:
            iqr_tan = 0.0
        if not np.isfinite(iqr_rad) or iqr_rad < 0:
            iqr_rad = 0.0

        ll_tan = max(q1_tan - ll_scaling * iqr_tan, min_plausible)
        ul_tan = q3_tan + ul_scaling * iqr_tan

        ll_rad = max(q1_rad - ll_scaling * iqr_rad, min_plausible)
        ul_rad = q3_rad + ul_scaling * iqr_rad

        def _apply_hard_limits(x: pd.Series, cand: pd.Series, ll: float, ul: float) -> pd.Series:
            out = x.copy()
            mask = cand & (x.notna()) & ((x < ll) | (x > ul))
            out.loc[mask] = np.nan
            return out

        cwt_pi_f = _apply_hard_limits(cwt_pi, cand_pi, ll_tan, ul_tan)
        cwt_ba_f = _apply_hard_limits(cwt_ba, cand_ba, ll_tan, ul_tan)
        cwt_le_f = _apply_hard_limits(cwt_le, cand_le, ll_rad, ul_rad)
        cwt_ri_f = _apply_hard_limits(cwt_ri, cand_ri, ll_rad, ul_rad)

        # --- Steps 7-8: context filtering (apply only to candidates; keep order) ---
        # Use "current" values (after hard limits) for comparisons
        pi = cwt_pi_f
        ba = cwt_ba_f
        le = cwt_le_f
        ri = cwt_ri_f

        # Step 7: opposite sides
        # a) CWTba > 1.5 * CWTpi -> remove CWTba
        mask = cand_ba & ba.notna() & pi.notna() & (ba > (opp_scaling * pi))
        ba.loc[mask] = np.nan

        # b) CWTpi > 1.5 * CWTba -> remove CWTpi
        mask = cand_pi & pi.notna() & ba.notna() & (pi > (opp_scaling * ba))
        pi.loc[mask] = np.nan

        # c) CWTle > 1.5 * CWTri -> remove CWTle
        mask = cand_le & le.notna() & ri.notna() & (le > (opp_scaling * ri))
        le.loc[mask] = np.nan

        # d) CWTri > 1.5 * CWTle -> remove CWTri
        mask = cand_ri & ri.notna() & le.notna() & (ri > (opp_scaling * le))
        ri.loc[mask] = np.nan

        # Step 8: adjacent sides (keep order!)
        ave_lr = (le + ri) / 2.0
        ave_pb = (ba + pi) / 2.0

        # a) CWTba > 2.5 * ave(CWTle, CWTri) -> remove CWTba
        mask = cand_ba & ba.notna() & ave_lr.notna() & (ba > (adj_scaling * ave_lr))
        ba.loc[mask] = np.nan

        # b) CWTpi > 2.5 * ave(CWTle, CWTri) -> remove CWTpi
        mask = cand_pi & pi.notna() & ave_lr.notna() & (pi > (adj_scaling * ave_lr))
        pi.loc[mask] = np.nan

        # c) CWTle > 2.5 * ave(CWTba, CWTpi) -> remove CWTle
        mask = cand_le & le.notna() & ave_pb.notna() & (le > (adj_scaling * ave_pb))
        le.loc[mask] = np.nan

        # d) CWTri > 2.5 * ave(CWTba, CWTpi) -> remove CWTri
        mask = cand_ri & ri.notna() & ave_pb.notna() & (ri > (adj_scaling * ave_pb))
        ri.loc[mask] = np.nan

        # --- Write filtered base values back ---
        df["CWT_pith"] = pi
        df["CWT_bark"] = ba
        df["CWT_left"] = le
        df["CWT_right"] = ri

        # --- Recompute dependent cell-level metrics to stay consistent ---
        # CWTTAN = (pith + bark)/2 if both positive
        cwttan = pd.Series(np.nan, index=df.index, dtype="float64")
        mask = pi.notna() & ba.notna() & (pi > 0) & (ba > 0)
        cwttan.loc[mask] = (pi.loc[mask] + ba.loc[mask]) / 2.0
        df["CWTTAN"] = cwttan

        # CWTRAD = (left + right)/2 if both positive
        cwtrad = pd.Series(np.nan, index=df.index, dtype="float64")
        mask = le.notna() & ri.notna() & (le > 0) & (ri > 0)
        cwtrad.loc[mask] = (le.loc[mask] + ri.loc[mask]) / 2.0
        df["CWTRAD"] = cwtrad

        # CWTALL = (CWTRAD + CWTTAN)/2 if both present
        cwtall = pd.Series(np.nan, index=df.index, dtype="float64")
        mask = cwtrad.notna() & cwttan.notna()
        cwtall.loc[mask] = (cwtrad.loc[mask] + cwttan.loc[mask]) / 2.0
        df["CWTALL"] = cwtall

        # RTSR = (4 * CWTTAN) / lumen_diam_rad
        if "lumen_diam_rad" in df.columns:
            drad = pd.to_numeric(df["lumen_diam_rad"], errors="coerce")
            rtsr = pd.Series(np.nan, index=df.index, dtype="float64")
            mask = cwttan.notna() & drad.notna() & (drad > 0)
            rtsr.loc[mask] = (4.0 * cwttan.loc[mask]) / drad.loc[mask]
            df["RTSR"] = rtsr

        # CTSR = (4 * CWTALL) / circle_diameter(area-equivalent)
        if "lumen_area" in df.columns:
            la = pd.to_numeric(df["lumen_area"], errors="coerce")
            ctsr = pd.Series(np.nan, index=df.index, dtype="float64")
            circle_diam = 2.0 * np.sqrt(la / np.pi)
            mask = cwtall.notna() & la.notna() & (la > 0) & circle_diam.notna() & (circle_diam > 0)
            ctsr.loc[mask] = (4.0 * cwtall.loc[mask]) / circle_diam.loc[mask]
            df["CTSR"] = ctsr

        # TB2 = min( (2*CWTRAD/DRAD)^2, (2*CWTTAN/DTAN)^2 )
        if "lumen_diam_rad" in df.columns and "lumen_diam_tang" in df.columns:
            drad = pd.to_numeric(df["lumen_diam_rad"], errors="coerce")
            dtan = pd.to_numeric(df["lumen_diam_tang"], errors="coerce")

            tb2 = pd.Series(np.nan, index=df.index, dtype="float64")

            # radial component
            rad_val = pd.Series(np.nan, index=df.index, dtype="float64")
            mask_r = cwtrad.notna() & drad.notna() & (cwtrad > 0) & (drad > 0)
            rad_val.loc[mask_r] = ((2.0 * cwtrad.loc[mask_r]) / drad.loc[mask_r]) ** 2

            # tangential component
            tan_val = pd.Series(np.nan, index=df.index, dtype="float64")
            mask_t = cwttan.notna() & dtan.notna() & (cwttan > 0) & (dtan > 0)
            tan_val.loc[mask_t] = ((2.0 * cwttan.loc[mask_t]) / dtan.loc[mask_t]) ** 2

            # min of available
            both = rad_val.notna() & tan_val.notna()
            tb2.loc[both] = np.minimum(rad_val.loc[both], tan_val.loc[both])

            only_r = rad_val.notna() & ~tan_val.notna()
            tb2.loc[only_r] = rad_val.loc[only_r]

            only_t = tan_val.notna() & ~rad_val.notna()
            tb2.loc[only_t] = tan_val.loc[only_t]

            df["TB2"] = tb2

    def analyze_cells(self) -> pd.DataFrame:
        """Main method to analyze cells."""
        self._smooth_cells_array()
        self._find_cells_contours()
        self._compute_cells_lumina()
        self._compute_cell_walls()
        self._cluster_cells()
        self._get_cells_table()

        # >>> ADD THIS (must be before any ring-level aggregations use cells_table)
        # self._apply_cwt_filters()

        sample_type = self.config.get("sample_type", None)
        print("Sample type: ", sample_type)

        self._compute_cluster_sizes()

        return self.cells_table

    def _rings_linear_regression(self, coordinates: list) -> tuple:
        y, x = np.array(coordinates).T
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        return m, c

    def _compute_rings_metrics(self):
        """Compute metrics for rings."""

        # Compute rings regressions
        self.rings_table[["boundary_slope", "boundary_intercept"]] = (
            self.rings_table["RBXY"]
            .apply(self._rings_linear_regression)
            .apply(pd.Series)
        )
        self.rings_table["boundary_angle"] = np.rad2deg(
            np.arctan(self.rings_table["boundary_slope"])
        )

        # Compute rings average widths
        self.rings_table["ring_vert_width"] = np.nan
        self.rings_table.iloc[
            1:, self.rings_table.columns.tolist().index("ring_vert_width")
        ] = np.diff(self.rings_table["cells_above"].values)
        self.rings_table["ring_vert_width"] = self.rings_table[
            "ring_vert_width"
        ] / (self.config["pixels_per_um"] * self.cells_array.shape[1])
        self.rings_table.loc[
            ~self.rings_table["enabled"], "ring_vert_width"
        ] = np.nan

        # Compute ring angle width
        self.rings_table["ring_angle_width"] = self.rings_table[
            "ring_vert_width"
        ] * np.cos(
            np.deg2rad(self.rings_table["boundary_angle"].rolling(2).mean())
        )

    def _compute_ring_area(self) -> None:
        # Compute ring area (RA) in mm² and store in rings_table["RA"].

        h, w = self.cells_array.shape[:2]
        px_per_um = self.config["pixels_per_um"]

        # Ensure RA exists and reset
        self.rings_table["RA"] = np.nan

        # ring i exists between boundary i and i+1
        for i in range(len(self.rings_table) - 1):
            ring_row = i + 1  # ring is represented by the lower boundary row

            # Only compute RA for enabled rings (not enabled boundary)
            if "enabled" in self.rings_table.columns:
                if not bool(self.rings_table.loc[ring_row, "enabled"]):
                    continue

            bounds = np.array(
                self.rings_table["RBXY"][i]
                + self.rings_table["RBXY"][i + 1][::-1],
                dtype=np.int32
            )

            # bounds are (y, x) in tables, but fillPoly expects (x, y)
            bounds = np.flip(bounds, axis=1)

            canvas = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(canvas, [bounds], 1)

            area_px = int(canvas.sum())
            # px² → µm²
            area_um2 = area_px / (px_per_um ** 2)
            # µm² → mm²
            area_mm2 = area_um2 / 1e6
            self.rings_table.loc[i + 1, "RA"] = area_mm2

    def _compute_cno(self) -> None:
        # Compute CNO = number of cells per ring (using bot_ring_id).
        self.rings_table["CNO"] = np.nan

        if "bot_ring_id" not in self.cells_table.columns:
            return

        counts = (
            self.cells_table["bot_ring_id"]
            .dropna()
            .astype(int)
            .value_counts()
        )

        # rings_table index -> ring_id
        for ring_id, cnt in counts.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CNO"] = int(cnt)

        # disabled rings -> NaN
        self.rings_table.loc[~self.rings_table["enabled"], "CNO"] = np.nan

    def _compute_cd(self) -> None:
        self.rings_table["CD"] = np.nan

        if "CNO" not in self.rings_table.columns:
            return
        if "RA" not in self.rings_table.columns:
            return

        cno = pd.to_numeric(self.rings_table["CNO"], errors="coerce")
        ra = pd.to_numeric(self.rings_table["RA"], errors="coerce")

        valid = cno.notna() & ra.notna() & (ra > 0)

        self.rings_table.loc[valid, "CD"] = cno[valid] / ra[valid]

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, "CD"] = np.nan

    def _compute_cta(self) -> None:
        # Compute CTA = cumulative lumen area of all counted cells per ring (mm²).
        self.rings_table["CTA"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "lumen_area" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "lumen_area"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["lumen_area"] = pd.to_numeric(df["lumen_area"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "lumen_area"])

        cta_um2 = df.groupby(df["bot_ring_id"].astype(int))["lumen_area"].sum()
        cta_mm2 = cta_um2 / 1e6

        self.rings_table.loc[cta_mm2.index, "CTA"] = cta_mm2.values

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, "CTA"] = np.nan

    def _compute_rcta(self) -> None:
        # Compute RCTA = percentage of conductive area = 100 * CTA / RA.
        self.rings_table["RCTA"] = np.nan

        if "CTA" not in self.rings_table.columns:
            return
        if "RA" not in self.rings_table.columns:
            return

        cta = pd.to_numeric(self.rings_table["CTA"], errors="coerce")
        ra = pd.to_numeric(self.rings_table["RA"], errors="coerce")

        valid = cta.notna() & ra.notna() & (ra > 0)
        self.rings_table.loc[valid, "RCTA"] = 100.0 * (cta[valid] / ra[valid])

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, "RCTA"] = np.nan

    def _compute_mla(self) -> None:
        # Compute MLA = mean lumen area per ring (µm²).
        self.rings_table["MLA"] = np.nan

        if "CTA" not in self.rings_table.columns:
            return
        if "CNO" not in self.rings_table.columns:
            return

        cta_mm2 = pd.to_numeric(self.rings_table["CTA"], errors="coerce")
        cno = pd.to_numeric(self.rings_table["CNO"], errors="coerce")

        valid = cta_mm2.notna() & cno.notna() & (cno > 0)
        self.rings_table.loc[valid, "MLA"] = (cta_mm2[valid] / cno[valid]) * 1e6

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, "MLA"] = np.nan

    def _compute_minla_maxla(self) -> None:
        # Compute MINLA and MAXLA = min/max lumen area per ring (µm²).
        self.rings_table["MINLA"] = np.nan
        self.rings_table["MAXLA"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "lumen_area" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "lumen_area"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["lumen_area"] = pd.to_numeric(df["lumen_area"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "lumen_area"])

        grouped = df.groupby(df["bot_ring_id"].astype(int))["lumen_area"]
        minla = grouped.min()
        maxla = grouped.max()

        self.rings_table.loc[minla.index, "MINLA"] = minla.values
        self.rings_table.loc[maxla.index, "MAXLA"] = maxla.values

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, ["MINLA", "MAXLA"]] = np.nan

    def _compute_kh_ring(self) -> None:
        # Compute ring-level KH as sum of cell-level KH within each ring.
        self.rings_table["KH"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "KH" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "KH"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["KH"] = pd.to_numeric(df["KH"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "KH"])

        kh_sum = df.groupby(df["bot_ring_id"].astype(int))["KH"].sum()

        self.rings_table.loc[kh_sum.index, "KH"] = kh_sum.values

        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, "KH"] = np.nan

    def _compute_ks(self) -> None:
        # Compute KS = KH / RA_m2. RA must be in m² (convert if RA column is stored as mm²).
        self.rings_table["KS"] = np.nan

        if "KH" not in self.rings_table.columns:
            return
        if "RA" not in self.rings_table.columns:
            return

        kh = pd.to_numeric(self.rings_table["KH"], errors="coerce")
        ra_mm2 = pd.to_numeric(self.rings_table["RA"], errors="coerce")

        # Convert RA to m²
        ra_m2 = ra_mm2 * 1e-6

        valid = (kh > 0) & (ra_m2 > 0)

        self.rings_table.loc[valid, "KS"] = kh[valid] / ra_m2[valid]

        self.rings_table.loc[~self.rings_table["enabled"], "KS"] = np.nan

    def _compute_vessel_grouping_metrics(self) -> None:
        """
        Compute vessel grouping metrics per ring:
          - RVGI: Vessel Grouping Index (mean number of cells per group; solitary cells count as group size 1)
          - RVSF: Vessel Solitary Fraction [%]
          - RGSGV: Mean group size of grouped / non-solitary cells e.g. groups with size > 1

        For conifers: metrics are not applicable and are always NA.
        For angiosperms: computed based on cluster IDs (cell-level "cluster") within each ring.
        """

        # Init columns
        self.rings_table["RVGI"] = np.nan
        self.rings_table["RVSF"] = np.nan
        self.rings_table["RGSGV"] = np.nan

        # Only compute for non-conifers
        if self.config.get("sample_type", None) == "conifer":
            return

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "cluster" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "cluster"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "cluster"])

        if df.empty:
            return

        # Iterate rings and compute metrics ring-wise
        for ring_id, ring_df in df.groupby(df["bot_ring_id"].astype(int)):
            if ring_id not in self.rings_table.index:
                continue

            # Total cells in ring
            n_total = len(ring_df)
            if n_total <= 0:
                continue

            # Ring-internal group sizes (clusters counted within ring)
            group_sizes = ring_df["cluster"].value_counts()

            n_groups = len(group_sizes)
            if n_groups <= 0:
                continue

            # RVGI: mean number of cells per group (solitary = group size 1)
            rvgi = n_total / n_groups

            # Solitary fraction: groups of size 1 correspond to solitary cells
            n_solitary = int((group_sizes == 1).sum())  # number of solitary groups
            # solitary cells = number of solitary groups * 1, so same number
            rvsf = 100.0 * (n_solitary / n_total)

            # RGSGV: mean group size for grouped (non-solitary) groups only
            grouped_sizes = group_sizes[group_sizes > 1]
            if len(grouped_sizes) > 0:
                rgsgv = float(grouped_sizes.mean())
            else:
                rgsgv = np.nan

            self.rings_table.loc[ring_id, "RVGI"] = float(rvgi)
            self.rings_table.loc[ring_id, "RVSF"] = float(rvsf)
            self.rings_table.loc[ring_id, "RGSGV"] = rgsgv

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            disabled = self.rings_table["enabled"] == False
            self.rings_table.loc[disabled, ["RVGI", "RVSF", "RGSGV"]] = np.nan

    def _compute_mean_cwtba(self) -> None:
        # CWTBA = Mean thickness of outer (bark-facing) cell wall per ring [µm]. Uses cell-level CWT_bark and aggregates by bot_ring_id.
        self.rings_table["CWTBA"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWT_bark" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWT_bark"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWT_bark"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWT_bark"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTBA"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTBA"] = np.nan

    def _compute_mean_cwtle(self) -> None:
        # CWTLE = Mean thickness of left cell wall (viewed from pith) per ring [µm]. Uses cell-level CWT_left and aggregates by bot_ring_id.
        self.rings_table["CWTLE"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWT_left" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWT_left"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWT_left"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWT_left"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTLE"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTLE"] = np.nan

    def _compute_mean_cwtri(self) -> None:
        # CWTRI = Mean thickness of right cell wall (viewed from pith) per ring [µm]. Uses cell-level CWT_right and aggregates by bot_ring_id.
        self.rings_table["CWTRI"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWT_right" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWT_right"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWT_right"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWT_right"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTRI"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTRI"] = np.nan

    def _compute_mean_cwttan(self) -> None:
        # CWTTAN = Mean thickness of tangential cell walls per ring [µm]. Uses cell-level CWTTAN and aggregates by bot_ring_id.
        self.rings_table["CWTTAN"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWTTAN" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWTTAN"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWTTAN"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWTTAN"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTTAN"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTTAN"] = np.nan

    def _compute_mean_cwtrad(self) -> None:
        # CWTRAD = Mean thickness of radial cell walls per ring [µm]. Uses cell-level CWTRAD and aggregates by bot_ring_id.
        self.rings_table["CWTRAD"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWTRAD" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWTRAD"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWTRAD"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWTRAD"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTRAD"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTRAD"] = np.nan

    def _compute_mean_cwtall(self) -> None:
        # CWTALL = Mean thickness of all cell walls per ring [µm]. Uses cell-level CWTALL and aggregates by bot_ring_id.
        self.rings_table["CWTALL"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWTALL" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWTALL"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWTALL"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWTALL"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTALL"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTALL"] = np.nan

    def _compute_mean_rtsr(self) -> None:
        # RTSR = Mean radial Thickness-to-span ratio per ring (Mork’s index). Uses cell-level RTSR and aggregates by bot_ring_id.
        self.rings_table["RTSR"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "RTSR" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "RTSR"]].copy()
        df = df.dropna(subset=["bot_ring_id", "RTSR"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["RTSR"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "RTSR"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "RTSR"] = np.nan

    def _compute_mean_ctsr(self) -> None:
        # CTSR = Mean circular Thickness-to-span ratio per ring. Uses cell-level CTSR and aggregates by bot_ring_id.
        self.rings_table["CTSR"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CTSR" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CTSR"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CTSR"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CTSR"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CTSR"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CTSR"] = np.nan

    def _compute_mean_dh(self) -> None:
        # DH = hydraulically weighted mean diameter per ring: sum(DH^5) / sum(DH^4). Uses cell-level DH and aggregates by bot_ring_id.
        self.rings_table["DH"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "DH" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "DH"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["DH"] = pd.to_numeric(df["DH"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "DH"])

        if df.empty:
            return

        # Keep only positive DH values
        df = df[df["DH"] > 0]
        if df.empty:
            return

        # group by ring id and compute hydraulically weighted diameter
        for ring_id, ring_df in df.groupby(df["bot_ring_id"].astype(int)):
            if ring_id not in self.rings_table.index:
                continue

            dh = ring_df["DH"].values.astype(float)
            denom = np.sum(dh ** 4)
            if denom <= 0 or np.isnan(denom):
                continue

            num = np.sum(dh ** 5)
            self.rings_table.loc[ring_id, "DH"] = float(num / denom)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DH"] = np.nan

    def _compute_mean_dh2(self) -> None:
        # DH2 = mean hydraulic diameter per ring: (sum(DH^4) / N)^0.25. Uses cell-level DH and aggregates by bot_ring_id.
        self.rings_table["DH2"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "DH" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "DH"]].copy()
        df["bot_ring_id"] = pd.to_numeric(df["bot_ring_id"], errors="coerce")
        df["DH"] = pd.to_numeric(df["DH"], errors="coerce")
        df = df.dropna(subset=["bot_ring_id", "DH"])

        if df.empty:
            return

        # Keep only positive DH values
        df = df[df["DH"] > 0]
        if df.empty:
            return

        # group by ring id and compute DH2
        for ring_id, ring_df in df.groupby(df["bot_ring_id"].astype(int)):
            if ring_id not in self.rings_table.index:
                continue

            dh = ring_df["DH"].values.astype(float)
            n = dh.size
            if n <= 0:
                continue

            mean_dh4 = np.sum(dh ** 4) / n
            if mean_dh4 <= 0 or np.isnan(mean_dh4):
                continue

            self.rings_table.loc[ring_id, "DH2"] = float(mean_dh4 ** 0.25)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DH2"] = np.nan

    def _compute_mean_drad(self) -> None:
        # DRAD = Mean radial cell lumen diameter per ring [µm]. Uses cell-level lumen_diam_rad and aggregates by bot_ring_id.
        self.rings_table["DRAD"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "lumen_diam_rad" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "lumen_diam_rad"]].copy()
        df = df.dropna(subset=["bot_ring_id", "lumen_diam_rad"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["lumen_diam_rad"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "DRAD"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DRAD"] = np.nan

    def _compute_mean_dtan(self) -> None:
        # DTAN = Mean tangential cell lumen diameter per ring [µm]. Uses cell-level lumen_diam_tang and aggregates by bot_ring_id.
        self.rings_table["DTAN"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "lumen_diam_tang" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "lumen_diam_tang"]].copy()
        df = df.dropna(subset=["bot_ring_id", "lumen_diam_tang"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["lumen_diam_tang"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "DTAN"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DTAN"] = np.nan

    def _compute_mean_tb2(self) -> None:
        # TB2 = Mean cell wall reinforcement index (t/b)^2 per ring. Uses cell-level TB2 and aggregates by bot_ring_id.
        self.rings_table["TB2"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "TB2" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "TB2"]].copy()
        df = df.dropna(subset=["bot_ring_id", "TB2"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["TB2"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "TB2"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "TB2"] = np.nan

    def _compute_mean_cwa(self) -> None:
        # CWA = Mean cell wall area per ring [µm²]. Uses cell-level CWA and aggregates by bot_ring_id.
        self.rings_table["CWA"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWA" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWA"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWA"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWA"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWA"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWA"] = np.nan

    def _compute_mean_rwd(self) -> None:
        # RWD = Mean relative anatomical cell density per ring. Uses cell-level RWD and aggregates by bot_ring_id.
        self.rings_table["RWD"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "RWD" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "RWD"]].copy()
        df = df.dropna(subset=["bot_ring_id", "RWD"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["RWD"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "RWD"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "RWD"] = np.nan

    def _compute_mean_cwtpi(self) -> None:
        # CWTPI = Mean thickness of inner (pith-facing) cell wall per ring [µm]. Uses cell-level CWT_pith and aggregates by bot_ring_id.
        self.rings_table["CWTPI"] = np.nan

        if self.cells_table.empty:
            return
        if "bot_ring_id" not in self.cells_table.columns:
            return
        if "CWT_pith" not in self.cells_table.columns:
            return

        df = self.cells_table[["bot_ring_id", "CWT_pith"]].copy()
        df = df.dropna(subset=["bot_ring_id", "CWT_pith"])

        if df.empty:
            return

        # group by ring id and compute mean
        ring_mean = df.groupby(df["bot_ring_id"].astype(int))["CWT_pith"].mean()

        for ring_id, val in ring_mean.items():
            if ring_id in self.rings_table.index:
                self.rings_table.loc[ring_id, "CWTPI"] = float(val)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "CWTPI"] = np.nan

    def _get_angled_distances(self, entry):
        """Compute angled distances for top and bottom rings."""

        if np.isnan(entry["bot_ring_id"]):
            return pd.Series(
                [np.nan, np.nan], index=["top_angled_dist", "bot_angled_dist"]
            )

        top_ring_angle = self.rings_table.loc[
            entry["bot_ring_id"] - 1, "boundary_angle"
        ]
        bot_ring_angle = self.rings_table.loc[
            entry["bot_ring_id"], "boundary_angle"
        ]

        avg_angle = (top_ring_angle + bot_ring_angle) / 2

        top_dist = entry["top_vert_dist"] * np.cos(np.deg2rad(avg_angle))
        bot_dist = entry["bot_vert_dist"] * np.cos(np.deg2rad(avg_angle))
        return pd.Series(
            [top_dist, bot_dist], index=["top_angled_dist", "bot_angled_dist"]
        )

    def _compute_cells_to_rings_distances(self):
        """Compute distances from cells to rings."""

        # Create a map of cell centroids to their corresponding cell IDs
        centroids_map = np.ones_like(self.cells_array).astype("int32") * -1
        for i, centroid in enumerate(self.cells_table["centroid"]):
            if not np.isnan(centroid).any():
                centroids_map[centroid] = i

        # Iitialize columns
        self.cells_table["bot_ring_id"] = np.nan
        self.cells_table["top_vert_dist"] = np.nan
        self.cells_table["bot_vert_dist"] = np.nan

        for i in range(len(self.rings_table) - 1):

            # Get cells in the current ring
            bounds = np.flip(
                np.array(
                    self.rings_table["RBXY"][i]
                    + self.rings_table["RBXY"][i + 1][::-1],
                    dtype=np.int32,
                ),
                axis=1,
            )
            canvas = np.zeros_like(self.cells_array)
            cv2.fillPoly(canvas, [bounds], 1)

            ids = centroids_map[
                np.where(
                    np.logical_and(centroids_map >= 0, canvas.astype(bool))
                )
            ]
            self.cells_table.loc[ids, "bot_ring_id"] = i + 1
            centroids = np.array(
                self.cells_table.loc[ids, "centroid"].tolist()
            )

            diff = np.diff(canvas.astype("int32"), axis=0)
            complete_top_ring = np.all(np.any(diff > 0, axis=0))
            complete_bottom_ring = np.all(np.any(diff < 0, axis=0))

            if complete_top_ring:
                self.cells_table.loc[ids, "top_vert_dist"] = (
                    centroids[:, 0]
                    - np.argmax(diff > 0, axis=0)[centroids[:, 1]]
                ) / self.config["pixels_per_um"]

            if complete_bottom_ring:
                self.cells_table.loc[ids, "bot_vert_dist"] = (
                    np.argmax(diff < 0, axis=0)[centroids[:, 1]]
                    - centroids[:, 0]
                ) / self.config["pixels_per_um"]

        self.cells_table[["top_angled_dist", "bot_angled_dist"]] = (
            self.cells_table.apply(self._get_angled_distances, axis=1)
        )
        self.cells_table["YEAR"] = (
            self.cells_table["bot_ring_id"]
            .map(self.rings_table["YEAR"])
        )
        self.cells_table["YEAR"] = (
            self.cells_table["YEAR"]
            .astype("Int64")
        )
        # cells outside all ring polygons (outermost incomplete band) ---
        # Those cells have bot_ring_id = NaN -> YEAR becomes NA.
        # Assign them to last_year + 1 (outermost incomplete ring year).
        missing_year = self.cells_table["YEAR"].isna()
        if missing_year.any():
            last_year = pd.to_numeric(self.rings_table.get("YEAR"), errors="coerce").max()
            if pd.notna(last_year):
                self.cells_table.loc[missing_year, "YEAR"] = int(last_year) + 1
                self.cells_table["YEAR"] = self.cells_table["YEAR"].astype("Int64")


        self.compute_rraddistr()

    def compute_rraddistr(self) -> None:
        """
        Compute RRADDISTR (relative radial distance within the annual ring).

        Definition:
            RRADDISTR = 100 * (top_angled_dist / RingWidth_local)

        where
            top_angled_dist  = distance from the proximal (inner / upper) ring boundary
                              to the cell centre
            RingWidth_local = distance from proximal to distal boundary along the
                              local radial line  → top_angled_dist + bot_angled_dist

        Interpretation:
            0%   → cell centre at proximal (upper) boundary
            100% → cell centre at distal (lower) boundary
        """

        # Ensure numeric types; invalid entries become NaN
        top = pd.to_numeric(self.cells_table["top_angled_dist"], errors="coerce")
        bot = pd.to_numeric(self.cells_table["bot_angled_dist"], errors="coerce")

        # Local ring width along the radial line through the cell
        ring_width_local = top + bot

        # Initialise RRADDISTR with NaN
        rr = pd.Series(np.nan, index=self.cells_table.index, dtype="float64")

        # Valid only with positive width (avoid division by zero)
        valid = ring_width_local > 0

        # Relative radial position in percent, 0% at proximal (top) boundary
        rr[valid] = 100.0 * top[valid] / ring_width_local[valid]

        # Assign back to the table
        self.cells_table["RRADDISTR"] = rr

        # cleanup: discard values outside the physical [0, 100] range
        self.cells_table.loc[self.cells_table["RRADDISTR"] < 0, "RRADDISTR"] = np.nan
        self.cells_table.loc[self.cells_table["RRADDISTR"] > 100, "RRADDISTR"] = np.nan


    def analyze_rings(self) -> pd.DataFrame:
        """Main method to analyze rings."""
        self._compute_rings_metrics()
        if not self.cells_table.empty:
            self._compute_cells_to_rings_distances()

        self._compute_ring_area()
        self._compute_cno()
        self._compute_cd()
        self._compute_cta()
        self._compute_rcta()
        self._compute_mla()
        self._compute_minla_maxla()
        self._compute_kh_ring()
        self._compute_ks()
        self._compute_vessel_grouping_metrics()
        self._compute_mean_cwtpi()
        self._compute_mean_cwtba()
        self._compute_mean_cwtle()
        self._compute_mean_cwtri()
        self._compute_mean_cwttan()
        self._compute_mean_cwtrad()
        self._compute_mean_cwtall()
        self._compute_mean_rtsr()
        self._compute_mean_ctsr()
        self._compute_mean_dh()
        self._compute_mean_dh2()
        self._compute_mean_drad()
        self._compute_mean_dtan()
        self._compute_mean_tb2()
        self._compute_mean_cwa()
        self._compute_mean_rwd()

        return self.rings_table

    def analyze_sample(self) -> tuple:
        """Main method to analyze sample."""
        self.analyze_cells()
        self.analyze_rings()

        return self.cells_table, self.rings_table


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
        "input",
        type=Path,
        help="Stem path to the sample image (without suffixes)",
    )
    args = parser.parse_args()

    sample_path = Path(args.input)

    with Image.open(
        sample_path.with_suffix("".join(sample_path.suffixes) + ".cells.png")
    ) as img:
        cells_array = np.array(img).astype("uint8")

    rings_table = pd.read_csv(
        sample_path.with_suffix(
            "".join(sample_path.suffixes) + ".rings_table.txt"
        ),
        sep="\t",
        index_col=0,
        converters={"RBXY": ast.literal_eval},
    )

    # Initialize the analyzer
    analyzer = SampleAnalyzer(CONFIG, cells_array, rings_table)
    # Perform analysis
    cells_table, rings_table = analyzer.analyze_sample()

    print(cells_table.head())
    print(rings_table.head())
