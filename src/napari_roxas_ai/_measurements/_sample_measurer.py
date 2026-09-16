"""
Module for analyzing segmented wood cells and rings images to measure cell wall thickness, rings width and related metrics.
Based on code by GitHub user triyan-b https://github.com/triyan-b
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
                - cluster_dbl_cwt_threshold: Minimum distance between clusters (µm)
                - smoothing_kernel_size: Size of morphological operation kernel
                - relwidth_cwt_integration: Fraction of wall used for thickness measurement
                - tangential_angle : Sample angle (degrees, clockwise)
                - lower_limit_cwt_iqr_multiplier: IQR multiplier for the lower CWT outlier fence
                - upper_limit_cwt_iqr_multiplier: IQR multiplier for the upper CWT outlier fence
                - opposite_cwt_ratio_limit: Max CWT ratio between opposite cell sides
                - adjacent_cwt_ratio_limit: Max CWT ratio between a side and its adjacent sides
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
        self.pixels_per_um = float(config["pixels_per_um"])
        self.cluster_dbl_cwt_px = (
            config["cluster_dbl_cwt_threshold"] * self.pixels_per_um
        )
        self.kernel = np.ones(
            (config["smoothing_kernel_size"], config["smoothing_kernel_size"])
        )
        self.relwidth_cwt_margin = (
            1 - config["relwidth_cwt_integration"]
        ) / 2
        self.radial_angle = config["tangential_angle"] - 90
        self.lower_limit_cwt_iqr_multiplier = self._config_float(
            "lower_limit_cwt_iqr_multiplier", 1.5
        )
        self.upper_limit_cwt_iqr_multiplier = self._config_float(
            "upper_limit_cwt_iqr_multiplier", 3.0
        )
        self.opposite_cwt_ratio_limit = self._config_float(
            "opposite_cwt_ratio_limit", 1.5
        )
        self.adjacent_cwt_ratio_limit = self._config_float(
            "adjacent_cwt_ratio_limit", 3.0
        )

    def _config_float(self, key: str, default: float) -> float:
        """
        Read a float from config, falling back to default.

        The fallback also covers a present-but-None value, which happens when a
        settings.json predates the key and SettingsManager.get() returns None.
        """
        value = self.config.get(key)
        return default if value is None else float(value)

    # TODO: Check whether smoothing is necessary / desired after DL cell detection
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

        # Sort contours by centroid (YPIX, XPIX) ascending so cell ID 0
        # starts at the image top-left, consistent with ring ordering.
        def _centroid_key(contour):
            M = cv2.moments(contour)
            if M["m00"] != 0:
                return (int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"]))
            return (float("inf"), float("inf"))

        self.lumen_contours = sorted(self.lumen_contours, key=_centroid_key)

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
                        "lumen_area": M["m00"] / self.pixels_per_um ** 2,
                        "lumen_peri": cv2.arcLength(contour, True) / self.pixels_per_um,
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
                        "major_axis_px": np.nan,
                        "minor_axis_px": np.nan,
                        "angle_major": np.nan,
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
            angle_major = angle
        else:
            a, b = h / 2, w / 2
            angle_major = angle + 90

        cell.update(
            {
                "major_axis_px": a,
                "minor_axis_px": b,
                "angle_major": angle_major,
            }
        )

    def _compute_lumen_metrics(self) -> None:
        """
        Vectorized computation of lumen-based metrics:
        KH, DH, MAJAX, ASP, and diameters (DRAD, DTAN).
        """
        if self.cells_table is None or self.cells_table.empty:
            return

        df = self.cells_table

        # Ensure we have the required base columns from ellipse fitting
        if not all(col in df.columns for col in ["major_axis_px", "minor_axis_px", "angle_major"]):
            return

        a_px = pd.to_numeric(df["major_axis_px"], errors="coerce")
        b_px = pd.to_numeric(df["minor_axis_px"], errors="coerce")
        angle_major = pd.to_numeric(df["angle_major"], errors="coerce")
        la_um2 = pd.to_numeric(df.get("lumen_area"), errors="coerce")

        # 1. MAJAX: deviation from image vertical (90°) normalized to [0, 90]
        norm_angle = angle_major % 180
        majax = (norm_angle - 90).abs()
        df["MAJAX"] = majax.where(majax <= 90, 180 - majax)

        # 2. ASP: Aspect ratio a/b
        df["ASP"] = (a_px / b_px).where(b_px > 0)

        # 3. Diameters DRAD and DTAN
        aoma_rad_deg = angle_major - self.radial_angle
        aoma_tang_deg = aoma_rad_deg + 90

        def _calc_diam(a, b, angle_deg):
            rad = np.deg2rad(angle_deg)
            denom = np.sqrt((b * np.cos(rad))**2 + (a * np.sin(rad))**2)
            return (2.0 * a * b / denom).where(denom > 0)

        df["lumen_diam_rad"] = _calc_diam(a_px, b_px, aoma_rad_deg) / self.pixels_per_um
        df["lumen_diam_tang"] = _calc_diam(a_px, b_px, aoma_tang_deg) / self.pixels_per_um

        # 4. DH: Hydraulic diameter for elliptical conduit (Lewis & Boose 1995)
        asp = (a_px / b_px).where(b_px > 0)
        a_semi_um = np.sqrt(asp * la_um2 / np.pi)
        b_semi_um = (la_um2 / (np.pi * a_semi_um)).where(a_semi_um > 0)

        a_um = 2 * a_semi_um
        b_um = 2 * b_semi_um
        a2, b2 = a_um**2, b_um**2
        df["DH"] = np.sqrt((2.0 * a2 * b2) / (a2 + b2)).where((a2 + b2) > 0)

        # 5. KH: Theoretical hydraulic conductance
        la_m2 = la_um2 * 1e-12
        a_m = np.sqrt(asp * la_m2 / np.pi)
        b_m = (la_m2 / (np.pi * a_m)).where(a_m > 0)

        # Eccentricity
        e2 = (a_m**2 - b_m**2) / (a_m**2).where(a_m > 0)
        e4 = e2**2
        # Circumference C (Ramanujan approximation)
        C = np.pi * (3 * (a_m + b_m) - np.sqrt((3 * a_m + b_m) * (a_m + 3 * b_m)))
        # Mean hydraulic radius m = Area / Circumference
        m = la_m2 / C.where(C > 0)
        # Form factor k
        k = 4.0 / (1.0 + np.sqrt((1.0 - e4).where(e4 <= 1, 0)))
        nu = 1.002e-9 # viscosity of water (MPa·s)
        df["KH"] = (la_m2 * m**2) / (nu * k.where(k > 0))

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
                    "cell_area": M["m00"] / self.pixels_per_um ** 2,
                    "cw_peri": cv2.arcLength(contour, True)
                    / self.pixels_per_um,
                }
            )

        # Calculate wall thickness measurements
        self._measure_cell_wall_thickness(contour, cell_id)

    def _find_contained_cell(self, contour: np.ndarray) -> int:
        """Find cell ID contained within the wall contour."""
        x, y, w, h = cv2.boundingRect(contour)

        # Make sure it does not get out of the image (a 1 pixel offset is possible)
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

        # Make sure it does not get out of the image (a 1 pixel offset is possible)
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
            # Get wall pixels where the label corresponds
            wall_pixel_coords = contour_points[
                np.where(contour_labels == label)[0], :
            ]

            # Crop to keep the middle % as defined by relwidth_cwt_margin (typically 75%)
            lower_bound = np.ceil(
                self.relwidth_cwt_margin * wall_pixel_coords.shape[0]
            ).astype("int32")
            upper_bound = np.ceil(
                (1 - self.relwidth_cwt_margin) * wall_pixel_coords.shape[0]
            ).astype("int32")

            # Write the mean thickness in the cell dict (might be median in the future)
            avg_dist = dist_crop[
                wall_pixel_coords[lower_bound:upper_bound, 1],
                wall_pixel_coords[lower_bound:upper_bound, 0],
            ].mean()
            self.cells[cell_id].update(
                {
                    f"CWT_{label_map[label]}": avg_dist
                    / self.pixels_per_um,
                }
            )

    def _cluster_cells(self) -> None:
        """Cluster cells based on proximity."""
        # Threshold distance transform for clustering
        _, dist_thresh = cv2.threshold(
            self.dist_transform / self.pixels_per_um,
            self.config["cluster_dbl_cwt_threshold"],
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
        self.cells_table = pd.DataFrame(self.cells).T
        if not self.cells_table.empty:
            self.cells_table.index.name = "id"
        return self.cells_table

    def _apply_cwt_filters(self) -> None:
        """
        Automatic filtering of cell wall thickness (CWT) measurements using the IQR method for outlier detection. (Tukey's fences method)

        Notes
        -----
        - This function modifies self.cells_table in-place.
        """

        if self.cells_table is None or self.cells_table.empty:
            return

        required = ["CWT_pith", "CWT_bark", "CWT_left", "CWT_right"]
        if not all(col in self.cells_table.columns for col in required):
            return

        # --- Settings ---
        lower_limit_cwt_iqr_multiplier = self.lower_limit_cwt_iqr_multiplier
        upper_limit_cwt_iqr_multiplier = self.upper_limit_cwt_iqr_multiplier
        opposite_cwt_ratio_limit = self.opposite_cwt_ratio_limit
        adjacent_cwt_ratio_limit = self.adjacent_cwt_ratio_limit
        min_plausible = 1.0 / self.pixels_per_um  # 1 pixel in µm (sub-pixel range is implausible)

        df = self.cells_table

        # Pull columns as numeric
        pi = pd.to_numeric(df["CWT_pith"], errors="coerce")
        ba = pd.to_numeric(df["CWT_bark"], errors="coerce")
        le = pd.to_numeric(df["CWT_left"], errors="coerce")
        ri = pd.to_numeric(df["CWT_right"], errors="coerce")

        # --- combined quantiles ---
        tan_all = pd.concat([pi, ba], ignore_index=True).dropna()
        rad_all = pd.concat([le, ri], ignore_index=True).dropna()

        if tan_all.empty or rad_all.empty:
            return

        q1_tan, q3_tan = tan_all.quantile([0.25, 0.75])
        q1_rad, q3_rad = rad_all.quantile([0.25, 0.75])

        # --- Apply hard limits ---
        iqr_tan = max(0.0, q3_tan - q1_tan)
        iqr_rad = max(0.0, q3_rad - q1_rad)

        ll_tan = max(
            q1_tan - lower_limit_cwt_iqr_multiplier * iqr_tan, min_plausible
        )
        ul_tan = q3_tan + upper_limit_cwt_iqr_multiplier * iqr_tan

        ll_rad = max(
            q1_rad - lower_limit_cwt_iqr_multiplier * iqr_rad, min_plausible
        )
        ul_rad = q3_rad + upper_limit_cwt_iqr_multiplier * iqr_rad

        def _apply_limits(s, ll, ul):
            s = s.copy()
            s.loc[(s < ll) | (s > ul)] = np.nan
            return s

        pi = _apply_limits(pi, ll_tan, ul_tan)
        ba = _apply_limits(ba, ll_tan, ul_tan)
        le = _apply_limits(le, ll_rad, ul_rad)
        ri = _apply_limits(ri, ll_rad, ul_rad)

        # --- Filter larger value of opposite sides if much larger ---
        ba.loc[ba > (opposite_cwt_ratio_limit * pi)] = np.nan
        pi.loc[pi > (opposite_cwt_ratio_limit * ba)] = np.nan
        le.loc[le > (opposite_cwt_ratio_limit * ri)] = np.nan
        ri.loc[ri > (opposite_cwt_ratio_limit * le)] = np.nan

        # --- Filter larger value compared to adjacent sides if much larger ---
        ave_lr = pd.concat([le, ri], axis=1).mean(axis=1, skipna=True)
        ave_pb = pd.concat([pi, ba], axis=1).mean(axis=1, skipna=True)

        ba.loc[ba > (adjacent_cwt_ratio_limit * ave_lr)] = np.nan
        pi.loc[pi > (adjacent_cwt_ratio_limit * ave_lr)] = np.nan
        le.loc[le > (adjacent_cwt_ratio_limit * ave_pb)] = np.nan
        ri.loc[ri > (adjacent_cwt_ratio_limit * ave_pb)] = np.nan

        # Write back filtered base values
        df["CWT_pith"] = pi
        df["CWT_bark"] = ba
        df["CWT_left"] = le
        df["CWT_right"] = ri

    def _compute_cwt_metrics(self) -> None:
        """
        Compute dependent CWT metrics (CWTTAN, CWTRAD, CWTALL, RTSR, CTSR, TB2, CWA, RWD)
        based on current base wall thickness values.
        """
        if self.cells_table is None or self.cells_table.empty:
            return

        required = ["CWT_pith", "CWT_bark", "CWT_left", "CWT_right"]
        if not all(col in self.cells_table.columns for col in required):
            return

        df = self.cells_table

        pi = pd.to_numeric(df["CWT_pith"], errors="coerce")
        ba = pd.to_numeric(df["CWT_bark"], errors="coerce")
        le = pd.to_numeric(df["CWT_left"], errors="coerce")
        ri = pd.to_numeric(df["CWT_right"], errors="coerce")

        def _get_avg_positive(s1, s2):
            v1 = s1.where(s1 > 0)
            v2 = s2.where(s2 > 0)
            return pd.concat([v1, v2], axis=1).mean(axis=1, skipna=True)

        df["CWTTAN"] = _get_avg_positive(pi, ba)
        df["CWTRAD"] = _get_avg_positive(le, ri)
        df["CWTALL"] = _get_avg_positive(df["CWTTAN"], df["CWTRAD"])

        if "lumen_diam_rad" in df.columns:
            drad = pd.to_numeric(df["lumen_diam_rad"], errors="coerce")
            df["RTSR"] = (4.0 * df["CWTTAN"]) / drad.where(drad > 0)

        if "lumen_area" in df.columns:
            la = pd.to_numeric(df["lumen_area"], errors="coerce")
            circle_diam = 2.0 * np.sqrt(la / np.pi)
            df["CTSR"] = (4.0 * df["CWTALL"]) / circle_diam.where(circle_diam > 0)

        if "lumen_diam_rad" in df.columns and "lumen_diam_tang" in df.columns:
            dtan = pd.to_numeric(df["lumen_diam_tang"], errors="coerce")
            drad = pd.to_numeric(df["lumen_diam_rad"], errors="coerce")

            rad_comp = (2.0 * df["CWTRAD"].where(df["CWTRAD"] > 0) / drad.where(drad > 0))**2
            tan_comp = (2.0 * df["CWTTAN"].where(df["CWTTAN"] > 0) / dtan.where(dtan > 0))**2
            df["TB2"] = pd.concat([rad_comp, tan_comp], axis=1).min(axis=1, skipna=True)

        if all(col in df.columns for col in ["cell_area", "lumen_area"]):
            ca = pd.to_numeric(df["cell_area"], errors="coerce")
            la = pd.to_numeric(df["lumen_area"], errors="coerce")

            # CWA and RWD only if all 4 base CWT measurements are not NaN
            all_cwt_present = pi.notna() & ba.notna() & le.notna() & ri.notna()

            cwa = (ca - la).where(all_cwt_present & (ca > la) & (ca > 0) & (la > 0))
            df["CWA"] = cwa
            df["RWD"] = (cwa / ca).where(cwa.notna() & (ca > 0))

    def analyze_cells(self) -> pd.DataFrame:
        """Main method to analyze cells."""
        self._smooth_cells_array()
        self._find_cells_contours()
        self._compute_cells_lumina()

        # Initial table creation
        self._get_cells_table()

        self._compute_cell_walls()

        # Sync updates from self.cells (from _compute_cell_walls) back to cells_table
        self._get_cells_table()

        self._cluster_cells()

        # Sync updates from self.cells (from _cluster_cells) back to cells_table
        self._get_cells_table()

        # Compute lumen metrics (needs lumen_area, major_axis_px, etc. from _get_cells_table)
        self._compute_lumen_metrics()

        # Initialize empty column for "AOI"
        self.cells_table["AOI"] = np.nan

        # Filter CWT base measurements and compute dependent metrics
        self._apply_cwt_filters()
        self._compute_cwt_metrics()

        # Compute NBRNO and NBRID
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
        if "cells_above" in self.rings_table.columns and len(self.rings_table) > 1:
            self.rings_table.iloc[
                1:, self.rings_table.columns.tolist().index("ring_vert_width")
            ] = np.diff(self.rings_table["cells_above"].values)
            self.rings_table["ring_vert_width"] = self.rings_table[
                "ring_vert_width"
            ] / (self.pixels_per_um * self.cells_array.shape[1])
            
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[
                ~self.rings_table["enabled"], "ring_vert_width"
            ] = np.nan

        # Compute ring angle width
        self.rings_table["ring_angle_width"] = self.rings_table[
            "ring_vert_width"
        ] * np.cos(
            np.deg2rad(self.rings_table["boundary_angle"].rolling(2).mean())
        )

        # Initialize empty columns for "AOIAR" and "RAOIAR"
        self.rings_table["AOIAR"] = np.nan
        self.rings_table["RAOIAR"] = np.nan

    def _compute_ring_area(self) -> None:
        # Compute ring area (RA) in mm² and store in rings_table["RA"].
        h, w = self.cells_array.shape[:2]

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
            area_um2 = area_px / (self.pixels_per_um ** 2)
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
        if "enabled" in self.rings_table.columns:
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

        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "KS"] = np.nan

        return self.rings_table

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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        # RTSR = Mean radial Thickness-to-span ratio per ring (Mork's index). Uses cell-level RTSR and aggregates by bot_ring_id.
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        # DHW = hydraulically weighted mean diameter per ring: sum(DH^5) / sum(DH^4). Uses cell-level DH and aggregates by bot_ring_id.
        if self.rings_table is None or self.rings_table.empty:
            return
        self.rings_table["DHW"] = np.nan

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
            self.rings_table.loc[ring_id, "DHW"] = float(num / denom)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DHW"] = np.nan

    def _compute_mean_dh2(self) -> None:
        # DHM = mean hydraulic diameter per ring: (sum(DH^4) / N)^0.25. Uses cell-level DH and aggregates by bot_ring_id.
        if self.rings_table is None or self.rings_table.empty:
            return
        self.rings_table["DHM"] = np.nan

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

        # group by ring id and compute DHM
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

            self.rings_table.loc[ring_id, "DHM"] = float(mean_dh4 ** 0.25)

        # disabled rings -> NaN
        if "enabled" in self.rings_table.columns:
            self.rings_table.loc[~self.rings_table["enabled"], "DHM"] = np.nan

    def _compute_mean_drad(self) -> None:
        # DRAD = Mean radial cell lumen diameter per ring [µm]. Uses cell-level lumen_diam_rad and aggregates by bot_ring_id.
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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
        if self.rings_table is None or self.rings_table.empty:
            return
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

    def _compute_cells_to_rings_distances(self):
        """Compute distances from cells to rings."""

        # Create a map of cell centroids to their corresponding cell IDs
        centroids_map = np.ones_like(self.cells_array).astype("int32") * -1
        for i, centroid in enumerate(self.cells_table["centroid"]):
            if not np.isnan(centroid).any():
                centroids_map[centroid] = i

        # Initialize columns
        self.cells_table["bot_ring_id"] = np.nan
        self.cells_table["top_vert_dist"] = np.nan
        self.cells_table["bot_vert_dist"] = np.nan
        self.cells_table["top_angled_dist"] = np.nan
        self.cells_table["bot_angled_dist"] = np.nan

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
            if len(ids) == 0:
                continue

            self.cells_table.loc[ids, "bot_ring_id"] = i + 1
            centroids_list = self.cells_table.loc[ids, "centroid"].tolist()
            centroids = np.array(centroids_list)

            if centroids.ndim == 1:
                centroids = centroids.reshape(-1, 2)

            diff = np.diff(canvas.astype("int32"), axis=0)

            self.cells_table.loc[ids, "top_vert_dist"] = (
                centroids[:, 0]
                - np.argmax(diff > 0, axis=0)[centroids[:, 1]]
            ) / self.pixels_per_um

            self.cells_table.loc[ids, "bot_vert_dist"] = (
                np.argmax(diff < 0, axis=0)[centroids[:, 1]]
                - centroids[:, 0]
            ) / self.pixels_per_um

        # --- Vectorized Angled Distances ---
        df = self.cells_table
        if "bot_ring_id" in df.columns and not df["bot_ring_id"].isna().all():
            bot_ids = pd.to_numeric(df["bot_ring_id"], errors="coerce")

            top_ring_angles = (bot_ids - 1).map(self.rings_table["boundary_angle"])
            bot_ring_angles = bot_ids.map(self.rings_table["boundary_angle"])

            avg_angles = (top_ring_angles + bot_ring_angles) / 2.0
            cos_avg = np.cos(np.deg2rad(avg_angles))

            df["top_angled_dist"] = df["top_vert_dist"] * cos_avg
            df["bot_angled_dist"] = df["bot_vert_dist"] * cos_avg

        df["YEAR"] = (
            df["bot_ring_id"]
            .map(self.rings_table["YEAR"])
        )
        df["YEAR"] = (
            df["YEAR"]
            .astype("Int64")
        )
        # Cells outside all ring polygons (incomplete bands at top / bottom).
        # Those cells have bot_ring_id = NaN -> YEAR becomes NA.
        # Distinguish top vs bottom by comparing YPIX to the mean
        # y-coordinate of the first and last ring boundaries.
        missing_year = df["YEAR"].isna()
        if missing_year.any():
            years = pd.to_numeric(self.rings_table.get("YEAR"), errors="coerce")
            first_year = years.min()
            last_year = years.max()

            if pd.notna(first_year) and pd.notna(last_year):
                top_boundary_y = np.mean([pt[0] for pt in self.rings_table["RBXY"].iloc[0]])
                bot_boundary_y = np.mean([pt[0] for pt in self.rings_table["RBXY"].iloc[-1]])
                mid_y = (top_boundary_y + bot_boundary_y) / 2

                ypix = pd.to_numeric(df.loc[missing_year, "YPIX"], errors="coerce")
                is_top = ypix <= mid_y

                df.loc[missing_year & is_top.reindex(df.index, fill_value=False), "YEAR"] = int(first_year)
                df.loc[missing_year & (~is_top).reindex(df.index, fill_value=False), "YEAR"] = int(last_year) + 1
                df["YEAR"] = df["YEAR"].astype("Int64")


        self._compute_rraddistr()

    def _compute_rraddistr(self) -> None:
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
        if self.rings_table is None or self.rings_table.empty:
            return self.rings_table

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
        "cluster_dbl_cwt_threshold": 3,  # µm
        "smoothing_kernel_size": 5,
        "relwidth_cwt_integration": 0.75,
        "tangential_angle": 0,  # Assuming vertical orientation
        "lower_limit_cwt_iqr_multiplier": 1.5,
        "upper_limit_cwt_iqr_multiplier": 3.0,
        "opposite_cwt_ratio_limit": 1.5,
        "adjacent_cwt_ratio_limit": 3.0,
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
