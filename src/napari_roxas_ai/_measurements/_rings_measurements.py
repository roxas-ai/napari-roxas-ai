"""
Original code by github user triyan-b https://github.com/triyan-b
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Helper function to check intersection between two lines
def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute the denominator
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Parallel lines

    # Compute intersection point
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom

    # Check if the intersection is within both line segments
    if (
        min(x1, x2) <= px <= max(x1, x2)
        and min(y1, y2) <= py <= max(y1, y2)
        and min(x3, x4) <= px <= max(x3, x4)
        and min(y3, y4) <= py <= max(y3, y4)
    ):
        return (px, py)

    return None  # Intersection not within the line segments


def measure_rings(path):
    print(f"measuring rings for file {path}")

    plt.figure(figsize=(12, 6))

    # Read and preprocess images
    path = Path(path)
    print(f"Reading image {path.name}")
    # img = np.load(path).astype("float32")
    with Image.open(path) as img:
        img = np.array(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    plt.subplot(1, 3, 1)
    plt.imshow(img)

    contours_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print("Read image with shape", img.shape)

    plt.subplot(1, 3, 2)
    plt.imshow(contours_img)

    # Detect ring borders (Edge detection followed by contour detection)
    img = cv2.Canny(img, 50, 150)

    plt.subplot(1, 3, 3)
    plt.imshow(img, cmap="gray")
    # plt.show()

    contours, _ = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    print(f"Found {len(contours)} ring borders")
    contours = tuple(np.unique(c, axis=0) for c in contours)

    ring_borders = []
    rings = []

    # Ring border measurements
    sample_angle = 0
    for border in contours:
        # Draw border
        for point in border:
            contours_img[tuple(point[0])[::-1]] = (255, 0, 0)
        # Best fit line and angle
        x = border[:, 0, 0]
        y = border[:, 0, 1]
        A = np.vstack([x, np.ones(len(x))]).T  # [x, 1] matrix for linear fit
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        ring_angle = np.rad2deg(np.arctan(m))
        # print(m, c, ring_angle)
        sample_angle += ring_angle
        x0, x1 = 0, img.shape[1] - 1
        y0, y1 = round(m * x0 + c), round(m * x1 + c)
        ring_borders.append(
            {
                "contour": border,
                "m": m,
                "c": c,
                "angle": ring_angle,
                "p0": np.array([x0, y0]),
                "p1": np.array([x1, y1]),
            }
        )
        cv2.line(contours_img, (x0, y0), (x1, y1), (0, 255, 0), 3)

    sample_angle /= len(contours)
    print(f"Sample angle: {sample_angle}")
    # Assuming ring borders do not cross, sort them from top to bottom
    ring_borders.sort(key=lambda ring: ring["c"])

    # Ring measurements
    for i in range(
        len(ring_borders) - 1
    ):  # TODO process rings independantly using multiprocessing
        border_a, border_b = ring_borders[i], ring_borders[i + 1]
        upper_contour = border_a["contour"]
        lower_contour = border_b["contour"]
        avg_angle = np.deg2rad((border_a["angle"] + border_b["angle"]) / 2)
        avg_gradient = (border_a["m"] + border_b["m"]) / 2
        vertical_ring_width = 0
        angled_ring_width = 0
        angled_intersections = 0
        ring_img = np.zeros_like(img).astype("uint8")

        for point in upper_contour:
            ring_img[tuple(point[0])[::-1]] = 255
        for point in lower_contour:
            ring_img[tuple(point[0])[::-1]] = 255

        for x in range(img.shape[1]):
            # Vertical ring width calculation
            y_a = max(
                [point[0, 1] for point in upper_contour if point[0, 0] == x]
            )
            y_b = min(
                [point[0, 1] for point in lower_contour if point[0, 0] == x]
            )  # TODO fix min() iterable argument is empty
            vertical_ring_width += y_b - y_a

            # Angled ring width calculation (average angle)
            # Consider only every Nth point for the angled calculation (otherwise it's slow)
            if x % 20 != 0:
                continue

            x1 = x + img.shape[0] * np.sin(-avg_angle) / np.cos(-avg_angle)
            y1 = y_a + img.shape[0]
            line = (x, y_a, x1, y1)
            intersections = []
            for j in range(
                x,
                (len(lower_contour) - 1) if avg_angle < 0 else -1,
                1 if avg_angle < 0 else -1,
            ):  # Find intersections by iterating through contour segments with an inital x coordinate guess
                segment = (
                    lower_contour[j, 0, 0],
                    lower_contour[j, 0, 1],
                    lower_contour[
                        j + 1, 0, 0
                    ],  # TODO fix index 5001 is out of bounds for axis 0 with size 5001
                    lower_contour[j + 1, 0, 1],
                )
                if intersection := line_intersection(line, segment):
                    intersections.append(intersection)

            # Print results
            if len(intersections) > 0:
                angled_intersections += 1
                angled_ring_width += np.linalg.norm(
                    np.array([intersections[0]]) - np.array([x, y_a])
                )

        vertical_ring_width /= img.shape[1]
        angled_ring_width /= angled_intersections  # TODO prevent zero division
        print(
            f"Ring {i + 1} has average vertical width {vertical_ring_width} pixels and average angled width {angled_ring_width} pixels"
        )
        rings.append(
            {
                "vert_width": vertical_ring_width,
                "angled_width": angled_ring_width,
                "avg_angle_rad": avg_angle,
                "avg_gradient": avg_gradient,
                "ring_img": ring_img,
            }
        )

    # Centroid measurements
    centroids = np.array(
        [[700, 1140], [370, 970], [100, 540], [300, 160], [795, 820]]
    )
    centroid_measurements = []
    for centroid in centroids:
        contours_img[tuple(centroid)[::-1]] = (0, 255, 0)
        # ring_id = argmax (d/di isCentroidInExtendedRingPolygon(i))
        ring_id = np.diff(
            [0]
            + [
                cv2.pointPolygonTest(
                    np.concatenate(
                        (
                            border["contour"],
                            np.array([[[img.shape[1] - 1, 0]]]),
                            np.array([[[0, 0]]]),
                        )
                    ),
                    (round(centroid[0]), round(centroid[1])),
                    False,
                )
                >= 0
                for border in ring_borders
            ]
            + [1]
        ).argmax()
        print(f"Centroid {centroid} belongs to ring {ring_id}")
        centroid_data = {"centroid": centroid, "ring_id": ring_id}
        if ring_id not in [0, len(ring_borders)]:
            ring = rings[ring_id - 1]
            border_a = ring_borders[ring_id - 1]
            border_b = ring_borders[ring_id]
            vertical_dst = centroid[1] - max(
                [
                    point[0, 1]
                    for point in border_a["contour"]
                    if point[0, 0] == centroid[0]
                ]
            )
            best_fit_dst = (
                np.cross(
                    border_a["p1"] - border_a["p0"], centroid - border_a["p0"]
                )
                / np.linalg.norm(border_a["p1"] - border_a["p0"])
            ).item()
            cv2.line(
                contours_img,
                (centroid[0], centroid[1]),
                (centroid[0], centroid[1] - vertical_dst),
                (0, 255, 0),
                1,
            )
            print(
                f"Centroid is {vertical_dst} pixels vertically below the ring border, {best_fit_dst} pixels away from best fit line"
            )
            centroid_data.update(
                {
                    "upper_vertical_dst": vertical_dst,
                    "upper_best_fit_dst": best_fit_dst,
                }
            )

            # Multiply 2 images to calculate intersections
            line_img = np.zeros_like(img)
            avg_m = ring["avg_gradient"]
            # TODO: Handle case where avg_m is very close to 0!!!
            perp_m = -1 / avg_m
            perp_c = round(centroid[1] - perp_m * centroid[0])
            y1 = round(perp_m * img.shape[1] + perp_c)
            cv2.line(line_img, (0, perp_c), (img.shape[1], y1), 255, 1)
            intersection_img = cv2.bitwise_and(ring["ring_img"], line_img)
            intersections = cv2.findNonZero(intersection_img)
            if len(intersections) != 2:
                print(
                    f"Centroid line does not intersect ring boundaries twice: {intersections}"
                )
            else:
                upper_intersection = min(intersections, key=lambda p: p[0, 1])
                lower_intersection = max(intersections, key=lambda p: p[0, 1])
                upper_angled_dst = np.linalg.norm(
                    centroid - upper_intersection
                )
                lower_angled_dst = np.linalg.norm(
                    centroid - lower_intersection
                )
                ring_width_at_centroid = np.linalg.norm(
                    upper_intersection - lower_intersection
                )
                print(
                    upper_angled_dst, lower_angled_dst, ring_width_at_centroid
                )
                centroid_data.update(
                    {
                        "upper_angled_dst": upper_angled_dst,
                        "lower_angled_dst": lower_angled_dst,
                        "ring_width_at_centroid": ring_width_at_centroid,
                    }
                )
        else:
            print(
                "Centroid does not have either an upper or lower ring border - cannot determine position within ring"
            )

        centroid_measurements.append(centroid_data)

    print(centroid_measurements)
    print(rings)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.title("Original Binary Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("-")
    plt.imshow(img, cmap="binary_r")

    plt.subplot(1, 4, 3)
    plt.title("-")
    plt.imshow(img, cmap="binary_r")

    plt.subplot(1, 4, 4)
    plt.title("Result")
    plt.imshow(contours_img)

    plt.tight_layout()
    # plt.show()
