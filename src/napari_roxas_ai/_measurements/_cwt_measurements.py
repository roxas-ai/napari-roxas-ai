"""
Original code by github user triyan-b https://github.com/triyan-b
"""

from enum import Enum
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


class WallClassificationMethod(Enum):
    NEAREST_MIN_AREA_RECT_WALL = (1,)
    MIN_AREA_RECT_DIAGONALS = 2


cells = {}
pixels_per_um = 2.2675
wall_classification_method = (
    WallClassificationMethod.NEAREST_MIN_AREA_RECT_WALL
)


def measure_cwt(path):
    print(f"measuring cell wall thickness for file {path}")

    # index = 5
    # img_files = sorted(glob.glob("CropsForTesting/*.png"))[index : index + 1]
    # Angles start from the positive horizontal axis and go clockwise
    tang_angle = 0  # This is the angle measured during ring analysis
    rad_angle = tang_angle - 90

    path = Path(path)
    print(f"Reading image {path.name}")

    # Read and preprocess images
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    # img_orig = cv2.cvtColor(
    #    cv2.imread(path.replace("png", "jpg")), cv2.COLOR_BGR2GRAY
    # )
    print("Read image with shape", img.shape)

    # Skeletonise the binary image to determine cell wall "centres"
    skeleton = cv2.ximgproc.thinning(np.uint8(cv2.bitwise_not(img)))

    # Distance transform
    dist_orig = cv2.distanceTransform(
        np.uint8(cv2.bitwise_not(img)), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    # dist = cv2.normalize(dist_orig, 0, 1.0, cv2.NORM_MINMAX)

    # Erode slightly to enable contour detection of cell walls
    eroded = cv2.erode(cv2.bitwise_not(skeleton), np.ones((3, 3)))

    # Finding initial cell wall contours
    print("Finding contours")
    cw_contours, hierarchy = cv2.findContours(
        eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(cw_contours)} initial CW contours")

    # Finding Lumen contours
    lumen_contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"Found {len(lumen_contours)} lumen contours")

    # Draw contours
    contours_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # contours_img = cv2.drawContours(contours_img, cw_contours, -1, (0,255,75), 1)
    # contours_img = cv2.drawContours(contours_img, lumen_contours, -1, (255,0,75), 1)

    # Locate CW centroids and detect "watershed" contours using a flood fill mechanism
    mask = np.zeros(
        (skeleton.shape[0] + 2, skeleton.shape[1] + 2), dtype=np.uint8
    )
    cwws_contours = []
    for i, c in enumerate(cw_contours[:]):
        # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
        M = cv2.moments(c)
        if (m00 := M["m00"]) != 0:
            cx = int(M["m10"] / m00)
            cy = int(M["m01"] / m00)
            # Perform flood fill
            flood = skeleton.copy()
            cv2.floodFill(flood, mask, (cx, cy), 127)
            ret, flood = cv2.threshold(flood, 128, 255, cv2.THRESH_TOZERO_INV)
            dilated = cv2.dilate(flood, np.ones((3, 3)))
            # Find single contour
            contours, hierarchy = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if len(contours) == 1:
                cwws_contours.append(contours[0])
            else:
                print(f"No unique contour found at index {i}")
        else:
            print(i, "CW centroid not found: contour not segmented properly")
    print(f"Found {len(cwws_contours)} CW (WS) contours")

    # Lumen measurements - this is also where the cells are given an ID
    for i, c in enumerate(lumen_contours[:]):
        M = cv2.moments(c)
        if (m00 := M["m00"]) != 0:
            cx = int(M["m10"] / m00)
            cy = int(M["m01"] / m00)
            cv2.circle(contours_img, (cx, cy), 1, (0, 0, 255), -1)
            cv2.putText(
                contours_img,
                str(i),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cells[i] = {
                "centroid": (cx, cy),
                "lumen_area": m00 / (pixels_per_um * pixels_per_um),
                "lumen_peri": cv2.arcLength(c, True) / pixels_per_um,
            }
        else:
            print(
                i,
                "Lumen centroid not found: contour not segmented properly near",
                np.mean(c, axis=0)[0],
            )
            continue
            # print(c)
        try:
            ellipse = cv2.fitEllipse(c)
            center, axes, angle = ellipse
            # print(centre, axes, angle)
            center_x, center_y = center
            w, h = axes
            if w >= h:
                a, b = w / 2, h / 2
                aoma_rad = angle - rad_angle
            else:
                a, b = h / 2, w / 2
                aoma_rad = angle - rad_angle - 90
            aoma_tang = aoma_rad + 90

            # https://en.wikipedia.org/wiki/Ellipse#Polar_form_relative_to_center
            lumen_diam_rad = (
                a
                * b
                / (
                    np.linalg.norm(
                        [
                            b * np.cos(np.deg2rad(aoma_rad)),
                            a * np.sin(np.deg2rad(aoma_rad)),
                        ]
                    )
                )
            )
            lumen_diam_tang = (
                a
                * b
                / (
                    np.linalg.norm(
                        [
                            b * np.cos(np.deg2rad(aoma_tang)),
                            a * np.sin(np.deg2rad(aoma_tang)),
                        ]
                    )
                )
            )
            cells[i].update(
                {
                    "lumen_aoma_rad": aoma_rad,
                    "lumen_diam_rad": lumen_diam_rad / pixels_per_um,
                    "lumen_diam_tang": lumen_diam_tang / pixels_per_um,
                }
            )

            cv2.ellipse(contours_img, ellipse, (0, 255, 0), 1)
            # cv2.line(contours_img, (round(center_x), round(center_y)), (round(center_x + b*np.cos(aoma_rad/180 * np.pi - np.pi/2)), round(center_y + b*np.sin(aoma_rad/180 * np.pi - np.pi/2))), (0, 255, 0))
            # cv2.ellipse(contours_img, (round(center_x), round(center_y)), (20, 20), 0, 0, angle, (255, 0, 0), 1)
        except cv2.error as e:
            print(f"OpenCV error during ellipse fitting or drawing: {e}")
        except ValueError as e:
            print(f"Value error during mathematical calculations: {e}")
        except TypeError as e:
            print(f"Type error during calculations: {e}")
        except AttributeError as e:
            print(f"Attribute error while updating cells: {e}")
        except IndexError as e:
            print(f"Index error while accessing cells: {e}")
    print(f"Found {len(cells)} lumen centroids")

    for i, c in enumerate(cwws_contours[:]):
        # First, find the cell corresponding to this cell wall. Assumption: lumen centroid is within CW WS contour
        cell_id = next(
            (
                idx
                for idx, cell in cells.items()
                if cv2.pointPolygonTest(c, cell["centroid"], measureDist=False)
                >= 0
            ),
            None,
        )
        if cell_id is None:
            print("Could not find this cell near", np.mean(c, axis=0)[0])
            continue

        M = cv2.moments(c)
        if (m00 := M["m00"]) != 0:
            cells[cell_id].update(
                {
                    "cell_area": m00 / (pixels_per_um * pixels_per_um),
                    "cw_peri": cv2.arcLength(c, True) / pixels_per_um,
                }
            )
        else:
            print(
                i, "CW (WS) centroid not found: contour not segmented properly"
            )

        colors = {
            "North": (100, 0, 0),
            "East": (0, 100, 0),
            "South": (100, 0, 100),
            "West": (100, 100, 0),
        }
        rect = cv2.minAreaRect(c)
        box = np.intp(cv2.boxPoints(rect))

        # Identify corners based on sums and differences
        sums = box[:, 0] + box[:, 1]  # x + y
        diffs = box[:, 1] - box[:, 0]  # y - x

        top_left = box[np.argmin(sums)]  # Smallest sum
        bottom_right = box[np.argmax(sums)]  # Largest sum
        top_right = box[np.argmin(diffs)]  # Smallest difference
        bottom_left = box[np.argmax(diffs)]  # Largest difference
        # cv2.circle(contours_img, top_left, 1, (255, 255, 0), 2)

        # Assign contour points to the nearest wall
        classified_walls = {"North": [], "East": [], "South": [], "West": []}
        for point_index, point in enumerate(c):
            point = point[0]  # Extract (x, y)

            if (
                wall_classification_method
                == WallClassificationMethod.NEAREST_MIN_AREA_RECT_WALL
            ):
                # Calculate perpendicular distances from point to walls
                distances = {
                    "North": np.linalg.norm(
                        np.cross(top_right - top_left, top_left - point)
                    )
                    / np.linalg.norm(top_right - top_left),
                    "South": np.linalg.norm(
                        np.cross(
                            bottom_right - bottom_left, bottom_left - point
                        )
                    )
                    / np.linalg.norm(bottom_right - bottom_left),
                    "East": np.linalg.norm(
                        np.cross(bottom_right - top_right, top_right - point)
                    )
                    / np.linalg.norm(bottom_right - top_right),
                    "West": np.linalg.norm(
                        np.cross(bottom_left - top_left, top_left - point)
                    )
                    / np.linalg.norm(bottom_left - top_left),
                }
                closest_wall = min(distances, key=distances.get)

            elif (
                wall_classification_method
                == WallClassificationMethod.MIN_AREA_RECT_DIAGONALS
            ):
                is_above_main_diagonal = (
                    np.cross(bottom_right - top_left, point - top_left).item()
                    <= 0
                )
                is_above_secondary_diagonal = (
                    np.cross(
                        top_right - bottom_left, point - bottom_left
                    ).item()
                    <= 0
                )
                if is_above_main_diagonal and is_above_secondary_diagonal:
                    closest_wall = "North"
                elif (
                    is_above_main_diagonal and not is_above_secondary_diagonal
                ):
                    closest_wall = "East"
                elif (
                    not is_above_main_diagonal and is_above_secondary_diagonal
                ):
                    closest_wall = "West"
                else:
                    closest_wall = "South"

            assert closest_wall is not None
            classified_walls[closest_wall].append((point_index, tuple(point)))

        # For each classified wall, maybe reorder the partial contours so it is continuous
        for wall, points in classified_walls.items():
            ids = np.array([point[0] for point in points])
            diffs = np.diff(ids)
            if (
                len(diffs) != 0
                and len(skips := (np.where(diffs >= 3)[0])) == 1
            ):
                skip_index = skips[0] + 1
                reordered_points = points[skip_index:] + points[:skip_index]
                # print(skip_index, len(c), ids, [point[1] for point in reordered_points])
            else:
                reordered_points = points
            classified_walls[wall] = [point[1] for point in reordered_points]

        # Perform the integrals
        for wall, points in classified_walls.items():
            # Visualise wall
            for point in points:
                contours_img[point[::-1]] = colors[wall]
                colors[wall] = tuple(
                    [min(255, x + 3) if x != 0 else x for x in colors[wall]]
                )  # gradient to verify continuity

            integration_interval_coeff = 0.75
            integration_margin_coeff = (1 - integration_interval_coeff) / 2
            lower_bound = int(integration_margin_coeff * len(points))
            upper_bound = int((1 - integration_margin_coeff) * len(points))
            assert lower_bound <= upper_bound
            integration_acc = 0
            for point in points[lower_bound:upper_bound]:
                integration_acc += dist_orig[point[::-1]]
                # print(point)
                # contours_img[point[::-1]] = (0, 0, 255) # Visualise integral path
            integration_acc /= upper_bound - lower_bound + 1
            cells[cell_id][f"CWT_{wall}"] = integration_acc / pixels_per_um

        # Draw the min area rectangle
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # color = (0, 0, 255)
        # cv2.drawContours(contours_img, [box], 0, color, 1)
        # cv2.line(contours_img, box[0], box[2], color, 1)

    print(list(cells.values())[20:25])

    # Visualization
    combined = cv2.addWeighted(img, 0.4, skeleton, 0.6, 0)  # img_orig
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.title("Original Binary Image")
    plt.imshow(img, cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("Segmented Walls")
    plt.imshow(skeleton, cmap="binary_r")

    plt.subplot(1, 4, 3)
    plt.title("Segmented Walls (overlay)")
    plt.imshow(combined, cmap="binary_r")

    plt.subplot(1, 4, 4)
    plt.title("Result")
    plt.imshow(
        cv2.addWeighted(
            contours_img,
            0.7,
            cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR),
            0.3,
            0,
        )
    )

    plt.tight_layout()
    # plt.show()
    # plt.savefig(f"{path.stem}_CW_Lumen_Contours.jpg")
