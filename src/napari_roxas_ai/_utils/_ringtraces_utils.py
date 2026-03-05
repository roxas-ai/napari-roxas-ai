import struct

import cv2
import numpy as np
import pandas as pd
import PIL.Image
import PIL.ImageFile

PIL.Image.MAX_IMAGE_PIXELS = None
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = False


def get_angular_step_size_xls(path):
    data = pd.read_excel(path, sheet_name="ROXAS settings")
    data.rename(
        columns=dict(zip(data.columns, list(range(len(data.columns))))),
        inplace=True,
        errors="raise",
    )
    step_size = float(data.iloc[116][1])  # round to 3 steps after comma
    return round(step_size, 3)


def get_angular_step_size_settings(path):
    settings = pd.read_csv(path, sep="\t", encoding="unicode_escape")
    step_size = float(settings["SETTING"][117])
    return round(step_size, 3)


def read_cal_file(path):
    """
    Reads the cal file into a dictionary with keys Origin, PixPerUnit, AngleOffset, SystemCal, RefernceCal
    """
    cal_dict = pd.read_csv(path, sep="=", encoding="unicode_escape").to_dict()[
        "[SPATIAL]"
    ]
    cal_dict["Origin"] = [float(s) for s in cal_dict["Origin"].split(",")]
    cal_dict["PixPerUnit"] = [
        float(s) for s in cal_dict["PixPerUnit"].split(",")
    ]
    cal_dict["AngleOffset"] = [
        float(s) for s in cal_dict["AngleOffset"].split(",")
    ]
    cal_dict["SystemCal"] = int(cal_dict["SystemCal"])
    cal_dict["ReferenceCal"] = int(cal_dict["ReferenceCal"])
    return cal_dict


def read_ring_traces(path):
    with open(path, "rb") as f:
        _: int = int.from_bytes(f.read(2), byteorder="little")  # 2 or 4 bytes
        long_byte_size: int = 4
        lower_limits_first: int = int.from_bytes(
            f.read(long_byte_size), byteorder="little"
        )
        lower_limits_second: int = int.from_bytes(
            f.read(long_byte_size), byteorder="little"
        )
        upper_limits_first: int = int.from_bytes(
            f.read(long_byte_size), byteorder="little"
        )
        upper_limits_second: int = int.from_bytes(
            f.read(long_byte_size), byteorder="little"
        )
        # dim1 = upper_limits_first - lower_limits_first + 1
        # dim2 = upper_limits_second - lower_limits_second + 1
        data = f.read()
        data = np.array(list(struct.iter_unpack("<f", data))).reshape(
            (upper_limits_first - lower_limits_first + 1, -1), order="F"
        )
    return (
        data,
        (lower_limits_second, upper_limits_second),
        (upper_limits_first, upper_limits_second),
    )


def paths_to_image_coordinates(
    ringtraces_file, cal_file, xls_file=None, settings_file=None
):
    assert (
        xls_file is not None or settings_file is not None
    ), "settings or xls file needs to be not None"
    if xls_file is not None:  # This can also be done by extension of the file
        angular_step_size = get_angular_step_size_xls(xls_file)
    if settings_file is not None:
        angular_step_size = get_angular_step_size_settings(settings_file)
    cal_dict = read_cal_file(cal_file)
    pix_per_unit, origin = cal_dict["PixPerUnit"], cal_dict["Origin"]
    assert (
        pix_per_unit[0] == pix_per_unit[1]
    ), "Different pix per unit in x and y need to be handled"
    pix_per_unit = pix_per_unit[0]
    ring_data = read_ring_traces(ringtraces_file)
    img_coord = ring_traces_to_image_coordinates(
        ring_data, pix_per_unit, origin, angular_step_size
    )
    return img_coord


def ring_traces_to_image_coordinates(
    ring_data, spatial_resolution, origin, angle_interval
):
    ring_data, angle_limits, _ = ring_data
    x_origin, y_origin = origin
    angles = (
        (
            (np.arange(angle_limits[0], angle_limits[1] + 1) * angle_interval)
            - 180
        )
        / 180
        * np.pi
    )  # use linspace here
    np.linspace(start=0, stop=100, num=5)
    lines = []
    for i in range(ring_data.shape[0]):
        ring = ring_data[i]
        if np.sum(ring) == 0:
            continue
        y = (ring * spatial_resolution) * np.cos(angles) + y_origin
        x = -(ring * spatial_resolution) * np.sin(angles) + x_origin
        lines.append(np.stack((x, y)))
    return lines


def get_instance_labels_linear(img_coords, image_shape, draw_lines=False):
    base = np.zeros((image_shape[0], image_shape[1], 3), dtype="uint8") + 255
    h, w, c = base.shape
    for i, xy in enumerate(img_coords[::-1]):
        cv2.fillPoly(
            base,
            pts=[
                np.concatenate(
                    (
                        np.round(xy.T.reshape((-1, 1, 2))),
                        np.array([[[0, 0]]]),
                        np.array([[[w, 0]]]),
                    ),
                    axis=0,
                ).astype(int)
            ],
            color=(len(img_coords) - i, 0, 0),
        )
        if draw_lines:
            cv2.polylines(
                base,
                [np.round(xy.T.reshape((-1, 1, 2))).astype(int)],
                False,
                (0),
                1,
            )
    base = base.astype(int)
    base[base == 255] = -1
    return base[:, :, 0]


def ring_labels_from_roxas(
    image_file,
    image_shape: tuple,
    with_boundary=True,
    return_img_coordinates=False,
):
    """
    Reads the ring traces from ROXAS and returns an instance segmentation mask.
    If with_boundary is True, the boundaries of the rings are drawn as lines in the mask.
    If return_img_coordinates is True, the image coordinates of the ring traces are returned instead of the instance segmentation mask.

    image_shape is expected to be a tuple of (height, width) of the image.
    This is needed to create the instance segmentation mask with the correct shape.
    If return_img_coordinates is True, this parameter is ignored.
    """
    trace_file = image_file.replace(".jpg", "_RingTraces.txt")
    cal_file = image_file.replace(".jpg", ".cal")
    settings_file = image_file.replace(".jpg", "_ROXAS_Settings.txt")
    img_coords = paths_to_image_coordinates(
        trace_file, cal_file, xls_file=None, settings_file=settings_file
    )
    if return_img_coordinates:
        return img_coords
    if with_boundary:
        return get_instance_labels_linear(
            img_coords, image_shape, draw_lines=True
        )
    else:
        return get_instance_labels_linear(
            img_coords, image_shape, draw_lines=False
        )
