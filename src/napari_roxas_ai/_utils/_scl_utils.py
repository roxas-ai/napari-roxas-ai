import numpy as np


def read_scl(file_path: str) -> np.ndarray:
    """
    Decodes the Image-Pro Plus SCL file format
    :param file_path: path to the SCL file
    :return: np.array with shape (w,h) with values 0 and 255
    """
    with open(file_path, "rb") as f:
        header = f.read(14).decode("utf-8")
        # 14 bytes 'IpWin Scanlist', 2 dim, 2 firstindex, 2 firstindex, 2 lastindex, 2 lastindex, 4 unknown
        assert (
            header == "IpWin Scanlist"
        ), "Wrong file format expect SCL from Image-Pro Plus in Scanlist file format"
        dim = int.from_bytes(f.read(2), byteorder="little", signed=False)
        assert dim == 2, "Expected two dimensions"
        x_min_ind = int.from_bytes(
            f.read(2), byteorder="little", signed=False
        )  # x = width
        assert x_min_ind == 0, "Expected to start with 0"
        y_min_ind = int.from_bytes(
            f.read(2), byteorder="little", signed=False
        )  # y = heigth
        assert y_min_ind == 0, "Expected to start with 0"
        x_max_ind = int.from_bytes(f.read(2), byteorder="little", signed=False)
        y_max_ind = int.from_bytes(f.read(2), byteorder="little", signed=False)
        img = np.zeros((y_max_ind + 1, x_max_ind + 1)).astype(np.uint8)
        _ = f.read(4)  # final unused header bytes
        cells_per_row = np.fromfile(
            f, dtype=np.short, count=y_max_ind + 1
        )  # number of instances per row
        for i, y in enumerate(list(cells_per_row)):
            for _j in range(y):
                start = int.from_bytes(
                    f.read(2), byteorder="little", signed=False
                )
                stop = int.from_bytes(
                    f.read(2), byteorder="little", signed=False
                )
                img[i, start : stop + 1] = 255
    return img


def write_scl(scl_array, file_path):
    """
    Converts numpy array to SCL file format

    :param scl_array: Binary image array with values 0 and 255
    :param file_path: Path where the file should be written
    :return: Nothing as it writes directly to file
    """

    scl_array /= 255  # convert to binary
    fwd_diff = np.diff(scl_array)
    horizontal_length = scl_array.shape[1] - 1
    final_result = np.zeros(0)
    cells_per_row = np.zeros(scl_array.shape[0])

    line_results = []

    for i in range(scl_array.shape[0]):
        up = np.argwhere(fwd_diff[i] == 1).reshape(-1)
        down = np.argwhere(fwd_diff[i] == -1).reshape(-1)
        if scl_array[i, 0] == 1:
            up = np.concatenate((np.zeros(1), up))  # account start at 0
        if scl_array[i, -1] == 1:
            down = np.concatenate(
                (down, np.ones(1) * horizontal_length)
            )  # account for end in last pixel
        result = np.stack((up + 1, down)).T.flatten()
        line_results.append(result)
        cells_per_row[i] = len(up)

    final_resulst = np.concatenate(line_results)
    max_idx2, max_idx1 = scl_array.shape
    with open(file_path, "wb") as f:
        f.write(bytes("IpWin Scanlist", encoding="utf-8"))
        f.write((2).to_bytes(2, byteorder="little", signed=False))  # dim
        f.write(
            (0).to_bytes(2, byteorder="little", signed=False)
        )  # firstindex
        f.write(
            (0).to_bytes(2, byteorder="little", signed=False)
        )  # firstindex
        f.write(
            (max_idx1 - 1).to_bytes(2, byteorder="little", signed=False)
        )  # lastindex
        f.write(
            (max_idx2 - 1).to_bytes(2, byteorder="little", signed=False)
        )  # lastindex
        f.write(
            (65536).to_bytes(4, byteorder="little", signed=False)
        )  # unknown
        cells_per_row.astype(np.short).tofile(f)
        final_resulst.astype(np.short).tofile(f)
