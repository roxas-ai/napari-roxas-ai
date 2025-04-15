import numpy as np
from PIL import Image


def save_image(image: np.ndarray, path: str, rescale: bool = False) -> str:
    """
    Save the image to a file.

    Parameters
    ----------
    image : np.ndarray
        The image to save.
    path : str
        The path to save the image to.

    Returns
    -------
    str
        The path to the saved image.
    """
    # Rescale the image to 0-255 uint8 if required
    if rescale:
        if np.max:
            # Normalize the image to the range [0, 255]
            image = (
                (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
            ).astype(np.uint8)
        else:  # If the image has a single value, set it to zero
            image = np.zeros_like(image, dtype=np.uint8)

    # Create a PIL Image from the numpy array
    pil_image = Image.fromarray(image)

    # Save the image to the specified path
    pil_image.save(path)

    return path
