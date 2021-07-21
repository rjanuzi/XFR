import numpy as np
from PIL import Image


def _normalize_img(img_data: np.ndarray) -> np.ndarray:
    """
    Normalize image data between [-1, 1].
    """
    return (img_data / 127.5) - 1.0


def show_img(img_data: np.ndarray) -> None:
    """
    Show the ndarray image data (normilized or not).
    """
    # If the image have some value bigger than 1.0, its not normalized
    img_data = img_data if img_data.max() > 1.0 else (img_data + 1) * 127.5

    im = Image.fromarray(obj=img_data.astype("int8"), mode="RGB")
    im.show()


def save_img(img_data: np.ndarray, path: str) -> None:
    """
    Save the ndarray image data (normilized or not).
    """
    # If the image have some value bigger than 1.0, its not normalized
    img_data = img_data if img_data.max() > 1.0 else (img_data + 1) * 127.5

    # If we have a 3 channel image, we need to use de mode 'RGB'
    im = Image.fromarray(obj=img_data.astype("int8"), mode="RGB")
    im.save(path)


def load_img(path, target_size: tuple = (1024, 1024)) -> np.ndarray:
    """
    Load the image from the given path.
    """
    img = Image.open(path)
    if target_size:
        img = img.resize(size=target_size)

    return np.array(img, dtype="float32")
