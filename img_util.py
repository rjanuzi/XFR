import numpy as np
from PIL import Image


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
