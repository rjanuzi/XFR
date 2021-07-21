from pathlib import Path

import numpy as np
from numpy.lib.function_base import iterable
from pandas import DataFrame
from PIL import Image


def _normalize_img(img_data: np.ndarray) -> np.ndarray:
    """
    Normalize image data between [-1, 1].
    """
    return (img_data / 127.5) - 1.0


def lookup_imgs(root_folder=None) -> DataFrame:
    """
    Generate the dataset entries as a dict.
    Out:
        Pandas DataFrame like:
        person_name | pose     | img_path
        rjanuzi     | leftside | './dataset/rjanuzi/leftside.jpg'
        rjanuzi     | normal   | './dataset/rjanuzi/normal.jpg'
        ffaria      | smile    | './dataset/rjanuzi/smile.jpg'
        ...
    """
    entries = []
    root_folder = Path(root_folder) if root_folder else Path(__file__).parent
    for person_dir in root_folder.iterdir():
        if person_dir.is_dir():
            try:
                poses_files = [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.jpg")
                ]

                poses_files += [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.jpeg")
                ]
            except NotADirectoryError as e:
                print("[ERROR]: dataset.__init__.py - Invalid dataset structure")
                raise e

            for pose, img_path in poses_files:
                entries.append(
                    {"person_name": person_dir.name, "pose": pose, "img_path": img_path}
                )

    return DataFrame(entries)


def read_imgs(
    img_paths: list = None,
    target_size: tuple = (1024, 1024),
    normalize: bool = True,
) -> iterable(np.ndarray):
    """
    Read a list of JPEG images.
    """

    # If no paths are provided, use all the available dataset
    if img_paths == None:
        img_paths = lookup_imgs()["img_path"]

    for img_path in img_paths:
        img = Image.open(img_path)
        if target_size:
            img = img.resize(size=target_size)

        if normalize:
            yield _normalize_img(np.array(img, dtype="float32"))
        else:
            yield np.array(img, dtype="float32")
