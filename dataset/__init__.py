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


def lookup_imgs(
    root_folder=None, person_names: list = None, poses: list = None
) -> DataFrame:
    """
    Generate the dataset entries as a dict.
    In:
        root_folder: The folder containing the dataset
        person_names: The list of person names to include, if None, all are include
        poses: The list of poses to include, if None, all are include
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
                    (file.stem, file.absolute())
                    for file in person_dir.glob("[!mask]*.jpg")
                ]

                poses_files += [
                    (file.stem, file.absolute())
                    for file in person_dir.glob("[!mask]*.jpeg")
                ]
            except NotADirectoryError as e:
                print("[ERROR]: dataset.__init__.py - Invalid dataset structure")
                raise e

            for pose, img_path in poses_files:
                entries.append(
                    {"person_name": person_dir.name, "pose": pose, "img_path": img_path}
                )

    entries = DataFrame(entries)

    if person_names:
        entries = entries.loc[entries["person_name"].isin(person_names)]

    if poses:
        entries = entries.loc[entries["pose"].isin(poses)]

    return entries


def read_img(
    img_path: str = None,
    target_size: tuple = (1024, 1024),
    normalize: bool = True,
) -> np.ndarray:
    """
    Read a JPEG image.
    """

    img = Image.open(img_path)

    if target_size:
        img = img.resize(size=target_size)

    if normalize:
        return _normalize_img(np.array(img, dtype="float32"))
    else:
        return np.array(img, dtype="float32")


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

    # Yields each img data
    for img_path in img_paths:
        yield read_img(img_path=img_path, target_size=target_size, normalize=normalize)


def lookup_latents(
    root_folder=None, person_names: list = None, poses: list = None
) -> DataFrame:
    """
    Generate the dataset latent entries as a dict.
    In:
        root_folder: The folder containing the dataset
        person_names: The list of person names to include, if None, all are include
        poses: The list of poses to include, if None, all are include
    Out:
        Pandas DataFrame like:
        person_name | pose     | latent_path
        rjanuzi     | leftside | './dataset/rjanuzi/leftside.npy'
        rjanuzi     | normal   | './dataset/rjanuzi/normal.npy'
        ffaria      | smile    | './dataset/rjanuzi/smile.npy'
        ...
    """
    entries = []
    root_folder = Path(root_folder) if root_folder else Path(__file__).parent
    for person_dir in root_folder.iterdir():
        if person_dir.is_dir():
            try:
                poses_files = [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.npy")
                ]
            except NotADirectoryError as e:
                print("[ERROR]: dataset.__init__.py - Invalid dataset structure")
                raise e

            for pose, latent_path in poses_files:
                entries.append(
                    {
                        "person_name": person_dir.name,
                        "pose": pose,
                        "latent_path": latent_path,
                    }
                )

    entries = DataFrame(entries)

    if person_names:
        entries = entries.loc[entries["person_name"].isin(person_names)]

    if poses:
        entries = entries.loc[entries["pose"].isin(poses)]

    return entries


def read_latent(latent_path: str = None) -> np.ndarray:
    """
    Read a latent data.
    """
    return np.load(latent_path)


def read_latents(latents_paths: list = None) -> iterable(np.ndarray):
    """
    Read a list of latents data.
    """

    # If no paths are provided, use all the available dataset
    if latents_paths == None:
        latents_paths = lookup_latents()["latent_path"]

    # Yields each latent data
    for latent_path in latents_paths:
        yield read_latent(latent_path=latent_path)
