from pathlib import Path

import numpy as np
from PIL import Image

PROCESSED_DATASET_FOLDER = Path(Path(__file__).parent, "processed")
PROCESSED_DATASET_FOLDER.mkdir(exist_ok=True)

PROCESSED_DS_FOLDER_KIND_ALIGNED = "aligned"
PROCESSED_DS_FOLDER_KIND_MASK = "mask"
PROCESSED_DS_FOLDER_KIND_LATENT = "latent"
PROCESSED_DS_FOLDER_KINDS = [
    PROCESSED_DS_FOLDER_KIND_ALIGNED,
    PROCESSED_DS_FOLDER_KIND_MASK,
    PROCESSED_DS_FOLDER_KIND_LATENT,
]


def get_folder(person_name: str, folder_kind: str) -> str:
    try:
        assert folder_kind in PROCESSED_DS_FOLDER_KINDS
    except AssertionError:
        raise ValueError(f"Invalid folder_kind: {folder_kind}")
    path = PROCESSED_DATASET_FOLDER.joinpath(person_name, folder_kind)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_path(person_name: str, folder_kind: str, file_name: str) -> Path:
    try:
        assert folder_kind in PROCESSED_DS_FOLDER_KINDS
    except AssertionError:
        raise ValueError(f"Invalid folder_kind: {folder_kind}")

    return Path(PROCESSED_DATASET_FOLDER, person_name, folder_kind, file_name)


def read_file(person_name: str, folder_kind: str, file_name: str) -> bytes:
    return get_file_path(person_name, folder_kind, file_name).read_bytes()


def read_latent(person_name: str) -> np.ndarray:
    return np.load(
        get_file_path(person_name, PROCESSED_DS_FOLDER_KIND_LATENT, "normal.npy")
    )


def read_mask(person_name: str) -> Image:
    return Image.open(
        get_file_path(person_name, PROCESSED_DS_FOLDER_KIND_MASK, "normal.png")
    ).convert(mode="L")
