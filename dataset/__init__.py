from pathlib import Path

import numpy as np
from pandas import DataFrame
from PIL import Image

VGGFACE2_FOLDER = Path(Path(__file__).parent, "vggface2")
LFW_FOLDER = Path(Path(__file__).parent, "lfw")

DATASETS = ["vggface2", "lfw"]

DATASET_KIND_RAW = 1
DATASET_KIND_ALIGNED = 2
DATASET_KIND_SEG_IMG = 3
DATASET_KIND_SEG_MAP = 4

DATASET_FOLDER_MAP = {
    "vggface2": VGGFACE2_FOLDER,
    "lfw": LFW_FOLDER,
}

DATASET_KIND_MAP = {
    DATASET_KIND_RAW: "raw_images",
    DATASET_KIND_ALIGNED: "aligned_images",
    DATASET_KIND_SEG_IMG: "seg_images",
    DATASET_KIND_SEG_MAP: "seg_maps",
}

DATASET_KIND_STR = {
    DATASET_KIND_RAW: "raw",
    DATASET_KIND_ALIGNED: "aligned",
    DATASET_KIND_SEG_IMG: "seg_img",
    DATASET_KIND_SEG_MAP: "seg_map",
}


def get_file_path(
    name: str, dataset: str, dataset_kind: int, file_extension: str = ".png"
) -> Path:
    try:
        dataset_folder = DATASET_FOLDER_MAP[dataset]
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")

    try:
        return dataset_folder.joinpath(
            DATASET_KIND_MAP[dataset_kind], name + file_extension
        )
    except KeyError:
        raise ValueError(f"Invalid folder_kind: {dataset_kind}")


def read_aligned(name: str, dataset: str) -> Image:
    return Image.open(
        get_file_path(
            name=name,
            dataset=dataset,
            dataset_kind=DATASET_KIND_ALIGNED,
            file_extension=".png",
        )
    ).convert("RGB")


def gen_dataset_index(kind: str = None) -> DataFrame:
    """
    Generate the dataset entries.
    Out:
        Pandas DataFrame like:
        name        |   Kind    | extension | img_path
        rjanuzi     |   raw     | 'jpg      | './dataset/raw/rjanuzi.jpg'
        rjanuzi     |   aligned | 'png'     | './dataset/aligned/rjanuzi.png'
        rjanuzi     |   mask    | 'png'     | './dataset/mask/rjanuzi.png'
        rjanuzi     |   latent  | 'npy'     | './dataset/latents/rjanuzi.npy'
        ...
    """
    entries = []
    for dataset in DATASETS:
        for kind_idx, folder_appendix in DATASET_KIND_MAP.items():
            folder = DATASET_FOLDER_MAP[dataset].joinpath(folder_appendix)
            kind_str = DATASET_KIND_STR[kind_idx]
            for img_path in folder.iterdir():
                if img_path.is_file():
                    entries.append(
                        {
                            "dataset": dataset,
                            "name": img_path.stem,
                            "kind": kind_str,
                            "extension": img_path.suffix,
                            "img_path": str(img_path),
                        }
                    )

    entries = DataFrame(entries)

    if kind is not None:
        return entries[entries["kind"] == kind]
    else:
        return entries


def ls_imgs_paths(kind: int = DATASET_KIND_RAW) -> list:
    """
    List the images paths.
    Out: A list with the raw images paths.
    """
    entries = gen_dataset_index()
    return entries.loc[entries["kind"] == DATASET_KIND_STR[kind], "img_path"].tolist()


def ls_imgs_names(kind: int = DATASET_KIND_RAW) -> list:
    """
    List the images paths.
    Out: A list with the raw images paths.
    """
    entries = gen_dataset_index()
    return entries.loc[entries["kind"] == DATASET_KIND_STR[kind], "name"].tolist()
