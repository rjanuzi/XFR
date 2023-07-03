from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATASET_LFW = "lfw"
DATASET_VGGFACE2 = "vggface2"
__DATASETS = [DATASET_LFW, DATASET_VGGFACE2]

DATASET_KIND_RAW = "raw"
DATASET_KIND_ALIGNED = "aligned"
DATASET_KIND_SEG_IMG = "segmented"
DATASET_KIND_SEG_MAP = "segmentation_maps"
__DATASET_KINDS = [
    DATASET_KIND_RAW,
    DATASET_KIND_ALIGNED,
    DATASET_KIND_SEG_IMG,
    DATASET_KIND_SEG_MAP,
]

# Create Ensure all folders
__ROOT_FOLDER = Path(__file__).parent
for dataset in __DATASETS:
    for kind in __DATASET_KINDS:
        __ROOT_FOLDER.joinpath(dataset, kind).mkdir(parents=True, exist_ok=True)

# Dataset index file
___DATASET_IDX = __ROOT_FOLDER.joinpath("dataset_index.pickle")


def get_dataset_index(recreate: bool = False) -> pd.DataFrame:
    """
    Generate the dataset index reference to improve image recovering performance.

    In:
        recreate: If True, recreate the index file.
    Out:
        Pandas DataFrame with the all dataset images references.
    """
    if recreate or not ___DATASET_IDX.exists():
        print("Creating the datasets index file, this can take a couple of minutes...")
        entries = []
        for dataset in __DATASETS:
            for kind in __DATASET_KINDS:
                tmp_folder = __ROOT_FOLDER.joinpath(dataset, kind)
                for img_path in tmp_folder.iterdir():
                    if img_path.is_file():
                        entries.append(
                            {
                                "dataset": dataset,
                                "kind": kind,
                                "person_name": img_path.stem,
                                "extension": img_path.suffix,
                                "img_path": str(img_path),
                            }
                        )

        dataset_idx = pd.DataFrame(entries)

        # Save the index file
        dataset_idx.to_pickle(___DATASET_IDX)

        print("Datasets index file created!")

        return dataset_idx
    else:
        return pd.read_pickle(___DATASET_IDX)


def gen_dataset_index(kind: str = None) -> pd.DataFrame:
    """
    Generate the dataset entries.
    Out:
        Pandas pd.DataFrame like:
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

    entries = pd.DataFrame(entries)

    if kind is not None:
        return entries[entries["kind"] == kind]
    else:
        return entries


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


def ls_imgs_paths(dataset: str, kind: int = DATASET_KIND_RAW) -> list:
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
