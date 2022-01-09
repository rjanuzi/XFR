from pathlib import Path

import numpy as np
from pandas import DataFrame
from PIL import Image

DATASET_RAW_FOLDER = Path(Path(__file__).parent, "raw_images")
DATASET_ALIGNED_FOLDER = Path(Path(__file__).parent, "aligned_images")
DATASET_MASKS_FOLDER = Path(Path(__file__).parent, "masks")
DATASET_LATENTS_FOLDER = Path(Path(__file__).parent, "latents")
DATASET_GENERATED_FOLDER = Path(Path(__file__).parent, "generated_images")
DATASET_MORPH_FOLDER = Path(Path(__file__).parent, "morphs")

DATASET_RAW_FOLDER.mkdir(exist_ok=True, parents=True)
DATASET_ALIGNED_FOLDER.mkdir(exist_ok=True, parents=True)
DATASET_MASKS_FOLDER.mkdir(exist_ok=True, parents=True)
DATASET_LATENTS_FOLDER.mkdir(exist_ok=True, parents=True)
DATASET_GENERATED_FOLDER.mkdir(exist_ok=True, parents=True)
DATASET_MORPH_FOLDER.mkdir(exist_ok=True, parents=True)

DATASET_KIND_RAW = 1
DATASET_KIND_ALIGNED = 2
DATASET_KIND_MASKS = 3
DATASET_KIND_LATENTS = 4
DATASET_KIND_GENERATED = 5
DATASET_KIND_MORPH = 6

DATASET_KIND_MAP = {
    DATASET_KIND_RAW: DATASET_RAW_FOLDER,
    DATASET_KIND_ALIGNED: DATASET_ALIGNED_FOLDER,
    DATASET_KIND_MASKS: DATASET_MASKS_FOLDER,
    DATASET_KIND_LATENTS: DATASET_LATENTS_FOLDER,
    DATASET_KIND_GENERATED: DATASET_GENERATED_FOLDER,
    DATASET_KIND_MORPH: DATASET_MORPH_FOLDER,
}

DATASET_KIND_STR = {
    DATASET_KIND_RAW: "raw",
    DATASET_KIND_ALIGNED: "aligned",
    DATASET_KIND_MASKS: "mask",
    DATASET_KIND_LATENTS: "latent",
    DATASET_KIND_GENERATED: "generated",
    DATASET_KIND_MORPH: "morph",
}


def get_file_path(name: str, dataset_kind: int, file_extension: str = ".png") -> Path:
    try:
        return Path(DATASET_KIND_MAP[dataset_kind], name + file_extension)
    except KeyError:
        raise ValueError(f"Invalid folder_kind: {dataset_kind}")


def read_latent(name: str) -> np.ndarray:
    return np.load(
        get_file_path(
            name=name, dataset_kind=DATASET_KIND_LATENTS, file_extension=".npy"
        )
    )


def read_latents(names: list) -> np.ndarray:
    latents = []
    for name in names:
        latents.append(read_latent(name=name))
    return np.array(latents)


def read_mask(name: str) -> Image:
    return Image.open(
        get_file_path(name=name, dataset_kind=DATASET_KIND_MASKS, file_extension=".png")
    ).convert(mode="L")


def read_aligned(name: str) -> Image:
    return Image.open(
        get_file_path(
            name=name, dataset_kind=DATASET_KIND_ALIGNED, file_extension=".png"
        )
    ).convert("RGB")


def gen_dataset_index() -> DataFrame:
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
    for kind_idx, folder in DATASET_KIND_MAP.items():
        kind_str = DATASET_KIND_STR[kind_idx]
        for img_path in folder.iterdir():
            if img_path.is_file():
                entries.append(
                    {
                        "name": img_path.stem,
                        "kind": kind_str,
                        "extension": img_path.suffix,
                        "img_path": str(img_path),
                    }
                )

    entries = DataFrame(entries)

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
