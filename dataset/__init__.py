from pathlib import Path

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


def get_dataset_index(recreate: bool = False, dataset: str = "all") -> pd.DataFrame:
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
        for tmp_dataset in __DATASETS:
            for kind in __DATASET_KINDS:
                tmp_folder = __ROOT_FOLDER.joinpath(tmp_dataset, kind)
                for person_folder in tmp_folder.iterdir():
                    for img_path in person_folder.iterdir():
                        if img_path.is_file():
                            entries.append(
                                {
                                    "dataset": tmp_dataset,
                                    "kind": kind,
                                    "person_name": person_folder.stem,
                                    "extension": img_path.suffix,
                                    "img_path": str(img_path),
                                }
                            )

        dataset_idx = pd.DataFrame(entries)

        # Save the index file
        dataset_idx.to_pickle(___DATASET_IDX)

        print("Datasets index file created!")

    else:
        dataset_idx = pd.read_pickle(___DATASET_IDX)

    if dataset == "all":
        return dataset_idx
    else:
        return dataset_idx.loc[dataset_idx["dataset"] == dataset]


def info():
    dataset_index = get_dataset_index()
    datasets = dataset_index.dataset.unique()
    print(f"Datasets: {datasets}")
    for ds in datasets:
        filtered = dataset_index.loc[dataset_index["dataset"] == ds]
        print(f"Dataset: {ds}")
        print(f"\tPersons: {len(filtered.person_name.unique())}")
        print(f"\tImages: {filtered.shape[0]}")


def get_file_path(
    dataset: str,
    kind: str,
    person_name: str,
    image_name: str,
    file_extension: str = ".png",
) -> Path:
    try:
        return __ROOT_FOLDER.joinpath(
            dataset, kind, person_name, image_name + file_extension
        )
    except KeyError:
        raise ValueError(f"Invalid dataset: {dataset}")


def read_aligned(dataset: str, person_name: str, image_name: str) -> Image:
    return Image.open(
        get_file_path(
            dataset=dataset,
            kind=DATASET_KIND_ALIGNED,
            person_name=person_name,
            image_name=image_name,
        )
    ).convert("RGB")


def ls_imgs_paths(dataset: str, kind: str = DATASET_KIND_RAW) -> list:
    """
    List the images paths.
    Out: A list with the raw images paths.
    """
    entries = get_dataset_index(dataset=dataset)
    return entries.loc[entries["kind"] == kind, "img_path"].tolist()


def ls_imgs_person_names(dataset: str, kind: str = DATASET_KIND_RAW) -> list:
    """
    List the images paths.
    Out: A list with the raw images paths.
    """
    entries = get_dataset_index(dataset=dataset)
    return entries.loc[entries["kind"] == kind, "person_name"].tolist()
