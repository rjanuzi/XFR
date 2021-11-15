from pathlib import Path

from pandas import DataFrame

RAW_DATASET_FOLDER = Path(Path(__file__).parent, "raw")
RAW_DATASET_FOLDER.mkdir(exist_ok=True)


def lookup_raw_imgs(
    root_folder=None, person_names: list = None, poses: list = None
) -> DataFrame:
    """
    Generate the dataset entries.
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
    root_folder = Path(root_folder) if root_folder else RAW_DATASET_FOLDER
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

                poses_files += [
                    (file.stem, file.absolute())
                    for file in person_dir.glob("[!mask]*.png")
                ]
            except NotADirectoryError as e:
                print("[ERROR]: dataset.raw_ds.py - Invalid dataset structure")
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


def ls_imgs_paths(root_folder=None, person_names: list = None, poses: list = None):
    """
    List the images paths and poses.
    In:
        root_folder: The folder containing the dataset
        person_names: The list of person names to include, if None, all are include
        poses: The list of poses to include, if None, all are include
    Out: A list with the images paths.
    """
    entries = lookup_raw_imgs(root_folder, person_names, poses)
    return entries["img_path"].tolist()
