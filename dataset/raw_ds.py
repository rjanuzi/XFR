from pathlib import Path

from pandas import DataFrame

RAW_DATASET_FOLDER = Path(Path(__file__).parent, "raw")
RAW_DATASET_FOLDER.mkdir(exist_ok=True)


def lookup_raw_imgs(person_names: list = None, poses: list = None) -> DataFrame:
    """
    Generate the dataset entries.
    In:
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
    for person_dir in RAW_DATASET_FOLDER.iterdir():
        if person_dir.is_dir():
            try:
                poses_files = [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.jpg")
                ]

                poses_files += [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.jpeg")
                ]

                poses_files += [
                    (file.stem, file.absolute()) for file in person_dir.glob("*.png")
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


def ls_imgs_paths(person_names: list = None, poses: list = None):
    """
    List the images paths and poses.
    In:
        person_names: The list of person names to include, if None, all are include
        poses: The list of poses to include, if None, all are include
    Out: A list with the images paths.
    """
    entries = lookup_raw_imgs(person_names, poses)
    return entries["img_path"].tolist()
