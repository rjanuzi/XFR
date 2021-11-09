import gdown

from raw_ds import RAW_DATASET_FOLDER
from processed_ds import PROCESSED_DATASET_FOLDER

DATASET_RAW_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1kXjCPA-WGGhP33WKBa-dmSEdtCwkPLXD?usp=sharing"

DATASET_PROCESSED_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1RDBurXFlOeSri6fONd3V3Q_PqGkkjNve?usp=sharing"


def download_raw_folder():
    gdown.download_folder(
        url=DATASET_RAW_DRIVE_SHARED_FOLDER, output=RAW_DATASET_FOLDER
    )


def download_processed_folder():
    gdown.download_folder(
        url=DATASET_PROCESSED_DRIVE_SHARED_FOLDER, output=PROCESSED_DATASET_FOLDER
    )


if __name__ == "__main__":
    download_raw_folder()
    download_processed_folder()