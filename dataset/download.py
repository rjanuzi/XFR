import gdown

from pathlib import Path

DATASET_FOLDER = Path(__file__).parent

DATASET_RAW_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1kXjCPA-WGGhP33WKBa-dmSEdtCwkPLXD?usp=sharing"
DATASET_ALIGNED_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1Vv0S90hy93UU4lZK8h3DQTA-kshfZA7e?usp=sharing"
DATASET_MASKS_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1b_hDr1SDeChxuhXLNZ-vAERS9hTkNeN8?usp=sharing"
DATASET_LATENTS_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1S31cZLzZPxuoHXTI2fVdOqkpCj1uoet8?usp=sharing"
DATASET_GENERATED_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/1bEmNrzZw-gi1tdpwWfdN2M5T_b4gM1Qm?usp=sharing"
DATASET_MORPH_DRIVE_SHARED_FOLDER = "https://drive.google.com/drive/folders/18atoh66hNDRAZLCxQ_voX_QPXO6gA6S9?usp=sharing"

GDRIVE_FOLDERS = [
    DATASET_RAW_DRIVE_SHARED_FOLDER,
    DATASET_ALIGNED_DRIVE_SHARED_FOLDER,
    DATASET_MASKS_DRIVE_SHARED_FOLDER,
    DATASET_LATENTS_DRIVE_SHARED_FOLDER,
    DATASET_GENERATED_DRIVE_SHARED_FOLDER,
    DATASET_MORPH_DRIVE_SHARED_FOLDER,
]

if __name__ == "__main__":
    for gdrive_folder in GDRIVE_FOLDERS:
        print("Downloading gdrive folder...")
        gdown.download_folder(url=gdrive_folder, output=DATASET_FOLDER)
        print("Done.")
