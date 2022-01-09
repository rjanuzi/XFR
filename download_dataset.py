import json
import sys
from pathlib import Path
from time import sleep, time

import gdown
from requests import get

from dataset import DATASET_RAW_FOLDER
from util._telegram import send_simple_message

DATASET_FOLDER = Path("dataset")
FFHQ_FRONTAL_REF_FILE = DATASET_FOLDER.joinpath("ffhq_frontal.json")
MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 30
RESTART_WAIT_TIME_SECONDS = 15 * 60

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


def get_content(url):
    img_data = get(drive_url).content
    if img_data and b"html" not in img_data:
        return img_data


if __name__ == "__main__":
    params = sys.argv[1:]

    if len(params) != 1:
        params.append("ffhq_frontal")

    print(f"Starting downaload {params[0]} images.")

    if params[0] == "drive_folders":
        for gdrive_folder in GDRIVE_FOLDERS:
            print("Downloading gdrive folder...")
            gdown.download_folder(url=gdrive_folder, output=DATASET_FOLDER)
            print("Done.")
    elif params[0] == "ffhq_frontal":
        try:
            start_time = time()
            ffhq_frontal_imgs = json.load(open(FFHQ_FRONTAL_REF_FILE, "r"))
            for img_ref in ffhq_frontal_imgs:
                img_name = img_ref["name"]
                local_img_path = DATASET_RAW_FOLDER.joinpath(f"{img_name}.jpg")
                if not local_img_path.exists():
                    drive_url = img_ref["drive_url"]
                    tries = 0
                    img_data = None
                    while img_data is None:
                        tries += 1
                        print(f"Downloading {img_name} from {drive_url}...")
                        img_data = get_content(drive_url)
                        if img_data is None:
                            if tries <= MAX_RETRIES:
                                print(
                                    f"Error downloading {img_name}. Waiting {RETRY_DELAY_SECONDS} seconds to retry..."
                                )
                                sleep(30)
                            else:
                                print(
                                    f"Error downloading {img_name}. Waiting {int(RESTART_WAIT_TIME_SECONDS/60)} minutes to restart process..."
                                )
                                send_simple_message(
                                    f"Error downloading {img_name}. Waiting {int(RESTART_WAIT_TIME_SECONDS/60)} minutes to restart process..."
                                )
                                tries = 0
                                sleep(15 * 60)
                            continue

                        with open(local_img_path, "wb") as handler:
                            handler.write(img_data)

            send_simple_message(
                f"FFHQ Frontal images downloaded. {int(time() - start_time)} seconds"
            )
        except Exception as e:
            send_simple_message("Error downloading FFHQ Frontal images.")
            raise e
    else:
        raise ValueError("Wrong parameters provided")
