import bz2
import os
import sys
from pathlib import Path

import requests

from util.ffhq_dataset.face_alignment import image_align
from util.ffhq_dataset.landmarks_detector import LandmarksDetector

LANDMARKS_TEMP_FOLDER = Path(".land_mark_cache")
LANDMARKS_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
LANDMARKS_MODEL_FNAME = "shape_predictor_68_face_landmarks.dat.bz2"


def unpack_bz2(src_path):
    print("Unpacking landmarks file...")

    dst_path = Path(src_path.parent, src_path.stem)
    if not dst_path.exists():
        data = bz2.BZ2File(src_path).read()
        with open(dst_path, "wb") as fp:
            fp.write(data)

    print("done.")

    return dst_path


def get_landmark_model(fname=LANDMARKS_MODEL_FNAME, url=LANDMARKS_MODEL_URL):
    print(
        "Downloading landmark model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.."
    )

    file = Path(LANDMARKS_TEMP_FOLDER, fname)
    if not file.exists():
        LANDMARKS_TEMP_FOLDER.mkdir(parents=True, exist_ok=True)
        r = requests.get(url)
        with open(file, "wb") as fp:
            fp.write(r.content)

    print("done.")

    return unpack_bz2(file)


def align_images(imgs_path_lst, output_folder_path_lst):
    """
    Extracts and aligns all faces listed in params using the function from original FFHQ dataset preparation step
    :param imgs_path_lst: list of paths to images
    :param output_path_lst: list of paths to output images
    """
    landmarks_detector = LandmarksDetector(get_landmark_model())
    print("Aligning images...")
    for img_path, output_folder in zip(imgs_path_lst, output_folder_path_lst):
        for i, face_landmarks in enumerate(
            landmarks_detector.get_landmarks(img_path), start=1
        ):
            output_img_path = output_folder.joinpath(f"{img_path.stem}.png")
            image_align(img_path, output_img_path, face_landmarks)
            print(f"Aligned {i} faces from {img_path} to {output_img_path}")
    print("done.")


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    RAW_IMAGES_DIR = sys.argv[1]
    ALIGNED_IMAGES_DIR = sys.argv[2]

    landmarks_detector = LandmarksDetector(get_landmark_model())
    for img_name in [x for x in os.listdir(RAW_IMAGES_DIR) if x[0] not in "._"]:
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(
            landmarks_detector.get_landmarks(raw_img_path), start=1
        ):
            face_img_name = "%s_%02d.png" % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            os.makedirs(ALIGNED_IMAGES_DIR, exist_ok=True)
            image_align(raw_img_path, aligned_face_path, face_landmarks)
