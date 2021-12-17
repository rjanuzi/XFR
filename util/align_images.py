import bz2
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
