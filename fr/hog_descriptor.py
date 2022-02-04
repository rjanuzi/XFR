import numpy as np
from cv2 import HOGDescriptor
from PIL import Image, ImageOps

from fr.face_decomposition import (
    decompose_face,
    get_ears,
    get_eyebrows,
    get_eyes,
    get_face,
    get_mouth,
    get_nose,
)

__HOG_DEFAULT_SIZE = (64, 128)
__HOG_DISTANCE_FACE_WEIGHT = 1.0
__HOG_DISTANCE_EYES_WEIGHT = 1.0
__HOG_DISTANCE_EYEBROWS_WEIGHT = 1.0
__HOG_DISTANCE_EARS_WEIGHT = 1.0
__HOG_DISTANCE_NOSE_WEIGHT = 1.0
__HOG_DISTANCE_MOUTH_WEIGHT = 1.0


def calc_face_hog(name: str):
    face_parts = decompose_face(name)
    face = get_face(face_parts=face_parts)
    eyes = get_eyes(face_parts=face_parts)
    eyebrows = get_eyebrows(face_parts=face_parts)
    ears = get_ears(face_parts=face_parts)
    nose = get_nose(face_parts=face_parts)
    mouth = get_mouth(face_parts=face_parts)

    # Adjust images to HOG size and to grayscale
    face = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(face).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )
    eyes = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(eyes).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )
    eyebrows = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(eyebrows).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )
    ears = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(ears).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )
    nose = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(nose).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )
    mouth = np.asarray(
        ImageOps.grayscale(
            Image.fromarray(mouth).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )

    hog = HOGDescriptor()
    face_hog = hog.compute(face)
    eyes_hog = hog.compute(eyes)
    eyebrows_hog = hog.compute(eyebrows)
    ears_hog = hog.compute(ears)
    nose_hog = hog.compute(nose)
    mouth_hog = hog.compute(mouth)

    seg_hogs = np.vstack(
        (face_hog, eyes_hog, eyebrows_hog, ears_hog, nose_hog, mouth_hog)
    )

    return seg_hogs


def compare_face_hogs(
    face_hog_1: np.array,
    face_hog_2: np.array,
    face_weight: float = __HOG_DISTANCE_FACE_WEIGHT,
    eyes_weight: float = __HOG_DISTANCE_EYES_WEIGHT,
    eyebrows_weight: float = __HOG_DISTANCE_EYEBROWS_WEIGHT,
    ears_weight: float = __HOG_DISTANCE_EARS_WEIGHT,
    nose_weight: float = __HOG_DISTANCE_NOSE_WEIGHT,
    mouth_weight: float = __HOG_DISTANCE_MOUTH_WEIGHT,
):
    hog_abs_delta = np.abs(face_hog_1 - face_hog_2)
    acc_weighted_delta = 0.0

    acc_weighted_delta += face_weight * hog_abs_delta[0].sum()
    acc_weighted_delta += eyes_weight * hog_abs_delta[1].sum()
    acc_weighted_delta += eyebrows_weight * hog_abs_delta[2].sum()
    acc_weighted_delta += ears_weight * hog_abs_delta[3].sum()
    acc_weighted_delta += nose_weight * hog_abs_delta[4].sum()
    acc_weighted_delta += mouth_weight * hog_abs_delta[5].sum()

    return acc_weighted_delta
