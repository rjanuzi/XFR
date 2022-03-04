import numpy as np
from cv2 import HOGDescriptor
from PIL import Image, ImageOps

from fr.face_decomposition import (
    get_ears,
    get_eyebrows,
    get_eyes,
    get_eyes_and_eyebrows,
    get_eyes_and_nose,
    get_face,
    get_full_face,
    get_left_ear,
    get_left_eye,
    get_left_eyebrow,
    get_lower_lip,
    get_mouth,
    get_mouth_and_nose,
    get_nose,
    get_right_ear,
    get_right_eye,
    get_rigth_eyebrow,
    get_upper_lip,
)

__HOG_DEFAULT_SIZE = (64, 128)
__HOG_DISTANCE_FACE_WEIGHT = 1.0
__HOG_DISTANCE_EYES_WEIGHT = 1.0
__HOG_DISTANCE_EYEBROWS_WEIGHT = 1.0
__HOG_DISTANCE_EARS_WEIGHT = 1.0
__HOG_DISTANCE_NOSE_WEIGHT = 1.0
__HOG_DISTANCE_MOUTH_WEIGHT = 1.0

HOG_OPT_ALL = 0
HOG_OPT_FACE = 1
HOG_OPT_EYES = 2
HOG_OPT_EYEBROWS = 3
HOG_OPT_EARS = 4
HOG_OPT_NOSE = 5
HOG_OPT_MOUTH = 6
HOG_OPT_LEFT_EYE = 7
HOG_OPT_RIGHT_EYE = 8
HOG_OPT_LEFT_EYEBROW = 9
HOG_OPT_RIGHT_EYEBROW = 10
HOG_OPT_LEFT_EAR = 11
HOG_OPT_RIGHT_EAR = 12
HOG_OPT_LOWER_LIP = 13
HOG_OPT_UPPER_LIP = 14
HOG_OPT_MOUTH_AND_NOSE = 15
HOG_OPT_EYES_AND_EYEBROWS = 16
HOG_OPT_EYES_AND_NOSE = 17
HOG_OPT_FULL_FACE = 18

__calc_hog_funcs = {
    HOG_OPT_FACE: get_face,
    HOG_OPT_LEFT_EYE: get_left_eye,
    HOG_OPT_RIGHT_EYE: get_right_eye,
    HOG_OPT_EYES: get_eyes,
    HOG_OPT_LEFT_EYEBROW: get_left_eyebrow,
    HOG_OPT_RIGHT_EYEBROW: get_rigth_eyebrow,
    HOG_OPT_EYEBROWS: get_eyebrows,
    HOG_OPT_LEFT_EAR: get_left_ear,
    HOG_OPT_RIGHT_EAR: get_right_ear,
    HOG_OPT_EARS: get_ears,
    HOG_OPT_NOSE: get_nose,
    HOG_OPT_LOWER_LIP: get_lower_lip,
    HOG_OPT_UPPER_LIP: get_upper_lip,
    HOG_OPT_MOUTH: get_mouth,
    HOG_OPT_MOUTH_AND_NOSE: get_mouth_and_nose,
    HOG_OPT_EYES_AND_EYEBROWS: get_eyes_and_eyebrows,
    HOG_OPT_EYES_AND_NOSE: get_eyes_and_nose,
    HOG_OPT_FULL_FACE: get_full_face,
}


def adjust_img_to_hog(img_array: np.array):
    return np.asarray(
        ImageOps.grayscale(
            Image.fromarray(img_array).resize(__HOG_DEFAULT_SIZE, Image.LANCZOS)
        )
    )


def calc_hog(face_parts: dict, opt: int):

    if opt == HOG_OPT_ALL:
        face = get_face(face_parts, use_hog_proportion=True)
        eyes = get_eyes(face_parts, use_hog_proportion=True)
        eyebrows = get_eyebrows(face_parts, use_hog_proportion=True)
        ears = get_ears(face_parts, use_hog_proportion=True)
        nose = get_nose(face_parts, use_hog_proportion=True)
        mouth = get_mouth(face_parts, use_hog_proportion=True)

        if any(
            [
                face is None,
                eyes is None,
                eyebrows is None,
                ears is None,
                nose is None,
                mouth is None,
            ]
        ):
            return None

        # Adjust images to HOG size and to grayscale
        face = adjust_img_to_hog(face)
        eyes = adjust_img_to_hog(eyes)
        eyebrows = adjust_img_to_hog(eyebrows)
        ears = adjust_img_to_hog(ears)
        nose = adjust_img_to_hog(nose)
        mouth = adjust_img_to_hog(mouth)

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
    else:
        # Use the function to get the face parts according to the option
        face_elements = __calc_hog_funcs[opt](face_parts, use_hog_proportion=True)
        if face_elements is None:
            return None
        else:
            face_elements = adjust_img_to_hog(face_elements)
            hog = HOGDescriptor()
            face_elements_hog = hog.compute(face_elements)
            return face_elements_hog


def compare_hogs(hog_1: np.array, hog_2: np.array, opt: int):
    hog_abs_delta = np.abs(hog_1 - hog_2)
    if opt == HOG_OPT_ALL:
        acc_weighted_delta = 0.0
        acc_weighted_delta += __HOG_DISTANCE_FACE_WEIGHT * hog_abs_delta[0].sum()
        acc_weighted_delta += __HOG_DISTANCE_EYES_WEIGHT * hog_abs_delta[1].sum()
        acc_weighted_delta += __HOG_DISTANCE_EYEBROWS_WEIGHT * hog_abs_delta[2].sum()
        acc_weighted_delta += __HOG_DISTANCE_EARS_WEIGHT * hog_abs_delta[3].sum()
        acc_weighted_delta += __HOG_DISTANCE_NOSE_WEIGHT * hog_abs_delta[4].sum()
        acc_weighted_delta += __HOG_DISTANCE_MOUTH_WEIGHT * hog_abs_delta[5].sum()

        return acc_weighted_delta
    else:
        return hog_abs_delta.sum()
