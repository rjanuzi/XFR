import numpy as np
from cv2 import HOGDescriptor
from PIL import Image, ImageOps

from fr.face_decomposition import (
    get_face,
    get_eyes,
    get_eyebrows,
    get_ears,
    get_left_eye,
    get_right_eye,
    get_nose,
    get_left_eyebrow,
    get_rigth_eyebrow,
    get_left_ear,
    get_right_ear,
    get_lower_lip,
    get_upper_lip,
    get_mouth,
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
    elif opt == HOG_OPT_FACE:
        face = get_face(face_parts, use_hog_proportion=True)
        if face is None:
            return None
        face = adjust_img_to_hog(face)
        hog = HOGDescriptor()
        face_hog = hog.compute(face)
        return face_hog
    elif opt == HOG_OPT_LEFT_EYE:
        left_eye = get_left_eye(face_parts, use_hog_proportion=True)
        if left_eye is None:
            return None
        left_eye = adjust_img_to_hog(left_eye)
        hog = HOGDescriptor()
        left_eye_hog = hog.compute(left_eye)
        return left_eye_hog
    elif opt == HOG_OPT_RIGHT_EYE:
        right_eye = get_right_eye(face_parts, use_hog_proportion=True)
        if right_eye is None:
            return None
        right_eye = adjust_img_to_hog(right_eye)
        hog = HOGDescriptor()
        right_eye_hog = hog.compute(right_eye)
        return right_eye_hog
    elif opt == HOG_OPT_EYES:
        eyes = get_eyes(face_parts, use_hog_proportion=True)
        if eyes is None:
            return None
        eyes = adjust_img_to_hog(eyes)
        hog = HOGDescriptor()
        eyes_hog = hog.compute(eyes)
        return eyes_hog
    elif opt == HOG_OPT_LEFT_EYEBROW:
        left_eyebrow = get_left_eyebrow(face_parts, use_hog_proportion=True)
        if left_eyebrow is None:
            return None
        left_eyebrow = adjust_img_to_hog(left_eyebrow)
        hog = HOGDescriptor()
        left_eyebrow_hog = hog.compute(left_eyebrow)
        return left_eyebrow_hog
    elif opt == HOG_OPT_RIGHT_EYEBROW:
        right_eyebrow = get_rigth_eyebrow(face_parts, use_hog_proportion=True)
        if right_eyebrow is None:
            return None
        right_eyebrow = adjust_img_to_hog(right_eyebrow)
        hog = HOGDescriptor()
        right_eyebrow_hog = hog.compute(right_eyebrow)
        return right_eyebrow_hog
    elif opt == HOG_OPT_EYEBROWS:
        eyebrows = get_eyebrows(face_parts, use_hog_proportion=True)
        if eyebrows is None:
            return None
        eyebrows = adjust_img_to_hog(eyebrows)
        hog = HOGDescriptor()
        eyebrows_hog = hog.compute(eyebrows)
        return eyebrows_hog
    elif opt == HOG_OPT_LEFT_EAR:
        left_ear = get_left_ear(face_parts, use_hog_proportion=True)
        if left_ear is None:
            return None
        left_ear = adjust_img_to_hog(left_ear)
        hog = HOGDescriptor()
        left_ear_hog = hog.compute(left_ear)
        return left_ear_hog
    elif opt == HOG_OPT_RIGHT_EAR:
        right_ear = get_right_ear(face_parts, use_hog_proportion=True)
        if right_ear is None:
            return None
        right_ear = adjust_img_to_hog(right_ear)
        hog = HOGDescriptor()
        right_ear_hog = hog.compute(right_ear)
        return right_ear_hog
    elif opt == HOG_OPT_EARS:
        ears = get_ears(face_parts, use_hog_proportion=True)
        if ears is None:
            return None
        ears = adjust_img_to_hog(ears)
        hog = HOGDescriptor()
        ears_hog = hog.compute(ears)
        return ears_hog
    elif opt == HOG_OPT_NOSE:
        nose = get_nose(face_parts, use_hog_proportion=True)
        if nose is None:
            return None
        nose = adjust_img_to_hog(nose)
        hog = HOGDescriptor()
        nose_hog = hog.compute(nose)
        return nose_hog
    elif opt == HOG_OPT_LOWER_LIP:
        lower_lip = get_lower_lip(face_parts, use_hog_proportion=True)
        if lower_lip is None:
            return None
        lower_lip = adjust_img_to_hog(lower_lip)
        hog = HOGDescriptor()
        lower_lip_hog = hog.compute(lower_lip)
        return lower_lip_hog
    elif opt == HOG_OPT_UPPER_LIP:
        upper_lip = get_upper_lip(face_parts, use_hog_proportion=True)
        if upper_lip is None:
            return None
        upper_lip = adjust_img_to_hog(upper_lip)
        hog = HOGDescriptor()
        upper_lip_hog = hog.compute(upper_lip)
        return upper_lip_hog
    elif opt == HOG_OPT_MOUTH:
        mouth = get_mouth(face_parts, use_hog_proportion=True)
        if mouth is None:
            return None
        mouth = adjust_img_to_hog(mouth)
        hog = HOGDescriptor()
        mouth_hog = hog.compute(mouth)
        return mouth_hog


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
