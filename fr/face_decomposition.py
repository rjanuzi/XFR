from turtle import right
import numpy as np
from PIL import Image

from dataset import DATASET_KIND_ALIGNED, DATASET_KIND_SEG_MAP, get_file_path

__DEFAULT_WIDTH = 512
__DEFAULT_HEIGHT = 512
__BACKGROUND_CLASS = 0
__FACE_CLASS = 1
__LEFT_EYE_BROW_CLASS = 2
__RIGHT_EYE_BROW_CLASS = 3
__LEFT_EYE_CLASS = 4
__RIGHT_EYE_CLASS = 5
__LEFT_EAR_CLASS = 7
__RIGHT_EAR_CLASS = 8
__NOSE_CLASS = 10
__UPPER_LIP_CLASS = 12
__LOWER_LIP_CLASS = 13
__NECK_CLASS = 14
__SHOULDER_CLASS = 16
__HAIR_CLASS = 17


def decompose_face(img_name: str) -> dict:
    seg_map = np.load(get_file_path(img_name, DATASET_KIND_SEG_MAP, ".npy"))
    original_img = Image.open(get_file_path(img_name, DATASET_KIND_ALIGNED, ".png"))
    original_img = original_img.resize(
        (__DEFAULT_WIDTH, __DEFAULT_HEIGHT), Image.LANCZOS
    )
    original_img_array = np.array(original_img)

    face_parts = {}
    for seg_class in np.unique(seg_map):
        tmp_face_seg = np.zeros(original_img_array.shape).astype(np.uint8)
        tmp_idxes = np.where(seg_map == seg_class)
        tmp_face_seg[tmp_idxes] = original_img_array[tmp_idxes]
        face_parts[seg_class] = tmp_face_seg

    return face_parts


def crop_roi(img_array: np.ndarray) -> np.ndarray:
    roi_idxes = np.where(img_array != 0)
    return img_array[
        roi_idxes[0].min() : roi_idxes[0].max(),
        roi_idxes[1].min() : roi_idxes[1].max(),
    ]


def get_face(face_parts: dict):
    face = face_parts[__FACE_CLASS]
    return crop_roi(face)


def get_eyes(face_parts: dict):
    left_eye = face_parts[__LEFT_EYE_CLASS]
    right_eye = face_parts[__RIGHT_EYE_CLASS]
    mixed = left_eye + right_eye

    return crop_roi(mixed)


def get_eyebrows(face_parts: dict):
    left_eyebrow = face_parts[__LEFT_EYE_BROW_CLASS]
    right_eyebrow = face_parts[__RIGHT_EYE_BROW_CLASS]
    mixed = left_eyebrow + right_eyebrow

    return crop_roi(mixed)


def get_ears(face_parts: dict):
    try:
        left_ear = face_parts[__LEFT_EAR_CLASS]
    except KeyError:
        left_ear = np.zeros(face_parts[__FACE_CLASS].shape).astype(np.uint8)

    try:
        right_ear = face_parts[__RIGHT_EAR_CLASS]
    except KeyError:
        right_ear = np.zeros(face_parts[__FACE_CLASS].shape).astype(np.uint8)

    mixed = left_ear + right_ear

    return crop_roi(mixed)


def get_nose(face_parts: dict):
    nose = face_parts[__NOSE_CLASS]
    return crop_roi(nose)


def get_mouth(face_parts: dict):
    upper_lip = face_parts[__UPPER_LIP_CLASS]
    lower_lip = face_parts[__LOWER_LIP_CLASS]
    mixed = upper_lip + lower_lip

    return crop_roi(mixed)


face_parts = decompose_face("01089")
face = Image.fromarray(get_face(face_parts))
eyes = Image.fromarray(get_eyes(face_parts))
eyebrows = Image.fromarray(get_eyebrows(face_parts))
ears = Image.fromarray(get_ears(face_parts))
nose = Image.fromarray(get_nose(face_parts))
mouth = Image.fromarray(get_mouth(face_parts))
