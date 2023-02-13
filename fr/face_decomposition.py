from turtle import right

import numpy as np
from matplotlib import use
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

__DEFAULT_HOG_WIDTH = 64
__DEFAULT_HOG_HEIGHT = 128


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


def decompose_face_no_blank(
    img_name: str, target_shape: tuple[int, int] = (128, 128)
) -> dict:
    """
    Decompose the face into its parts, but without adding blank space to the image.

    :param img_name: The name of the image to decompose
    :param target_shape: The target shape of the image. The target shape for this version SHALL BE SQUARED (width == height).

    :return: A dictionary with the face parts, where the key is the class of the part and the value is the image of the part.
    """
    seg_map = np.load(get_file_path(img_name, DATASET_KIND_SEG_MAP, ".npy"))
    original_img = Image.open(get_file_path(img_name, DATASET_KIND_ALIGNED, ".png"))
    original_img = original_img.resize(
        (__DEFAULT_WIDTH, __DEFAULT_HEIGHT), Image.LANCZOS
    )
    original_img_array = np.array(original_img)

    face_parts = {}
    for seg_class in np.unique(seg_map):
        try:
            tmp_idxes = np.where(seg_map == seg_class)

            # Get ROI crop limits
            y_min = tmp_idxes[0].min()
            y_max = tmp_idxes[0].max()
            x_min = tmp_idxes[1].min()
            x_max = tmp_idxes[1].max()

            # Get ROI size
            roi_width = x_max - x_min
            roi_height = y_max - y_min

            if roi_width > roi_height:
                # If the ROI is wider than it is high, expand the ROI to the target size
                height_inc = (roi_width - roi_height) // 2
                y_min -= height_inc
                y_max += height_inc

                # If the ROI is expanded to the top, make sure it doesn't go out of bounds
                if y_min < 0:
                    y_max += abs(y_min)
                    y_min = 0

            elif roi_height > roi_width:
                # If the ROI is higher than it is wide, expand the ROI to the target size
                width_inc = (roi_height - roi_width) // 2
                x_min -= width_inc
                x_max += width_inc

                # If the ROI is expanded to the left, make sure it doesn't go out of bounds
                if x_min < 0:
                    x_max += abs(x_min)
                    x_min = 0

            # Crop the image using the ROI limits, adjusted to squared size and trying to center the ROI
            cropped_img = original_img_array[
                y_min:y_max,
                x_min:x_max,
            ]

            # Resize the image to the target size (both same aspect ratio)
            cropped_img = Image.fromarray(cropped_img)
            cropped_img = cropped_img.resize(size=target_shape)

            face_parts[seg_class] = np.array(cropped_img)
        except:
            face_parts[seg_class] = None

    return face_parts


def crop_roi(img_array: np.ndarray, use_hog_proportion) -> np.ndarray:
    roi_idxes = np.where(img_array != 0)

    # Simply crop using only the relevant part of the image
    cropped_img = img_array[
        roi_idxes[0].min() : roi_idxes[0].max(),
        roi_idxes[1].min() : roi_idxes[1].max(),
    ]

    if use_hog_proportion:
        # Generate a image 1:2 (HOG input) with the cropped ROI centered
        target_width = __DEFAULT_HOG_WIDTH
        target_height = __DEFAULT_HOG_HEIGHT

        # While the ROI is bigger than the target size, expand the image size
        while (
            cropped_img.shape[0] > target_height or cropped_img.shape[1] > target_width
        ):
            target_width = target_width * 2
            target_height = target_height * 2

        # Pad the image with zeros in order to center the ROI in a image with the target size
        dy = (target_height - cropped_img.shape[0]) // 2
        dx = (target_width - cropped_img.shape[1]) // 2

        # Pad half of the difference to the left, right, top and bottom
        cropped_img = np.pad(
            array=cropped_img, pad_width=[(dy, dy), (dx, dx), (0, 0)], mode="constant"
        )

    return cropped_img


def get_face(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        face = face_parts[__FACE_CLASS]
        if no_blank:
            return face
        return crop_roi(img_array=face, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_left_eye(
    face_parts: dict, use_hog_proportion=False, target_shape=(128, 128), no_blank=False
):
    try:
        left_eye = face_parts[__LEFT_EYE_CLASS]

        if no_blank:
            return left_eye
        else:
            return crop_roi(img_array=left_eye, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_right_eye(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        right_eye = face_parts[__RIGHT_EYE_CLASS]

        if no_blank:
            return right_eye

        return crop_roi(img_array=right_eye, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_eyes(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:

        if not no_blank:
            left_eye = face_parts[__LEFT_EYE_CLASS]
            right_eye = face_parts[__RIGHT_EYE_CLASS]

            mixed = left_eye + right_eye

            return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
        else:
            # TODO
            # tmp_img = Image.fromarray(mixed)
            # tmp_img = tmp_img.resize(size=left_eye.shape[:2])
            # return np.array(tmp_img)
            return None
    except Exception:
        return None


def get_left_eyebrow(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_eyebrow = face_parts[__LEFT_EYE_BROW_CLASS]

        if no_blank:
            return left_eyebrow

        return crop_roi(img_array=left_eyebrow, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_rigth_eyebrow(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        right_eyebrow = face_parts[__RIGHT_EYE_BROW_CLASS]

        if no_blank:
            return right_eyebrow

        return crop_roi(img_array=right_eyebrow, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_eyebrows(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_eyebrow = face_parts[__LEFT_EYE_BROW_CLASS]
        right_eyebrow = face_parts[__RIGHT_EYE_BROW_CLASS]

        mixed = left_eyebrow + right_eyebrow

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=left_eyebrow.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_left_ear(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_ear = face_parts[__LEFT_EAR_CLASS]

        if no_blank:
            return left_ear

        return crop_roi(img_array=left_ear, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_right_ear(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        right_ear = face_parts[__RIGHT_EAR_CLASS]

        if no_blank:
            return right_ear

        return crop_roi(img_array=right_ear, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_ears(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_ear = face_parts[__LEFT_EAR_CLASS]
        right_ear = face_parts[__RIGHT_EAR_CLASS]

        mixed = left_ear + right_ear

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=left_ear.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_nose(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        nose = face_parts[__NOSE_CLASS]

        if no_blank:
            return nose

        return crop_roi(img_array=nose, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_lower_lip(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        lower_lip = face_parts[__LOWER_LIP_CLASS]

        if no_blank:
            return lower_lip

        return crop_roi(img_array=lower_lip, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_upper_lip(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        upper_lip = face_parts[__UPPER_LIP_CLASS]

        if no_blank:
            return upper_lip

        return crop_roi(img_array=upper_lip, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_mouth(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        upper_lip = face_parts[__UPPER_LIP_CLASS]
        lower_lip = face_parts[__LOWER_LIP_CLASS]

        mixed = upper_lip + lower_lip

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=upper_lip.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_mouth_and_nose(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        upper_lip = face_parts[__UPPER_LIP_CLASS]
        lower_lip = face_parts[__LOWER_LIP_CLASS]
        nose = face_parts[__NOSE_CLASS]

        mixed = upper_lip + lower_lip + nose

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=upper_lip.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_eyes_and_eyebrows(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_eye = face_parts[__LEFT_EYE_CLASS]
        right_eye = face_parts[__RIGHT_EYE_CLASS]
        left_eyebrow = face_parts[__LEFT_EYE_BROW_CLASS]
        right_eyebrow = face_parts[__RIGHT_EYE_BROW_CLASS]

        mixed = left_eye + right_eye + left_eyebrow + right_eyebrow

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=left_eye.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_eyes_and_nose(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        left_eye = face_parts[__LEFT_EYE_CLASS]
        right_eye = face_parts[__RIGHT_EYE_CLASS]
        nose = face_parts[__NOSE_CLASS]

        mixed = left_eye + right_eye + nose

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=left_eye.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None


def get_full_face(face_parts: dict, use_hog_proportion=False, no_blank=False):
    try:
        face = face_parts[__FACE_CLASS]
        left_eye = face_parts[__LEFT_EYE_CLASS]
        right_eye = face_parts[__RIGHT_EYE_CLASS]
        left_eyebrow = face_parts[__LEFT_EYE_BROW_CLASS]
        right_eyebrow = face_parts[__RIGHT_EYE_BROW_CLASS]
        nose = face_parts[__NOSE_CLASS]
        upper_lip = face_parts[__UPPER_LIP_CLASS]
        lower_lip = face_parts[__LOWER_LIP_CLASS]

        mixed = face
        mixed += left_eye
        mixed += right_eye
        mixed += left_eyebrow
        mixed += right_eyebrow
        mixed += nose
        mixed += upper_lip
        mixed += lower_lip

        if no_blank:
            tmp_img = Image.fromarray(mixed)
            tmp_img = tmp_img.resize(size=face.shape[:2])
            return np.array(tmp_img)

        return crop_roi(img_array=mixed, use_hog_proportion=use_hog_proportion)
    except Exception:
        return None
