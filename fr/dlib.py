from fr.ifr import IFr
from pathlib import Path
import numpy as np

import face_recognition

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

__DLIB_DISTANCE_FACE_WEIGHT = 1.0
__DLIB_DISTANCE_EYES_WEIGHT = 1.0
__DLIB_DISTANCE_EYEBROWS_WEIGHT = 1.0
__DLIB_DISTANCE_EARS_WEIGHT = 1.0
__DLIB_DISTANCE_NOSE_WEIGHT = 1.0
__DLIB_DISTANCE_MOUTH_WEIGHT = 1.0

DLIB_OPT_ALL = 0
DLIB_OPT_FACE = 1
DLIB_OPT_EYES = 2
DLIB_OPT_EYEBROWS = 3
DLIB_OPT_EARS = 4
DLIB_OPT_NOSE = 5
DLIB_OPT_MOUTH = 6
DLIB_OPT_LEFT_EYE = 7
DLIB_OPT_RIGHT_EYE = 8
DLIB_OPT_LEFT_EYEBROW = 9
DLIB_OPT_RIGHT_EYEBROW = 10
DLIB_OPT_LEFT_EAR = 11
DLIB_OPT_RIGHT_EAR = 12
DLIB_OPT_LOWER_LIP = 13
DLIB_OPT_UPPER_LIP = 14
DLIB_OPT_MOUTH_AND_NOSE = 15
DLIB_OPT_EYES_AND_EYEBROWS = 16
DLIB_OPT_EYES_AND_NOSE = 17
DLIB_OPT_FULL_FACE = 18

__calc_dlib_funcs = {
    DLIB_OPT_FACE: get_face,
    DLIB_OPT_LEFT_EYE: get_left_eye,
    DLIB_OPT_RIGHT_EYE: get_right_eye,
    DLIB_OPT_EYES: get_eyes,
    DLIB_OPT_LEFT_EYEBROW: get_left_eyebrow,
    DLIB_OPT_RIGHT_EYEBROW: get_rigth_eyebrow,
    DLIB_OPT_EYEBROWS: get_eyebrows,
    DLIB_OPT_LEFT_EAR: get_left_ear,
    DLIB_OPT_RIGHT_EAR: get_right_ear,
    DLIB_OPT_EARS: get_ears,
    DLIB_OPT_NOSE: get_nose,
    DLIB_OPT_LOWER_LIP: get_lower_lip,
    DLIB_OPT_UPPER_LIP: get_upper_lip,
    DLIB_OPT_MOUTH: get_mouth,
    DLIB_OPT_MOUTH_AND_NOSE: get_mouth_and_nose,
    DLIB_OPT_EYES_AND_EYEBROWS: get_eyes_and_eyebrows,
    DLIB_OPT_EYES_AND_NOSE: get_eyes_and_nose,
    DLIB_OPT_FULL_FACE: get_full_face,
}


class DlibFr(IFr):
    """
    Face recognition algorithm that uses DLIB's HOG + Linear SVM method
    """

    def gen_features(self, img_path: Path):
        try:
            tmp_img = face_recognition.load_image_file(img_path)
            return face_recognition.face_encodings(tmp_img)[0]
        except IndexError:
            print(f"Error reading features from {img_path}")
            return np.zeros(128)

    def calc_distance(self, img_path_1: Path, img_path_2: Path):
        features_1 = self.gen_features(img_path_1)
        features_2 = self.gen_features(img_path_2)

        results = face_recognition.face_distance([features_1], features_2)
        return results[0]

    def calc_distance_from_features(self, img1_features: list, img2_features: Path):
        results = face_recognition.face_distance([img1_features], img2_features)
        return results[0]

    def calc_distances(self, ref_img_path: Path, imgs_to_compare: Path):
        ref_features = self.gen_features(ref_img_path)
        features_batch = [self.gen_features(i) for i in imgs_to_compare]

        results = face_recognition.face_distance(features_batch, ref_features)
        return results

    def check(
        self, img_path_1: Path, img_path_2: Path, distance_tolerance: float = 0.6
    ):
        features_1 = self.gen_features(img_path_1)
        features_2 = self.gen_features(img_path_2)

        results = face_recognition.compare_faces(
            [features_1], features_2, tolerance=distance_tolerance
        )
        return results[0]

    def gen_facepart_features(self, face_parts: dict, opt: int):
        if opt == DLIB_OPT_ALL:
            face = get_face(face_parts)
            eyes = get_eyes(face_parts)
            eyebrows = get_eyebrows(face_parts)
            ears = get_ears(face_parts)
            nose = get_nose(face_parts)
            mouth = get_mouth(face_parts)

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

            face_dlib = face_recognition.face_encodings(face)[0]
            eyes_dlib = face_recognition.face_encodings(eyes)[0]
            eyebrows_dlib = face_recognition.face_encodings(eyebrows)[0]
            ears_dlib = face_recognition.face_encodings(ears)[0]
            nose_dlib = face_recognition.face_encodings(nose)[0]
            mouth_dlib = face_recognition.face_encodings(mouth)[0]

            seg_dlibs = np.vstack(
                (face_dlib, eyes_dlib, eyebrows_dlib, ears_dlib, nose_dlib, mouth_dlib)
            )

            return seg_dlibs
        else:
            # Use the function to get the face parts according to the option
            face_elements = __calc_dlib_funcs[opt](face_parts)
            if face_elements is None:
                return None
            else:
                face_elements_dlib = face_recognition.face_encodings(face_elements)[0]
                return face_elements_dlib

    def gen_faceparts_distance(self, dlib_1: np.array, dlib_2: np.array, opt: int):
        hog_abs_delta = np.abs(dlib_1 - dlib_2)
        if opt == DLIB_OPT_ALL:
            acc_weighted_delta = 0.0
            acc_weighted_delta += __DLIB_DISTANCE_FACE_WEIGHT * hog_abs_delta[0].sum()
            acc_weighted_delta += __DLIB_DISTANCE_EYES_WEIGHT * hog_abs_delta[1].sum()
            acc_weighted_delta += (
                __DLIB_DISTANCE_EYEBROWS_WEIGHT * hog_abs_delta[2].sum()
            )
            acc_weighted_delta += __DLIB_DISTANCE_EARS_WEIGHT * hog_abs_delta[3].sum()
            acc_weighted_delta += __DLIB_DISTANCE_NOSE_WEIGHT * hog_abs_delta[4].sum()
            acc_weighted_delta += __DLIB_DISTANCE_MOUTH_WEIGHT * hog_abs_delta[5].sum()

            return acc_weighted_delta
        else:
            return hog_abs_delta.sum()
