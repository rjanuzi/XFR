import json
from math import inf
from pathlib import Path
from time import time

import numpy as np
from dataset import DATASET_KIND_ALIGNED, ls_imgs_paths
from util._telegram import send_simple_message

from fr.dlib import DlibFr
from fr.face_decomposition import decompose_face
from fr.hog_descriptor import (
    HOG_OPT_ALL,
    HOG_OPT_EARS,
    HOG_OPT_EYEBROWS,
    HOG_OPT_EYES,
    HOG_OPT_FACE,
    HOG_OPT_LEFT_EAR,
    HOG_OPT_LEFT_EYE,
    HOG_OPT_LEFT_EYEBROW,
    HOG_OPT_LOWER_LIP,
    HOG_OPT_MOUTH,
    HOG_OPT_NOSE,
    HOG_OPT_RIGHT_EAR,
    HOG_OPT_RIGHT_EYE,
    HOG_OPT_RIGHT_EYEBROW,
    HOG_OPT_UPPER_LIP,
    calc_hog,
    compare_hogs,
)

__DISTANCES_PATH = Path("fr", "distances.json")
__DLIB_DATA_PATH = Path("fr", "dlib_data.json")
__HOG_DATA_PATH = Path("fr", "hog_data.json")

__DLIB_KEY = "dlib"
__HOG_KEY = "hog_all"
__HOG_FACE_KEY = "hog_face"
__HOG_LEFT_EYE_KEY = "hog_left_eye"
__HOG_RIGHT_EYE_KEY = "hog_right_eye"
__HOG_EYES_KEY = "hog_eyes"
__HOG_EYEBROWS_KEY = "hog_eyebrows"
__HOG_LEFT_EYEBROW_KEY = "hog_left_eyebrow"
__HOG_RIGHT_EYEBROW_KEY = "hog_right_eyebrow"
__HOG_EARS_KEY = "hog_ears"
__HOG_LEFT_EAR_KEY = "hog_left_ear"
__HOG_RIGHT_EAR_KEY = "hog_right_ear"
__HOG_NOSE_KEY = "hog_nose"
__HOG_LOWER_LIP_KEY = "hog_lower_lip"
__HOG_UPPER_LIP_KEY = "hog_upper_lip"
__HOG_MOUTH_KEY = "hog_mouth"
__HOG_KEY_TO_OPT = {
    __HOG_KEY: HOG_OPT_ALL,
    __HOG_FACE_KEY: HOG_OPT_FACE,
    __HOG_LEFT_EYE_KEY: HOG_OPT_LEFT_EYE,
    __HOG_RIGHT_EYE_KEY: HOG_OPT_RIGHT_EYE,
    __HOG_EYES_KEY: HOG_OPT_EYES,
    __HOG_EYEBROWS_KEY: HOG_OPT_EYEBROWS,
    __HOG_LEFT_EYEBROW_KEY: HOG_OPT_LEFT_EYEBROW,
    __HOG_RIGHT_EYEBROW_KEY: HOG_OPT_RIGHT_EYEBROW,
    __HOG_EARS_KEY: HOG_OPT_EARS,
    __HOG_LEFT_EAR_KEY: HOG_OPT_LEFT_EAR,
    __HOG_RIGHT_EAR_KEY: HOG_OPT_RIGHT_EAR,
    __HOG_NOSE_KEY: HOG_OPT_NOSE,
    __HOG_LOWER_LIP_KEY: HOG_OPT_LOWER_LIP,
    __HOG_UPPER_LIP_KEY: HOG_OPT_UPPER_LIP,
    __HOG_MOUTH_KEY: HOG_OPT_MOUTH,
}


def get_distances():
    try:
        return json.load(open(__DISTANCES_PATH, "r"))
    except FileNotFoundError:
        distancies = {}
        json.dump(distancies, open(__DISTANCES_PATH, "w"))
        return distancies


def update_distances(new_distances_idx):
    json.dump(new_distances_idx, open(__DISTANCES_PATH, "w"))


def get_dlib_data():
    try:
        return json.load(open(__DLIB_DATA_PATH, "r"))
    except FileNotFoundError:
        features_maps = {}
        json.dump(features_maps, open(__DLIB_DATA_PATH, "w"))
        return features_maps


def update_dlib_data(new_features_maps):
    json.dump(new_features_maps, open(__DLIB_DATA_PATH, "w"))


def get_hog_data():
    try:
        return json.load(open(__HOG_DATA_PATH, "r"))
    except FileNotFoundError:
        hog_data = {}
        json.dump(hog_data, open(__HOG_DATA_PATH, "w"))
        return hog_data


def update_hog_data(new_hog_data):
    json.dump(new_hog_data, open(__HOG_DATA_PATH, "w"))


def gen_dlib_distances():
    distances_idx = get_distances()
    features_maps = get_dlib_data()
    aligned_imgs_paths = ls_imgs_paths(kind=DATASET_KIND_ALIGNED)
    dlib_fr = DlibFr()

    calculated_distances = 0
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
    start_loop_time = time()
    for path1 in aligned_imgs_paths:
        for path2 in aligned_imgs_paths:
            if path1 == path2:
                calculated_distances += 1
                continue  # skip same image

            tmp_p1 = Path(path1)
            tmp_p2 = Path(path2)
            name_1 = tmp_p1.stem
            name_2 = tmp_p2.stem
            tmp_key_1 = f"{name_1} x {name_2}"
            tmp_key_2 = f"{name_2} x {name_1}"

            tmp_distances = distances_idx.get(tmp_key_1, {})
            if not tmp_distances:
                tmp_distances = distances_idx.get(tmp_key_2, {})

            dlib_distance = tmp_distances.get(__DLIB_KEY, None)
            if dlib_distance is None:
                img1_features = features_maps.get(name_1, None)
                if img1_features is None:
                    img1_features = dlib_fr.gen_features(path1)
                    features_maps[name_1] = {__DLIB_KEY: img1_features.tolist()}
                elif img1_features.get(__DLIB_KEY, None) is None:
                    img1_features[__DLIB_KEY] = dlib_fr.gen_features(path1).tolist()
                else:
                    img1_features = img1_features[__DLIB_KEY]

                img2_features = features_maps.get(name_2, None)
                if img2_features is None:
                    img2_features = dlib_fr.gen_features(path2)
                    features_maps[name_2] = {__DLIB_KEY: img2_features.tolist()}
                elif img2_features.get(__DLIB_KEY, None) is None:
                    img2_features[__DLIB_KEY] = dlib_fr.gen_features(path2).tolist()
                else:
                    img2_features = img2_features[__DLIB_KEY]

                tmp_distances[__DLIB_KEY] = dlib_fr.calc_distance_from_features(
                    img1_features=np.asarray(img1_features),
                    img2_features=np.asarray(img2_features),
                )

                distances_idx[tmp_key_1] = tmp_distances

            calculated_distances += 1
            if calculated_distances % 5e2 == 0:
                # Backup
                update_distances(distances_idx)
                update_dlib_data(features_maps)
                print(
                    f"Calculating DLIB distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Loop time: {round((time() - start_loop_time)/5e3, 4)} s"
                )
            if calculated_distances % 5e4 == 0:
                send_simple_message(
                    f"Calculating DLIB distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Loop time: {round((time() - start_loop_time)/5e3, 4)} s"
                )
            start_loop_time = time()

    update_distances(distances_idx)
    update_dlib_data(features_maps)
    print(f"DLIB Disances calculation done. Total time: {int(time() - start_time)}")
    send_simple_message(
        f"DLIB Disances calculation done. Total time: {int(time() - start_time)}"
    )


def gen_hog_distances():
    distances_idx = get_distances()
    hog_data = get_hog_data()
    aligned_imgs_paths = ls_imgs_paths(kind=DATASET_KIND_ALIGNED)

    calculated_distances = 0
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
    start_loop_time = time()
    for path1 in aligned_imgs_paths:
        for path2 in aligned_imgs_paths:
            if path1 == path2:
                calculated_distances += 1
                continue  # skip same image

            tmp_p1 = Path(path1)
            tmp_p2 = Path(path2)
            name_1 = tmp_p1.stem
            name_2 = tmp_p2.stem
            tmp_key_1 = f"{name_1} x {name_2}"
            tmp_key_2 = f"{name_2} x {name_1}"

            tmp_distances = distances_idx.get(tmp_key_1, {})
            if not tmp_distances:
                tmp_distances = distances_idx.get(tmp_key_2, {})

            hog_distance = tmp_distances.get(__HOG_KEY, None)
            if hog_distance is None:
                img1_features = hog_data.get(name_1, None)
                if img1_features is None:
                    # Calculate hog features
                    face_parts = decompose_face(name_1)
                    all_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_ALL)
                    face_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_FACE)
                    left_eye_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EYE
                    )
                    right_eye_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EYE
                    )
                    eyes_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_EYES)
                    left_eyebrow_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EYEBROW
                    )
                    right_eyebrow_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EYEBROW
                    )
                    eyebrows_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_EYEBROWS
                    )
                    left_ear_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EAR
                    )
                    right_ear_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EAR
                    )
                    ears_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_EARS)
                    nose_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_NOSE)
                    lower_lip_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LOWER_LIP
                    )
                    upper_lip_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_UPPER_LIP
                    )
                    mouth_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_MOUTH)

                    # Save HOG features
                    hog_data[name_1] = {
                        __HOG_KEY: all_features.tolist()
                        if all_features is not None
                        else False,
                        __HOG_FACE_KEY: face_features.tolist()
                        if face_features is not None
                        else False,
                        __HOG_LEFT_EYE_KEY: left_eye_features.tolist()
                        if left_eye_features is not None
                        else False,
                        __HOG_RIGHT_EYE_KEY: right_eye_features.tolist()
                        if right_eye_features is not None
                        else False,
                        __HOG_EYES_KEY: eyes_features.tolist(),
                        __HOG_LEFT_EYEBROW_KEY: left_eyebrow_features.tolist()
                        if left_eyebrow_features is not None
                        else False,
                        __HOG_RIGHT_EYEBROW_KEY: right_eyebrow_features.tolist()
                        if right_eyebrow_features is not None
                        else False,
                        __HOG_EYEBROWS_KEY: eyebrows_features.tolist()
                        if eyebrows_features is not None
                        else False,
                        __HOG_LEFT_EAR_KEY: left_ear_features.tolist()
                        if left_ear_features is not None
                        else False,
                        __HOG_RIGHT_EAR_KEY: right_ear_features.tolist()
                        if right_ear_features is not None
                        else False,
                        __HOG_EARS_KEY: ears_features.tolist()
                        if ears_features is not None
                        else False,
                        __HOG_NOSE_KEY: nose_features.tolist()
                        if nose_features is not None
                        else False,
                        __HOG_LOWER_LIP_KEY: lower_lip_features.tolist()
                        if lower_lip_features is not None
                        else False,
                        __HOG_UPPER_LIP_KEY: upper_lip_features.tolist()
                        if upper_lip_features is not None
                        else False,
                        __HOG_MOUTH_KEY: mouth_features.tolist()
                        if mouth_features is not None
                        else False,
                    }

                    img1_features = hog_data[name_1]

                img2_features = hog_data.get(name_2, None)
                if img2_features is None:
                    # Calculate hog features
                    face_parts = decompose_face(name_2)
                    all_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_ALL)
                    face_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_FACE)
                    left_eye_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EYE
                    )
                    right_eye_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EYE
                    )
                    eyes_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_EYES)
                    left_eyebrow_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EYEBROW
                    )
                    right_eyebrow_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EYEBROW
                    )
                    eyebrows_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_EYEBROWS
                    )
                    left_ear_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LEFT_EAR
                    )
                    right_ear_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_RIGHT_EAR
                    )
                    ears_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_EARS)
                    nose_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_NOSE)
                    lower_lip_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_LOWER_LIP
                    )
                    upper_lip_features = calc_hog(
                        face_parts=face_parts, opt=HOG_OPT_UPPER_LIP
                    )
                    mouth_features = calc_hog(face_parts=face_parts, opt=HOG_OPT_MOUTH)

                    # Save HOG features
                    hog_data[name_2] = {
                        __HOG_KEY: all_features.tolist()
                        if all_features is not None
                        else False,
                        __HOG_FACE_KEY: face_features.tolist()
                        if face_features is not None
                        else False,
                        __HOG_LEFT_EYE_KEY: left_eye_features.tolist()
                        if left_eye_features is not None
                        else False,
                        __HOG_RIGHT_EYE_KEY: right_eye_features.tolist()
                        if right_eye_features is not None
                        else False,
                        __HOG_EYES_KEY: eyes_features.tolist()
                        if eyes_features is not None
                        else False,
                        __HOG_LEFT_EYEBROW_KEY: left_eyebrow_features.tolist()
                        if left_eyebrow_features is not None
                        else False,
                        __HOG_RIGHT_EYEBROW_KEY: right_eyebrow_features.tolist()
                        if right_eyebrow_features is not None
                        else False,
                        __HOG_EYEBROWS_KEY: eyebrows_features.tolist()
                        if eyebrows_features is not None
                        else False,
                        __HOG_LEFT_EAR_KEY: left_ear_features.tolist()
                        if left_ear_features is not None
                        else False,
                        __HOG_RIGHT_EAR_KEY: right_ear_features.tolist()
                        if right_ear_features is not None
                        else False,
                        __HOG_EARS_KEY: ears_features.tolist()
                        if ears_features is not None
                        else False,
                        __HOG_NOSE_KEY: nose_features.tolist()
                        if nose_features is not None
                        else False,
                        __HOG_LOWER_LIP_KEY: lower_lip_features.tolist()
                        if lower_lip_features is not None
                        else False,
                        __HOG_UPPER_LIP_KEY: upper_lip_features.tolist()
                        if upper_lip_features is not None
                        else False,
                        __HOG_MOUTH_KEY: mouth_features.tolist()
                        if mouth_features is not None
                        else False,
                    }

                    img2_features = hog_data[name_2]

                for key, tmp_features_1 in img1_features.items():
                    tmp_features_2 = img2_features.get(key)
                    if tmp_features_1 == False or tmp_features_2 == False:
                        tmp_distances[key] = inf
                    else:
                        tmp_distances[key] = compare_hogs(
                            hog_1=np.asarray(tmp_features_1),
                            hog_2=np.asarray(tmp_features_2),
                            opt=__HOG_KEY_TO_OPT[key],
                        )

                distances_idx[tmp_key_1] = tmp_distances

            calculated_distances += 1
            if calculated_distances % 5e2 == 0:
                # Backup
                update_distances(distances_idx)
                update_hog_data(hog_data)
                print(
                    f"Calculating HOG distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Loop time: {round((time() - start_loop_time)/5e3, 4)} s"
                )

            if calculated_distances % 5e4 == 0:
                send_simple_message(
                    f"Calculating HOG distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Loop time: {round((time() - start_loop_time)/5e3, 4)} s"
                )

            start_loop_time = time()

    update_distances(distances_idx)
    update_hog_data(hog_data)
    print(f"HOG Distances calculation done. Total time: {int(time() - start_time)}")
    send_simple_message(
        f"HOG Distances calculation done. Total time: {int(time() - start_time)}"
    )
