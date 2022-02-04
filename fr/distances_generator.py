import json
from math import inf
from pathlib import Path
from time import time

import numpy as np
from dataset import DATASET_KIND_ALIGNED, ls_imgs_paths
from util._telegram import send_simple_message

from fr.dlib import DlibFr
from fr.hog_descriptor import calc_face_hog, compare_face_hogs

__DISTANCES_INDEX_PATH = Path("fr", "distances_index.json")
__FEATURES_MAPS_PATH = Path("fr", "features_maps.json")
__HOG_DATA_PATH = Path("fr", "hog_data.json")

__DLIB_KEY = "dlib"
__HOG_KEY = "hog"


def get_distances_idx():
    try:
        return json.load(open(__DISTANCES_INDEX_PATH, "r"))
    except FileNotFoundError:
        distancies = {}
        json.dump(distancies, open(__DISTANCES_INDEX_PATH, "w"))
        return distancies


def update_distances_idx(new_distances_idx):
    json.dump(new_distances_idx, open(__DISTANCES_INDEX_PATH, "w"))


def get_features_maps():
    try:
        return json.load(open(__FEATURES_MAPS_PATH, "r"))
    except FileNotFoundError:
        features_maps = {}
        json.dump(features_maps, open(__FEATURES_MAPS_PATH, "w"))
        return features_maps


def update_features_maps(new_features_maps):
    json.dump(new_features_maps, open(__FEATURES_MAPS_PATH, "w"))


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
    distancies_idx = get_distances_idx()
    features_maps = get_features_maps()
    aligned_imgs_paths = ls_imgs_paths(kind=DATASET_KIND_ALIGNED)
    dlib_fr = DlibFr()

    calculated_distances = 0
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
    start_loop_time = time()
    for path1 in aligned_imgs_paths:
        for path2 in aligned_imgs_paths:
            if path1 == path2:
                continue  # skip same image

            tmp_p1 = Path(path1)
            tmp_p2 = Path(path2)
            name_1 = tmp_p1.stem
            name_2 = tmp_p2.stem
            tmp_key_1 = f"{name_1} x {name_2}"
            tmp_key_2 = f"{name_2} x {name_1}"

            tmp_distances = distancies_idx.get(tmp_key_1, {})
            if not tmp_distances:
                tmp_distances = distancies_idx.get(tmp_key_2, {})

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

                distancies_idx[tmp_key_1] = tmp_distances

            calculated_distances += 1
            if calculated_distances % 1e4 == 0:
                # Backup
                update_distances_idx(distancies_idx)
                update_features_maps(features_maps)
                print(
                    f"Calculating DLIB distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} | Loop time: {round((time() - start_loop_time)/5e3, 4)}s"
                )
                send_simple_message(
                    f"Calculating DLIB distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} | Loop time: {round((time() - start_loop_time)/5e3, 4)}s"
                )
                start_loop_time = time()

    update_distances_idx(distancies_idx)
    update_features_maps(features_maps)
    print(f"DLIB Disances calculation done. Total time: {int(time() - start_time)}")
    send_simple_message(
        f"DLIB Disances calculation done. Total time: {int(time() - start_time)}"
    )
    start_loop_time = time()


def gen_hog_distances():
    distancies_idx = get_distances_idx()
    hog_data = get_hog_data()
    aligned_imgs_paths = ls_imgs_paths(kind=DATASET_KIND_ALIGNED)

    calculated_distances = 0
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
    start_loop_time = time()
    for path1 in aligned_imgs_paths:
        for path2 in aligned_imgs_paths:
            if path1 == path2:
                continue  # skip same image

            tmp_p1 = Path(path1)
            tmp_p2 = Path(path2)
            name_1 = tmp_p1.stem
            name_2 = tmp_p2.stem
            tmp_key_1 = f"{name_1} x {name_2}"
            tmp_key_2 = f"{name_2} x {name_1}"

            tmp_distances = distancies_idx.get(tmp_key_1, {})
            if not tmp_distances:
                tmp_distances = distancies_idx.get(tmp_key_2, {})

            hog_distance = tmp_distances.get(__HOG_KEY, None)
            if hog_distance is None:
                img1_features = hog_data.get(name_1, None)
                if img1_features is None:
                    try:
                        img1_features = calc_face_hog(name_1)
                    except (ValueError, KeyError):
                        tmp_distances[__HOG_KEY] = inf
                        continue  # Ignore images with no faces

                    hog_data[name_1] = {__HOG_KEY: img1_features.tolist()}
                elif img1_features.get(__HOG_KEY, None) is None:
                    try:
                        img1_features[__HOG_KEY] = calc_face_hog(name_1).tolist()
                    except (ValueError, KeyError):
                        tmp_distances[__HOG_KEY] = inf
                        continue  # Ignore images with no faces
                else:
                    img1_features = img1_features[__HOG_KEY]

                img2_features = hog_data.get(name_2, None)
                if img2_features is None:
                    try:
                        img2_features = calc_face_hog(name_2)
                    except (ValueError, KeyError):
                        tmp_distances[__HOG_KEY] = inf
                        continue  # Ignore images with no faces
                    hog_data[name_2] = {__HOG_KEY: img2_features.tolist()}
                elif img2_features.get(__HOG_KEY, None) is None:
                    try:
                        img2_features[__HOG_KEY] = calc_face_hog(name_2).tolist()
                    except (ValueError, KeyError):
                        tmp_distances[__HOG_KEY] = inf
                        continue  # Ignore images with no faces
                else:
                    img2_features = img2_features[__HOG_KEY]

                tmp_distances[__HOG_KEY] = compare_face_hogs(
                    face_hog_1=np.asarray(img1_features),
                    face_hog_2=np.asarray(img2_features),
                )

                distancies_idx[tmp_key_1] = tmp_distances

            calculated_distances += 1
            if calculated_distances % 1e4 == 0:
                # Backup
                update_distances_idx(distancies_idx)
                update_features_maps(hog_data)
                print(
                    f"Calculating HOG distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} | Loop time: {round((time() - start_loop_time)/5e3, 4)}s"
                )
                send_simple_message(
                    f"Calculating HOG distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} | Loop time: {round((time() - start_loop_time)/5e3, 4)}s"
                )
                start_loop_time = time()

    update_distances_idx(distancies_idx)
    update_features_maps(hog_data)
    print(f"HOG Distances calculation done. Total time: {int(time() - start_time)}")
    send_simple_message(
        f"HOG Distances calculation done. Total time: {int(time() - start_time)}"
    )
