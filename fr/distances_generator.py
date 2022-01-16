import json
from os import path
from pathlib import Path
from time import time

from dataset import DATASET_KIND_ALIGNED, ls_imgs_paths

from fr.dlib import DlibFr

__DISTANCES_INDEX_PATH = Path("fr", "distances_index.json")


def get_distances_idx():
    try:
        return json.load(open(__DISTANCES_INDEX_PATH, "r"))
    except FileNotFoundError:
        distancies = {}
        json.dump(distancies, open(__DISTANCES_INDEX_PATH, "w"))
        return distancies


def gen_dlib_distances():
    distancies_idx = get_distances_idx()
    aligned_imgs_paths = ls_imgs_paths(kind=DATASET_KIND_ALIGNED)
    dlib_fr = DlibFr()

    calculated_distances = 0
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
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

            dlib_distance = tmp_distances.get("dlib", None)
            if dlib_distance is None:
                tmp_distances["dlib"] = dlib_fr.calc_distance(
                    img_path_1=path1, img_path_2=path2
                )

                distancies_idx[tmp_key_1] = tmp_distances

            calculated_distances += 1
            if calculated_distances % 10 == 0:
                # Backup
                json.dump(distancies_idx, open(__DISTANCES_INDEX_PATH, "w"))
                print(
                    f"Calculating distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Loop time: {int((time() - start_time)/10)}s"
                )
                start_time = time()


gen_dlib_distances()
