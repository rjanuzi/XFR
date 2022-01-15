import json
from os import path
from pathlib import Path
from time import time

from dataset import DATASET_KIND_ALIGNED, ls_imgs_paths

from fr.dlib import DlibFr

__DISTANCES_INDEX_PATH = Path("fr", "distances_index.json")
__BATCH_SIZE = 100


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
    results = {}
    for ref_path in aligned_imgs_paths:
        for start_idx in range(0, len(aligned_imgs_paths), __BATCH_SIZE):
            paths = aligned_imgs_paths[start_idx : start_idx + __BATCH_SIZE]

            ref_path = Path(ref_path)
            paths = [Path(p) for p in paths]
            ref_name = ref_path.stem
            names = [p.stem for p in paths]

            distances = dlib_fr.calc_distances(
                ref_img_path=ref_path, imgs_to_compare=paths
            )
            for i, distancy in enumerate(distances):
                dlib_distance = dict(dlib=distancy)
                results[f"{ref_name}-{names[i]}"] = dlib_distance

            calculated_distances += __BATCH_SIZE
            json.dump(results, open(__DISTANCES_INDEX_PATH, "w"))
            print(
                f"Calculating distances... {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Loop time: {int((time() - start_time)/__BATCH_SIZE)}s"
            )
            start_time = time()


gen_dlib_distances()
