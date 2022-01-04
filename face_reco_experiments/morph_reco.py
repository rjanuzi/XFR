import json
import traceback
from datetime import datetime
from pathlib import Path
from time import time
from typing import Iterable

from dataset import (
    DATASET_KIND_ALIGNED,
    DATASET_KIND_MORPH,
    get_file_path,
    ls_imgs_names,
)
from fr.ifr import IFr

MORPH_IDX_COUNT = 19


def generate_fr_results(names_pairs: Iterable, fr: IFr) -> list:
    results = []
    tmp_file_results = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_tmp_results.json"

    count_done = 0
    start_time = time()
    for name_1, name_2 in names_pairs:
        try:
            loop_time = time()
            tmp_result = {
                "name_1": name_1,
                "name_2": name_2,
                "name_1_ref_distancy": [],
                "name_2_ref_distancy": [],
            }

            # Generate results against first person
            ref_person = get_file_path(name_1, DATASET_KIND_ALIGNED, ".png")
            for idx in range(MORPH_IDX_COUNT):
                tmp_other = get_file_path(
                    f"{name_1}_morph_{name_2}_{idx}", DATASET_KIND_MORPH, ".png"
                )
                distancy = fr.calc_distance(ref_person, tmp_other)
                tmp_result["name_1_ref_distancy"].append(distancy)

            # Generate results against second person
            ref_person = get_file_path(name_2, DATASET_KIND_ALIGNED, ".png")
            for idx in range(MORPH_IDX_COUNT):
                tmp_other = get_file_path(
                    f"{name_1}_morph_{name_2}_{idx}", DATASET_KIND_MORPH, ".png"
                )
                distancy = fr.calc_distance(ref_person, tmp_other)
                tmp_result["name_2_ref_distancy"].append(distancy)

            results.append(tmp_result)

            count_done += 1
            if count_done % 5 == 0:
                print(
                    f"FR Experiment done to {count_done} done. | Elapsed time: {int(time() - start_time)}s | Step time: {int(time() - loop_time)}s"
                )
                json.dump(results, open(tmp_file_results, "w"))
        except:
            print(f"Error on {name_1} and {name_2}")
            print(traceback.format_exc())

    return results


def run_fr_experiment(ifr: IFr, output_file_path: Path) -> None:
    def make_pairs_generator(files_names: Iterable):
        for file_name in files_names:
            if file_name.endswith("_0"):  # Let's use only the first morph as reference
                tmp_file_name = file_name.replace("_0", "")
                yield tmp_file_name.split("_morph_")

    morphed_files_names = ls_imgs_names(DATASET_KIND_MORPH)

    results = generate_fr_results(make_pairs_generator(morphed_files_names), ifr)

    json.dump(results, output_file_path.open("w"))
