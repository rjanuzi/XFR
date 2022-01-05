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


def generate_fr_results(names_pairs: Iterable, fr: IFr, backup_file: Path) -> list:
    results = []
    tmp_file_results = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_bkp_results.json"

    bkup_names_dict = {}
    if backup_file:
        print(f"Using backup file: {backup_file}")
        # Let's create a dict for fast consults, where the key is the concatenation of the two names
        tmp_bkp = json.load(open(backup_file, "r"))
        for bkup_entry in tmp_bkp:
            tmp_key = f"{bkup_entry['name_1']+bkup_entry['name_2']}"
            bkup_names_dict[tmp_key] = bkup_entry
        print(f"{len(tmp_bkp)} entries loaded from backup file.")
    else:
        print("No backup file provided.")

    count_done = 0
    start_time = time()
    for name_1, name_2 in names_pairs:
        if backup_file:
            tmp_key = f"{name_1+name_2}"
            tmp_result = bkup_names_dict.get(tmp_key, False)
            if tmp_result:
                results.append(tmp_result)
                print(f"Getting from backup {name_1 + '_' + name_2}")
        else:
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
                if count_done % 10 == 0:
                    print(
                        f"FR Experiment done to {count_done} done. | Elapsed time: {int(time() - start_time)}s | Step time: {int(time() - loop_time)}s"
                    )
                    json.dump(results, open(tmp_file_results, "w"))
            except:
                print(f"Error on {name_1} and {name_2}")
                print(traceback.format_exc())

    return results


def run_fr_experiment(
    ifr: IFr, output_file_path: Path, backup_file: Path = None
) -> None:
    def make_pairs_generator(files_names: Iterable):
        for file_name in files_names:
            if file_name.endswith("_0"):  # Let's use only the first morph as reference
                tmp_file_name = file_name.replace("_0", "")
                yield tmp_file_name.split("_morph_")

    morphed_files_names = ls_imgs_names(DATASET_KIND_MORPH)

    results = generate_fr_results(
        make_pairs_generator(morphed_files_names), ifr, backup_file
    )

    print(f"Saving results at {output_file_path}")
    json.dump(results, open(output_file_path, "w"))
    print("Done.")
