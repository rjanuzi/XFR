import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from experiments.dlib_resnet_ga_approximation import calc_rank

# Read params
EXPERIMENT_ID = int(sys.argv[1])
CLUSTER_ID = int(sys.argv[2])
try:
    EXPERIMENT_FOLDER_NAME = sys.argv[3]
except:
    EXPERIMENT_FOLDER_NAME = "20230316230459_results_nb"

print(
    f"Checking experiment {EXPERIMENT_ID} in folder {EXPERIMENT_FOLDER_NAME} for cluster {CLUSTER_ID}"
)

EXPERIMENT_ROOT_FOLDER = Path("experiments", EXPERIMENT_FOLDER_NAME)
EXPERIMENTS_SUMMARY = EXPERIMENT_ROOT_FOLDER.joinpath("experiments_nb.csv")
EXPERIMENT_FOLDER = EXPERIMENT_ROOT_FOLDER.joinpath(
    f"{str(EXPERIMENT_ID).zfill(5)}_individuals"
)

DLIB_DISTANCES_FILE = Path("fr", "distances_dlib.json")
RESNET_DISTANCES_FILE = Path("fr", "distances_resnet.json")
RESNET_FACEPARTS_DISTANCES_FILE = Path("fr", "distances_resnet_faceparts_nb.json")
DLIB_DATASET_CLUSTERS_FILE = Path("fr", "dlib_clusters.json")

DLIB_RESNET_BEST_COMB = EXPERIMENT_FOLDER.joinpath("best_individual.json")
DLIB_RESNET_BEST_COMBS = EXPERIMENT_FOLDER.joinpath("best_individuals.json")

PRECISION_RECALL_FILE_PATH = EXPERIMENT_FOLDER.joinpath(f"precision_recall.json")


# Output Files
CHECK_RANK_PREPEND = (
    f"CHECK_RANK_{EXPERIMENT_FOLDER_NAME}_{EXPERIMENT_ID}_{CLUSTER_ID}_"
)


def prepare_distances():
    tmp_raw_data = json.load(open(DLIB_DISTANCES_FILE, "r"))
    dlib_distances = pd.DataFrame(
        dict(
            pair=tmp_raw_data.keys(),
            dlib_distance=(d["dlib"] for d in tmp_raw_data.values()),
        )
    )
    del tmp_raw_data

    # ResNET Distances ({<pair>: distance})
    tmp_raw_data = json.load(open(RESNET_DISTANCES_FILE, "r"))
    resnet_distances = pd.DataFrame(
        dict(pair=tmp_raw_data.keys(), resnet_distance=tmp_raw_data.values())
    )
    del tmp_raw_data

    # ResNET Faceparts Distances
    def rows_generator(resnet_faceparts_raw_data):
        for pair, distances in resnet_faceparts_raw_data.items():
            distances.update({"pair": pair})
            yield distances

    tmp_raw_data = json.load(open(RESNET_FACEPARTS_DISTANCES_FILE, "r"))
    generator = rows_generator(tmp_raw_data)
    del tmp_raw_data

    resnet_faceparts_distances = pd.DataFrame(generator)

    # Join distances into a sigle dataframe
    distances = dlib_distances.merge(resnet_distances, on="pair", how="outer")
    distances = distances.merge(resnet_faceparts_distances, on="pair", how="outer")

    # Filter only images with "n" (from VGGFACE2)
    distances = distances[distances.pair.apply(lambda p: "n" in p)]

    # Generate extra columns
    distances["img1"] = distances.pair.apply(lambda p: p.split(" x ")[0])
    distances["img2"] = distances.pair.apply(lambda p: p.split(" x ")[1])
    distances["person1"] = distances.img1.apply(lambda p: p.split("_")[0])
    distances["person2"] = distances.img2.apply(lambda p: p.split("_")[0])
    distances["same_person"] = (distances.person1 == distances.person2).apply(
        lambda s: "same" if s else "different"
    )

    # Delete unnecessary columns
    distances.drop(columns="pair", inplace=True)

    # Sort columns by name
    distances = distances.reindex(sorted(distances.columns), axis=1)

    # Load clusters
    if CLUSTER_ID is not None:
        clusters_ref = pd.DataFrame(
            data=json.load(open(DLIB_DATASET_CLUSTERS_FILE, "r"))
        )
        clusters_ref.set_index("label", inplace=True)

        distances["img1_cluster"] = distances.img1.apply(
            lambda i: clusters_ref.cluster.get(i, None)
        )
        distances["img2_cluster"] = distances.img2.apply(
            lambda i: clusters_ref.cluster.get(i, None)
        )

        distances = distances[
            (distances.img1_cluster == CLUSTER_ID)
            & (distances.img2_cluster == CLUSTER_ID)
        ]

        del clusters_ref

    # Normalize distances
    img1_cluster_bkp = distances.img1_cluster
    img2_cluster_bkp = distances.img2_cluster
    distances_num = distances.select_dtypes(include="number")
    for col in distances_num.columns:
        distances_num[col] = (distances_num[col] - distances_num[col].min()) / (
            distances_num[col].max() - distances_num[col].min()
        )

    distances[distances_num.columns] = distances_num
    distances.img1_cluster = img1_cluster_bkp
    distances.img2_cluster = img2_cluster_bkp

    del dlib_distances
    del resnet_distances
    del resnet_faceparts_distances
    del distances_num

    return distances


def get_resnet_comb_data(distances: pd.DataFrame):
    # Load best combination of dlib and resnet
    dlib_resnet_best_comb = json.load(open(DLIB_RESNET_BEST_COMB, "r"))
    dlib_resnet_best_comb = pd.DataFrame(
        dict(
            resnet_part=dlib_resnet_best_comb.keys(),
            multiplier=dlib_resnet_best_comb.values(),
        )
    )
    dlib_resnet_best_comb.sort_values(by="resnet_part", inplace=True)

    # Calculate resnet_combination column
    best_multipliers_lst = dlib_resnet_best_comb.multiplier.tolist()
    individual_sum = sum(best_multipliers_lst)
    best_multipliers_lst = [i / individual_sum for i in best_multipliers_lst]
    distances["resnet_comb"] = distances.loc[:, dlib_resnet_best_comb.resnet_part].dot(
        best_multipliers_lst
    )

    return (best_multipliers_lst, dlib_resnet_best_comb)


def generate_results(distances, best_multipliers_lst, dlib_resnet_best_comb):
    min_rank, max_rank, median_rank, mean_rank = calc_rank(
        individual=best_multipliers_lst,
        cluster_norm_distances=distances,
        resnet_distances_norm=distances.loc[:, dlib_resnet_best_comb.resnet_part],
        save_data=True,
        files_prepend=CHECK_RANK_PREPEND + datetime.now().strftime("%Y%m%d%H%M%S_"),
        use_scipy=False,
    )

    print(
        f"Min rank: {min_rank:.4f} | Max rank: {max_rank:.4f} | Median rank: {median_rank:.4f} | Mean rank: {mean_rank:.4f}"
    )


if __name__ == "__main__":
    print("Preparing distances data...")
    distances = prepare_distances()
    print("Done!\nPreparing resnet combination data...")
    best_multipliers_lst, dlib_resnet_best_comb = get_resnet_comb_data(distances)
    print("Done!\nGenerating results...")
    for _ in range(10):
        generate_results(distances, best_multipliers_lst, dlib_resnet_best_comb)
    print("Done!")
