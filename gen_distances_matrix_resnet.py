import json
import logging
import traceback
from pathlib import Path

from dataset import DATASET_KIND_ALIGNED, DATASET_KIND_STR, gen_dataset_index
from fr.face_decomposition import decompose_face
from fr.resnet_descriptor import gen_resnet_distances, gen_resnet_faceparts_distances
from util._telegram import send_simple_message

logging.basicConfig(
    filename="gen_distances_matrix_resenet.log",
    format="%(name)s - %(levelname)s - %(message)s",
)

__NAMES_TO_CALCULATE_DISTANCES_PATH = Path("names_to_calculate_distances.json")


def get_names_to_calculate(how_many_imgs_by_person=5):

    # Recover from file if it exists
    if __NAMES_TO_CALCULATE_DISTANCES_PATH.exists():
        with open(__NAMES_TO_CALCULATE_DISTANCES_PATH, "r") as f:
            names = json.load(f)
            send_simple_message(f"{len(names)} names recovered from file")
            return names
    else:
        send_simple_message("Generating names list to calculate distances.")
        # Get aligned images and generate person index based on file name (<person>_<img_id>)
        dataset_idx = gen_dataset_index(kind=DATASET_KIND_STR.get(DATASET_KIND_ALIGNED))
        dataset_idx["person"] = dataset_idx.name.apply(lambda n: n.split("_")[0])
        dataset_idx["name_sub"] = dataset_idx.name.apply(
            lambda n: "_".join(n.split("_")[1:])
        )

        # Calculate the number of face elements visible in each image
        dataset_idx["face_parts_count"] = 0
        count = 0
        for idx, row in dataset_idx.iterrows():
            tmp_img_name = row["name"]
            try:
                tmp_face_parts = decompose_face(tmp_img_name)
            except Exception:
                pass
            else:
                dataset_idx.loc[idx, "face_parts_count"] = len(tmp_face_parts)

            count += 1
            if count % 1e4 == 0:
                send_simple_message(
                    f"{count} faces evaluated. {count} / {len(dataset_idx)} -- {round(count / len(dataset_idx) * 100, 2)}%"
                )

        dataset_idx.sort_values(by=["face_parts_count"], ascending=False, inplace=True)

        # Group images by person and sort by face elements (we are looking for images with the higher number of face elements)
        grouped_ds = (
            dataset_idx.groupby(by=["person", "name"])
            .sum()
            .sort_values(by=["face_parts_count"], ascending=False)
        )

        # List all different persons
        persons = dataset_idx["person"].unique()

        # Generate the list of top N images for each person, considering the images were the most visible face elements
        send_simple_message("Generating names list.")
        names = []
        count = 0
        for person in persons:
            names += grouped_ds.loc[person].head(how_many_imgs_by_person).index.tolist()

        # Dump to file to avoid recalculation
        with open(__NAMES_TO_CALCULATE_DISTANCES_PATH, "w") as f:
            json.dump(names, f)

        send_simple_message(f"Names list generated. ({len(names)})")

        return names


if __name__ == "__main__":
    try:
        imgs_names = get_names_to_calculate()

        logging.info("Starting ResNET distances calculation")
        send_simple_message("Starting ResNET distances calculation")
        gen_resnet_distances(imgs_names=imgs_names)
        send_simple_message("Gen ResNET distances matrix done!")

        logging.info("Starting ResNET Faceparts distances calculation")
        send_simple_message("Starting ResNET Faceparts distances calculation")
        gen_resnet_faceparts_distances(imgs_names=imgs_names)
        send_simple_message("Gen ResNET Faceparts distances matrix done!")
    except:
        send_simple_message("Error generating ResNET distances matrix.")
        logging.error(traceback.format_exc())
