from time import time

from BiSeNet.face_seg import segment_images
from dataset import (
    DATASET_KIND_ALIGNED,
    DATASET_KIND_SEG_IMG,
    DATASET_KIND_SEG_MAP,
    DATASET_KIND_STR,
    gen_dataset_index,
    get_file_path,
)
from util._telegram import send_simple_message

# Get the list of images to align, ignoring the already aligned ones
dataset_idx = gen_dataset_index()
aligned_dataset = dataset_idx.loc[
    dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_ALIGNED]
]
seg_map_dataset = dataset_idx.loc[
    dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_SEG_MAP]
]

# Select only rows that are not in the aligned dataset
aligned_dataset = aligned_dataset.merge(
    seg_map_dataset, on="name", how="left", indicator=True
)
aligned_dataset = aligned_dataset.loc[aligned_dataset["_merge"] == "left_only"]

# Based on the name of the persons, generate the expected folders structure to place the aligned images
output_maps_path_lst = (
    aligned_dataset["name"]
    .apply(
        lambda name: get_file_path(
            name=name, dataset_kind=DATASET_KIND_SEG_MAP, file_extension=".npy"
        )
    )
    .tolist()
)

output_imgs_path_lst = (
    aligned_dataset["name"]
    .apply(
        lambda name: get_file_path(
            name=name, dataset_kind=DATASET_KIND_SEG_IMG, file_extension=".png"
        )
    )
    .tolist()
)

send_simple_message(f"Starting segmentation of {len(output_maps_path_lst)} images.")

start_time = time()
try:
    segment_images(
        input_path_lst=aligned_dataset["img_path_x"].tolist(),
        output_maps_path_lst=output_maps_path_lst,
        output_imgs_path_lst=output_imgs_path_lst,
        save_imgs=True,
    )
    send_simple_message("Segmentation finished")
except Exception as e:
    send_simple_message("Some error occurred while segmenting images")
    raise e

send_simple_message(f"Segmentation finished in {round(time() - start_time)} seconds")
