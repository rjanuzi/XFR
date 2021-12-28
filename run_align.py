from dataset import (
    DATASET_KIND_ALIGNED,
    DATASET_KIND_RAW,
    DATASET_KIND_STR,
    gen_dataset_index,
    get_file_path,
)

from util.align_images import align_images

# Get the list of images to align, ignoring the already aligned ones
dataset_idx = gen_dataset_index()
raw_dataset = dataset_idx.loc[dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_RAW]]
aligned_dataset = dataset_idx.loc[
    dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_ALIGNED]
]

# Select only rows that are not in the aligned dataset
raw_dataset = raw_dataset.merge(aligned_dataset, on="name", how="left", indicator=True)
raw_dataset = raw_dataset.loc[raw_dataset["_merge"] == "left_only"]

# Based on the name of the persons, generate the expected folders structure to place the aligned images
output_folder_paths = (
    raw_dataset["name"]
    .apply(
        lambda name: get_file_path(
            name=name, dataset_kind=DATASET_KIND_ALIGNED, file_extension=".png"
        )
    )
    .tolist()
)

align_images(
    imgs_path_lst=raw_dataset["img_path_x"].tolist(),
    output_path_lst=output_folder_paths,
)
