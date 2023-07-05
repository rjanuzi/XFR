from pathlib import Path
from time import time

import dataset as ds
from BiSeNet.face_seg import segment_images
from util._telegram import send_simple_message

# Get the list of imgs to segment, except the already segmented ones
aligned_dataset = ds.get_aligned_imgs_dataset_index()
seg_map_dataset = ds.get_seg_maps_dataset_index()

# Select only rows that haven't been segmented yet
aligned_dataset = aligned_dataset.merge(
    seg_map_dataset["img_path"],
    on="img_path",
    how="left",
    indicator=True,
)
aligned_dataset = aligned_dataset.loc[aligned_dataset["_merge"] == "left_only"]

# Generate the paths to save the segmentation maps and segmented images
aligned_dataset["segmentation_map_path"] = aligned_dataset.img_path.apply(
    lambda path: path.replace(ds.DATASET_KIND_ALIGNED, ds.DATASET_KIND_SEG_MAP)
)

aligned_dataset["segmentation_segmented_path"] = aligned_dataset.img_path.apply(
    lambda path: path.replace(ds.DATASET_KIND_ALIGNED, ds.DATASET_KIND_SEG_IMG)
)

# Create output folders
_ = aligned_dataset.segmentation_map_path.apply(
    lambda path: Path(path).parent.mkdir(parents=True, exist_ok=True)
)
_ = aligned_dataset.segmentation_segmented_path.apply(
    lambda path: Path(path).parent.mkdir(parents=True, exist_ok=True)
)

send_simple_message(f"Starting segmentation of {aligned_dataset.shape[0]} images.")

start_time = time()
try:
    segment_images(
        input_path_lst=aligned_dataset["img_path"].tolist(),
        output_maps_path_lst=aligned_dataset.segmentation_map_path.tolist(),
        output_imgs_path_lst=aligned_dataset.segmentation_segmented_path.tolist(),
        save_imgs=True,
    )
    send_simple_message("Segmentation finished")
except Exception as e:
    send_simple_message("Some error occurred while segmenting images")
    raise e

send_simple_message(f"Segmentation finished in {round(time() - start_time)} seconds")
