from dataset import (
    DATASET_KIND_ALIGNED,
    DATASET_KIND_RAW,
    DATASET_KIND_STR,
    DATASET_VGGFACE2_FOLDER,
    gen_dataset_index,
    get_file_path,
)
from util._telegram import send_simple_message
from util.align_images import align_images

# ----------------------------------------------------------------------------------------------
# Align dataset
# ----------------------------------------------------------------------------------------------

# # Get the list of images to align, ignoring the already aligned ones
# dataset_idx = gen_dataset_index()
# raw_dataset = dataset_idx.loc[dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_RAW]]
# aligned_dataset = dataset_idx.loc[
#     dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_ALIGNED]
# ]

# # Select only rows that are not in the aligned dataset
# raw_dataset = raw_dataset.merge(aligned_dataset, on="name", how="left", indicator=True)
# raw_dataset = raw_dataset.loc[raw_dataset["_merge"] == "left_only"]

# # Based on the name of the persons, generate the expected folders structure to place the aligned images
# output_folder_paths = (
#     raw_dataset["name"]
#     .apply(
#         lambda name: get_file_path(
#             name=name, dataset_kind=DATASET_KIND_ALIGNED, file_extension=".png"
#         )
#     )
#     .tolist()
# )

# try:
#     align_images(
#         imgs_path_lst=raw_dataset["img_path_x"].tolist(),
#         output_path_lst=output_folder_paths,
#     )
#     send_simple_message("Alignment finished")
# except Exception as e:
#     send_simple_message("Some error occurred while aligning images")
#     raise e

# ----------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------
# Align VGGFace2
# ----------------------------------------------------------------------------------------------

vggface2_folder = DATASET_VGGFACE2_FOLDER
input_imgs_paths = []
output_imgs_paths = []
for person_folder in vggface2_folder.glob("*/"):
    person_id = person_folder.stem
    for img_path in person_folder.glob("*"):
        img_name = f"{person_id}_{img_path.stem}"
        input_imgs_paths.append(img_path)
        output_imgs_paths.append(
            get_file_path(
                name=img_name, dataset_kind=DATASET_KIND_ALIGNED, file_extension=".png"
            )
        )

try:
    align_images(
        imgs_path_lst=input_imgs_paths,
        output_path_lst=output_imgs_paths,
        output_size=256,
        transform_size=512,
    )
    send_simple_message("Alignment finished")
except Exception as e:
    send_simple_message("Some error occurred while aligning images")
    raise e

# ----------------------------------------------------------------------------------------------
