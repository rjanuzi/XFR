from dataset import raw_ds, processed_ds
from util.align_images import align_images

# Get the list of all the images in the dataset
raw_dataset = raw_ds.lookup_raw_imgs(poses=["normal"])
imgs_paths = raw_dataset["img_path"].tolist()

# Based on the name of the persons, generate the expected folders structure to place the aligned images
output_folder_paths = (
    raw_dataset["person_name"]
    .apply(
        lambda person_name: processed_ds.get_folder(
            person_name=person_name, folder_kind="aligned"
        )
    )
    .tolist()
)

align_images(
    imgs_path_lst=imgs_paths,
    output_folder_path_lst=output_folder_paths,
)
