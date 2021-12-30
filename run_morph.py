from img_factory.imgs_morph import morph
from util.gif import imgs_to_gif

from dataset import (
    DATASET_KIND_LATENTS,
    DATASET_KIND_MASKS,
    DATASET_KIND_STR,
    DATASET_KIND_MORPH,
    gen_dataset_index,
    get_file_path,
)

# Get the list of images to align, ignoring the already aligned ones
dataset_idx = gen_dataset_index()
latents_dataset = dataset_idx.loc[
    dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_LATENTS]
]

# Split the dataset at mid point (we want to morph the imgs at index n with n+mid)
mid = int(len(latents_dataset) / 2)
latents_dataset_1 = latents_dataset.iloc[:mid].reset_index(drop=True)
latents_dataset_2 = latents_dataset.iloc[mid:].reset_index(drop=True)

latents_to_morph = latents_dataset_1.join(
    other=latents_dataset_2, how="left", lsuffix="_1", rsuffix="_2"
)

for idx, row in latents_to_morph.iterrows():
    source_a_name = row.name_1
    source_b_name = row.name_2
    mask_to_apply = source_a_name

    generated_imgs = morph(
        name_1=source_a_name, name_2=source_b_name, mask_to_apply=mask_to_apply
    )

    for idx, generated_img in enumerate(generated_imgs):
        generated_img.save(
            get_file_path(
                f"{source_a_name}_morph_{source_b_name}_{idx}",
                DATASET_KIND_MORPH,
                ".png",
            )
        )

    imgs_to_gif(
        generated_imgs,
        get_file_path(
            f"{source_a_name}_morph_{source_b_name}",
            DATASET_KIND_MORPH,
            ".gif",
        ),
    )
