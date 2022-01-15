import traceback
from time import time

from dataset import (
    DATASET_KIND_ALIGNED,
    DATASET_KIND_LATENTS,
    DATASET_KIND_MORPH,
    DATASET_KIND_STR,
    gen_dataset_index,
    get_file_path,
)
from img_factory.imgs_morph import morph
from util._telegram import send_gif, send_simple_message
from util.gif import imgs_to_gif

__MAX_MORPHS = 200

# Get the list of images to align, ignoring the already aligned ones
dataset_idx = gen_dataset_index()
latents_dataset = dataset_idx.loc[
    dataset_idx["kind"] == DATASET_KIND_STR[DATASET_KIND_LATENTS]
]

# Split the dataset at mid point (we want to morph the imgs at index n with n+mid)
latents_dataset = latents_dataset.sort_values(by=["name"]).reset_index(drop=True)
latents_dataset = latents_dataset.iloc[:__MAX_MORPHS]
mid = int(len(latents_dataset) / 2)
latents_dataset_1 = latents_dataset.iloc[:mid].reset_index(drop=True)
latents_dataset_2 = latents_dataset.iloc[mid:].reset_index(drop=True)

latents_to_morph = latents_dataset_1.join(
    other=latents_dataset_2, how="left", lsuffix="_1", rsuffix="_2"
)

done_count = 0
start_time = time()
print(f"Starting morphs (total: {len(latents_to_morph)})")
for _, row in latents_to_morph.iterrows():
    step_start_time = time()
    source_a_name = row.name_1
    source_b_name = row.name_2

    tmp_path = get_file_path(
        f"{source_a_name}_morph_{source_b_name}_0",
        DATASET_KIND_MORPH,
        ".png",
    )

    last_gif_path = get_file_path(
        f"{source_a_name}_morph_{source_b_name}",
        DATASET_KIND_MORPH,
        ".gif",
    )

    if not tmp_path.exists():
        try:
            generated_imgs = morph(
                name_1=source_a_name, name_2=source_b_name, mask_to_apply=None
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
                last_gif_path,
            )
        except Exception:
            print(traceback.format_exc())
            print(f"Failed to morph {source_a_name} with {source_b_name}")

        done_count += 1
        if done_count % 10 == 0:
            print(
                f"Morphing... {done_count}/{len(latents_to_morph)} -- ({round((done_count/len(latents_to_morph))*100.0,2)}%) -- {int((time()-start_time)/60)} min -- {int(time()-step_start_time)} sec / pair"
            )
            send_simple_message(
                f"Morphing... {done_count}/{len(latents_to_morph)} -- ({round((done_count/len(latents_to_morph))*100.0,2)}%) -- {int((time()-start_time)/60)} min -- {int(time()-step_start_time)} sec / pair"
            )
            print(f"Sending {last_gif_path} by gif")
            send_gif(gif_path=last_gif_path, caption="Last morph result")

    else:
        done_count += 1
        print(f"Skipping {source_a_name} with {source_b_name}")
