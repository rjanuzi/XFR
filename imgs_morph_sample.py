from datetime import datetime
from pathlib import Path

from img_factory.imgs_morph import morph
from util.gif import imgs_to_gif

OUTPUT_FOLDER = Path(".morph_samples", f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


NAME_1 = "25001"
NAME_2 = "25007"
MASK_TO_APPLY = NAME_1
MASK_TO_APPLY = None

generated_imgs = morph(name_1=NAME_1, name_2=NAME_2, mask_to_apply=MASK_TO_APPLY)

for generated_img in generated_imgs:
    generated_img.save(
        OUTPUT_FOLDER.joinpath(f"{NAME_1}_to_{NAME_2}_with_{MASK_TO_APPLY}.png")
    )

imgs_to_gif(
    generated_imgs,
    OUTPUT_FOLDER.joinpath(f"{NAME_1}_to_{NAME_2}_with_{MASK_TO_APPLY}.gif"),
)
