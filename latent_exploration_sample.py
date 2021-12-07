from os import utime
from img_factory.latent2img import add_original_background
from img_factory.latent_exploration import explore
from util.video import imgs_to_video
from pathlib import Path
from util.gif import imgs_to_gif
from datetime import datetime

OUTPUT_PATH = Path(f"latent_exploration/{datetime.now().strftime('%Y%m%d%H%M%S')}")
OUTPUT_PATH.mkdir(exist_ok=True)

PERSON = "ffaria"

generated_imgs = explore(0, PERSON)
reconstructed = [
    add_original_background(person_name=PERSON, generated_img=g) for g in generated_imgs
]

img_paths = []
for idx, img in enumerate(reconstructed):
    tmp_path = OUTPUT_PATH.joinpath(f"{idx}.png")
    img_paths.append(str(tmp_path))
    img.save(tmp_path)

video_path = imgs_to_video(
    img_paths, OUTPUT_PATH.joinpath("latent_exploration_sample.avi")
)
gif_path = imgs_to_gif(
    reconstructed, OUTPUT_PATH.joinpath("latent_exploration_sample.gif")
)
