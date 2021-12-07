from img_factory.latent2img import add_original_background
from img_factory.latent_exploration import explore
from util.video import imgs_to_video
from pathlib import Path

PERSON = "ffaria"

generated_imgs = explore(0, PERSON)
reconstructed = [
    add_original_background(person_name=PERSON, generated_img=g) for g in generated_imgs
]
imgs_to_video(reconstructed, Path("videos/latent_exploration_sample.mp4"))
