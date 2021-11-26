import numpy as np
from PIL import Image

import pretrained_networks
from dataset.processed_ds import read_latent, read_mask
from encoder.generator_model import Generator

PERSON = "ffaria"


def generate(person_name: str) -> Image:
    latent = read_latent(person_name=PERSON)
    latent = latent[
        np.newaxis
    ]  # Expand dimension, since the model expects a batch of images

    # Recover trained generator
    _, _, Gs_network = pretrained_networks.load_networks(
        "gdrive:networks/stylegan2-ffhq-config-f.pkl"
    )

    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    generated_images = generator.generate_images()
    return Image.fromarray(generated_images[0], "RGB")


def combine_with_mask(person_name: str, generated_img: Image) -> Image:
    mask = read_mask(person_name=PERSON)

    return mask


generated_img = generate(PERSON)
mask = combine_with_mask(person_name=PERSON, generated_img=generated_img)
