import numpy as np
import pretrained_networks
from dataset.processed_ds import read_latent
from encoder.generator_model import Generator
from PIL import Image

LATENT_COMBINATOR_METHOD_CONCAT = 1
LATENT_COMBINATOR_METHOD_LINEAR_INTERPOLATION = 2


class UnknowLatentCombinatorMethod(Exception):
    pass


def make_latent_combinator(latent_1, latent_2, method=LATENT_COMBINATOR_METHOD_CONCAT):
    if method == LATENT_COMBINATOR_METHOD_CONCAT:
        for split_idx in range(1, latent_1.shape[0]):
            yield np.append(latent_1[:split_idx], latent_2[split_idx:], axis=0)
    elif method == LATENT_COMBINATOR_METHOD_LINEAR_INTERPOLATION:
        # TODO - Interpolação linear
        raise UnknowLatentCombinatorMethod


def morph(person_name_1: str, person_name_2: str) -> list:
    latent_1 = read_latent(person_name=person_name_1)
    latent_2 = read_latent(person_name=person_name_2)

    # Generate mixed latent vectors
    latents = latent_2[np.newaxis]
    for mixed_latent in make_latent_combinator(latent_1, latent_2):
        latents = np.append(latents, [mixed_latent], axis=0)
    latents = np.append(latents, [latent_1], axis=0)

    # Recover trained generator
    _, _, Gs_network = pretrained_networks.load_networks(
        "gdrive:networks/stylegan2-ffhq-config-f.pkl"
    )

    generator = Generator(
        Gs_network, batch_size=latents.shape[0], randomize_noise=False
    )

    # Generate mixed images
    generator.set_dlatents(latents)
    generated_imgs = generator.generate_images()

    return [Image.fromarray(generated_img, "RGB") for generated_img in generated_imgs]
