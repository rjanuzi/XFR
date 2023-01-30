from os import name

import numpy as np
from PIL import Image

import pretrained_networks
from dataset import read_latent
from encoder.generator_model import Generator
from img_factory.latent2img import DEFAULT_TRAINED_GENERATOR, add_original_background

LATENT_COMBINATOR_METHOD_CONCAT = 1
LATENT_COMBINATOR_METHOD_LINEAR_INTERPOLATION = 2

LINEAR_INTERPOLATION_IMGS_TO_GEN = 20


class UnknowLatentCombinatorMethod(Exception):
    pass


def make_latent_combinator(latent_1, latent_2, method):
    if method == LATENT_COMBINATOR_METHOD_CONCAT:
        for split_idx in range(1, latent_1.shape[0] + 1):
            yield np.append(latent_1[:split_idx], latent_2[split_idx:], axis=0)
    elif method == LATENT_COMBINATOR_METHOD_LINEAR_INTERPOLATION:
        for alpha in np.linspace(0, 1, num=LINEAR_INTERPOLATION_IMGS_TO_GEN):
            yield latent_1 * alpha + latent_2 * (1 - alpha)


def morph(
    name_1: str,
    name_2: str,
    interpolation_method: int = LATENT_COMBINATOR_METHOD_CONCAT,
    mask_to_apply: str = None,
) -> list:
    latent_1 = read_latent(name=name_1)
    latent_2 = read_latent(name=name_2)

    # Generate mixed latent vectors
    latents = latent_2[np.newaxis]
    for mixed_latent in make_latent_combinator(
        latent_1=latent_1, latent_2=latent_2, method=interpolation_method
    ):
        latents = np.append(latents, [mixed_latent], axis=0)

    # Recover trained generator
    _, _, Gs_network = pretrained_networks.load_networks(DEFAULT_TRAINED_GENERATOR)

    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    # Generate mixed images
    generated_img_arrays = []
    for latent in latents:
        generator.set_dlatents(latent[np.newaxis])
        generated_img_arrays.append(generator.generate_images())

    generated_imgs = (
        Image.fromarray(generated_img[0], "RGB")
        for generated_img in generated_img_arrays
    )

    if mask_to_apply is None:
        return list(generated_imgs)
    else:
        return [
            add_original_background(name=mask_to_apply, generated_img=img)
            for img in generated_imgs
        ]
