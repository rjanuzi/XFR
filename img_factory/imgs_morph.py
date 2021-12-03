import numpy as np
import pretrained_networks
from dataset.processed_ds import read_aligned, read_latent, read_mask
from encoder.generator_model import Generator
from PIL import Image, ImageFilter

LATENT_COMBINATOR_METHOD_CONCAT = 1
LATENT_COMBINATOR_METHOD_INTERPOLATION = 2


class UnknowLatentCombinatorMethod(Exception):
    pass


def make_latent_combinator(latent_1, latent_2, method=LATENT_COMBINATOR_METHOD_CONCAT):
    if method == LATENT_COMBINATOR_METHOD_CONCAT:
        for split_idx in range(1, latent_1.shape[0]):
            yield np.append(latent_1[:split_idx], latent_2[split_idx:], axis=0)
    else:
        raise UnknowLatentCombinatorMethod


def morph(person_name_1: str, person_name_2: str) -> list:
    latent_1 = read_latent(person_name=person_name_1)
    latent_2 = read_latent(person_name=person_name_2)

    # Generate mixed latent vectors
    latents = latent_1[np.newaxis]
    for mixed_latent in make_latent_combinator(latent_1, latent_2):
        latents = np.append(latents, [mixed_latent], axis=0)
    latents = np.append(latents, [latent_2], axis=0)

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


def add_original_background(person_name: str, generated_img: Image) -> Image:
    imask = read_mask(person_name=person_name)
    aligned = read_aligned(person_name=person_name)

    # Process the mask image
    width, height = aligned.size
    imask = imask.resize((width, height))
    imask = imask.filter(ImageFilter.GaussianBlur(radius=8))
    mask = np.array(imask) / 255
    mask = np.expand_dims(mask, axis=-1)

    # Composite the background (with hair) from original with the generated image
    background = (1.0 - mask) * np.array(aligned)
    generated_face = mask * np.array(generated_img)
    img_array = generated_face + background

    # Convert to PIL image
    img_array = img_array.astype(np.uint8)
    reconstructed = Image.fromarray(img_array, "RGB")

    return reconstructed
