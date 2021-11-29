import numpy as np
import pretrained_networks
from dataset.processed_ds import read_aligned, read_latent, read_mask
from encoder.generator_model import Generator
from PIL import Image, ImageFilter


def generate(person_name: str) -> Image:
    latent = read_latent(person_name=person_name)
    latent = latent[
        np.newaxis
    ]  # Expand dimension, since the model expects a batch of images

    # Recover trained generator
    _, _, Gs_network = pretrained_networks.load_networks(
        "gdrive:networks/stylegan2-ffhq-config-f.pkl"
    )

    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

    generator.set_dlatents(latent)
    generated_images = generator.generate_images()

    return Image.fromarray(generated_images[0], "RGB")


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
    reconstructed = Image.fromarray(img_array[0], "RGB")

    return reconstructed
