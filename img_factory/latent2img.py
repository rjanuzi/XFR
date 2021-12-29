import numpy as np
import pretrained_networks
from dataset import read_latents, read_mask, read_aligned
from encoder.generator_model import Generator
from PIL import Image, ImageFilter

DEFAULT_TRAINED_GENERATOR = "gdrive:networks/stylegan2-ffhq-config-f.pkl"


def generate(latents: np.ndarray = None, names: list = None) -> list:
    assert (
        latents is not None or names is not None
    ), "Either latents or name must be provided"

    latents = latents if latents is not None else read_latents(names=names)

    # Recover trained generator
    _, _, Gs_network = pretrained_networks.load_networks(DEFAULT_TRAINED_GENERATOR)

    generator = Generator(
        Gs_network, batch_size=latents.shape[0], randomize_noise=False
    )

    generator.set_dlatents(latents)
    generated_images = generator.generate_images()

    return [Image.fromarray(img_array, "RGB") for img_array in generated_images]


def add_original_background(name: str, generated_img: Image) -> Image:
    imask = read_mask(name=name)
    aligned = read_aligned(name=name)

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
