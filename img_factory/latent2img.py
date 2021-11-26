from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from PIL import Image, ImageFilter

import pretrained_networks
from encoder.generator_model import Generator

_, _, Gs_network = pretrained_networks.load_networks(
    "gdrive:networks/stylegan2-ffhq-config-f.pkl"
)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

LATENT_PATH = Path("dataset/processed/ffaria/latent")
ORIGINAL_PATH = Path("dateset/processed/ffaria/aligned/normal.png")
MASK_PATH = Path("dataset/processed/ffaria/mask/normal.png")

# Read latent
latent = np.load(LATENT_PATH)
latent = latent[np.newaxis]

# Gen image from latent
generator.set_dlatents(latent)
generated_images = generator.generate_images()
generated = Image.fromarray(generated_images[0], "RGB")

# Read the original image (aligned)
orig_img = Image.open(ORIGINAL_PATH).convert("RGB")
width, height = orig_img.size

# Read mask in grayscale ("L") and apply a blur filter to better composition
imask = Image.open(MASK_PATH).convert(mode="L").resize((width, height))
imask = imask.filter(ImageFilter.GaussianBlur(radius=8))
mask = np.array(imask) / 255
mask = np.expand_dims(mask, axis=-1)

# Composite the background from original with the generated image
img_array = mask * np.array(generated_images) + (1.0 - mask) * np.array(orig_img)
img_array = img_array.astype(np.uint8)
reconstructed = Image.fromarray(img_array[0], "RGB")

# Plot
rcParams["figure.figsize"] = 24, 12
fig, ax = plt.subplots(1, 4)
ax[0].imshow(generated)
ax[1].imshow(imask)
ax[2].imshow(orig_img)
ax[3].imshow(reconstructed)
