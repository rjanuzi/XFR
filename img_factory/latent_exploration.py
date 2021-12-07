import numpy as np
import pretrained_networks
from dataset.processed_ds import read_latent

from img_factory.latent2img import generate


def make_latent_generator(latent_idx_to_explore: int, person_name: str) -> np.ndarray:
    latent = read_latent(person_name=person_name)
    multiplier = np.arange(1, 0, -0.05)
    for multiplier in np.arange(1, 0, -0.05):
        tmp_latent = latent.copy()
        tmp_latent[latent_idx_to_explore] *= multiplier
        yield tmp_latent


def explore(latent_idx_to_explore: int, person_name: str) -> list:
    latent_generator = make_latent_generator(
        latent_idx_to_explore=latent_idx_to_explore, person_name=person_name
    )
    return [generate(latent=l) for l in latent_generator]
