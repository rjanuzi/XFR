import numpy as np
import pretrained_networks
from dataset.processed_ds import read_latent

from img_factory.latent2img import generate


def make_latent_generator(
    latent_idxes_to_explore: list, person_name: str
) -> np.ndarray:
    latent = read_latent(person_name=person_name)
    noise = np.random.normal(loc=0, scale=200, size=latent[0].shape)
    for multiplier in np.arange(0, 1, 0.01):
        tmp_latent = latent.copy()
        tmp_noise = multiplier * noise
        for idx in latent_idxes_to_explore:
            tmp_latent[idx] += tmp_noise
        yield tmp_latent


def explore(latent_idxes_to_explore: int, person_name: str) -> list:
    latent_generator = make_latent_generator(
        latent_idxes_to_explore=latent_idxes_to_explore, person_name=person_name
    )
    return [generate(latent=l) for l in latent_generator]
