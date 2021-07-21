from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import dataset as ds
import dnnlib as dnnlib
import dnnlib.tflib as tflib
import generators.stylegan2.pretrained_networks as pretrained_networks

RESULTS_FOLDER = Path("results").joinpath(datetime.now().strftime("%Y%m%d-%H%M%S"))
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def prepare_inputs(person_names: list = None, poses: list = None) -> zip:
    """
    Generate the matrix data of inputs
    Out: A list with pairs of ndarrays images to be mixed.
    """
    imgs_in_dataset = ds.lookup_imgs()

    # Filter the dataset with the person names and poses
    if person_names:
        imgs_in_dataset = imgs_in_dataset.loc[
            imgs_in_dataset["person_name"].isin(person_names)
        ]

    if poses:
        imgs_in_dataset = imgs_in_dataset.loc[imgs_in_dataset["pose"].isin(poses)]

    # Load all the image data (as ndarrays)
    imgs = list(ds.read_imgs(img_paths=imgs_in_dataset["img_path"].values.tolist()))

    # Return a list of pairs of images to be mixed
    return zip(imgs, imgs)


def mix_images(
    network_pkl,
    row_seeds,
    col_seeds,
    truncation_psi,
    col_styles,
    minibatch_size=4,
) -> None:
    """
    Execute the StyleGAN 2 to generate the mix of the images
    """
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    Gs.print_layers()
    w_avg = Gs.get_var("dlatent_avg")  # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(
        func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
    )
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print("Generating W vectors...")
    all_seeds = list(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]
    )  # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None)  # [minibatch, layer, component]
    all_w = w_avg + (all_w - w_avg) * truncation_psi  # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}  # [layer, component]

    print("Generating images...")
    all_images = Gs.components.synthesis.run(
        all_w, **Gs_syn_kwargs
    )  # [minibatch, height, width, channel]
    image_dict = {
        (seed, seed): image for seed, image in zip(all_seeds, list(all_images))
    }

    print("Generating style-mixed images...")
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            image_dict[(row_seed, col_seed)] = image

    print("Saving images...")
    for (row_seed, col_seed), image in image_dict.items():
        img_path = RESULTS_FOLDER.joinpath("%d-%d.jpg" % (row_seed, col_seed))
        Image.fromarray(image, "RGB").save(dnnlib.make_run_dir_path(img_path))

    print("Saving image grid...")
    _N, _C, H, W = Gs.output_shape
    canvas = Image.new(
        "RGB", (W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), "black"
    )
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas.paste(
                Image.fromarray(image_dict[key], "RGB"), (W * col_idx, H * row_idx)
            )
    canvas.save(dnnlib.make_run_dir_path(RESULTS_FOLDER.joinpath("grid.jpg")))


if __name__ == "__main__":
    mix_images(
        network_pkl="gdrive:networks/stylegan2-ffhq-config-f.pkl",
        row_seeds=[1000, 150, 20000],
        col_seeds=[2000, 3500, 500000],
        truncation_psi=1.0,
        col_styles=[0, 1, 2, 3, 4, 5, 6, 7],
    )
