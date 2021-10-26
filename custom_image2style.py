from os import path

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.applications.vgg16 import (VGG16,
                                                        decode_predictions,
                                                        preprocess_input)
from tensorflow.python.keras.metrics import mean_squared_error
from tensorflow.python.keras.optimizers import Adam

import dataset as ds
import dnnlib.tflib as tflib
from generators.stylegan2.pretrained_networks import load_networks
from img_util import load_img, show_img

# Perceptual layers: conv_1_1, conv_1_2, conv_3_2, conv_4_2
VGG16_PERCEPTUAL_LAYERS = [1, 2, 8, 12]

# Initialize StyleGAN lib
tflib.init_tf()
SYNTHESIS_KWARGS = dict(
    output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
    minibatch_size=1,
    randomize_noise=False,
)
LATENT_DIM = (18, 512)


def prepare_inputs(
    person_names: list = None, poses: list = None, target_size=(1024, 1024)
) -> zip:
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
    imgs = list(
        ds.read_imgs(
            img_paths=imgs_in_dataset["img_path"].values.tolist(),
            target_size=target_size,
            normalize=False,
        )
    )

    # Return a list of pairs of images to be mixed
    return ((x, y) for x in imgs for y in imgs if not np.array_equal(x, y))


def load_pretrained_vgg16(include_top=True):
    """
    Loads a pre-trained VGG-16 model from Keras
    """
    vgg_16_model = VGG16(include_top=include_top, weights="imagenet")
    vgg_16_model.trainable = False
    return vgg_16_model


def predict_with_vgg16(vgg16_model, img):
    """
    Execute the VGG-16 model agaist the img and return the predicted class with the related probability
    """
    img = preprocess_input(img)
    return decode_predictions(vgg16_model.predict(img), top=3)


def try_vgg16():
    """
    Run the VGG-16 model and print the predicted class and probability for a sample.
    """
    # Load model and image sample
    model = load_pretrained_vgg16()
    img = load_img(path="mug.jpg", target_size=(224, 224))
    show_img(img)

    # Predict and show image and results (reshape input to (1, 224, 224, 3))
    prediction = predict_with_vgg16(model, img[np.newaxis])
    print(prediction)


def load_stylegan_generator():
    _, _, Gs = load_networks("gdrive:networks/stylegan2-ffhq-config-f.pkl")

    return Gs


def synthesize_img(generator, latent):
    return generator.components.synthesis.run(latent, **SYNTHESIS_KWARGS)


def gen_vgg16_perceptual_outputs(vgg16_model, img):
    """
    Pass the images trhough the VGG-16 model and return the perceptual outputs
    """
    max_layers_to_forward = max(VGG16_PERCEPTUAL_LAYERS)
    perceptual_outputs = []
    for idx, layer in enumerate(vgg16_model.layers):
        if idx > max_layers_to_forward:
            break
        img = layer(img)
        if idx in VGG16_PERCEPTUAL_LAYERS:
            perceptual_outputs.append(img)

    return perceptual_outputs


def latent_opt_loss(
    vgg16_model, generated_img, target_img, target_img_perceptual_outputs
):
    """
    This function calculate the loss to compare two images, using the
    VGG-16 perceptual loss (L_percept) and pixel-wise MSE loss.

    Reference: Equations 1 and 2 of paper https://arxiv.org/abs/1904.03189
    """
    # Generate perceptual outputs of the generated image
    generated_img_perceptual_outputs = gen_vgg16_perceptual_outputs(
        img=generated_img, vgg16_model=vgg16_model
    )

    # Calculate the perceptual loss
    L_percept = 0.0
    for idx, gen_perceptual_output in enumerate(generated_img_perceptual_outputs):
        L_percept += tf.reduce_mean(
            mean_squared_error(
                gen_perceptual_output, target_img_perceptual_outputs[idx]
            )
        )

    # Calculate the MSE loss
    MSE = tf.reduce_mean(mean_squared_error(generated_img, target_img))

    return (L_percept + MSE).eval()


def optmize_latent_w(
    target_img,
    optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
    loss_function=latent_opt_loss,
    steps=10,
    truncation_psi=1.0,
):
    # Load StyleGAN generator
    generator = load_stylegan_generator()
    w_avg = generator.get_var("dlatent_avg")

    # Initial w
    z = np.random.standard_normal(size=(1, 512))
    w = generator.components.mapping.run(z, None)
    w = w_avg + (w - w_avg) * 1.0

    # Initialize the looking latent vector
    vgg_target_img = preprocess_input(target_img[np.newaxis])
    with tf.Session() as sess:
        model = load_pretrained_vgg16()
        target_perceptual_outputs = gen_vgg16_perceptual_outputs(model, vgg_target_img)

        for step in range(steps):
            # Synthesize the image
            generated_img = synthesize_img(generator, w)

            # Prepare img to calculate loss
            vgg_generated_img = preprocess_input(generated_img)

            # Calculate
            loss = loss_function(
                model, vgg_generated_img, target_img, target_perceptual_outputs
            )

            # TODO - Calculate Gradients
            grads = tf.gradients(loss, w)
            # TODO - Update w
            optimizer.apply_gradients(zip(grads, w))


generator = load_stylegan_generator()
w_avg = generator.get_var("dlatent_avg")

for _ in range(10):
    z = np.random.standard_normal(size=(1, 512))
    w = generator.components.mapping.run(z, None)
    w = w_avg + (w - w_avg) * 1.0
    img = synthesize_img(generator, w)
    show_img(img[0])

# if __name__ == "__main__":
#     img_pairs = list(prepare_inputs(poses=["normal"], target_size=(224, 224)))

#     img1, img2 = img_pairs[0]
#     show_img(img1)
#     show_img(img2)

#     img1 = preprocess_input(img1[np.newaxis])
#     img2 = preprocess_input(img2[np.newaxis])

#     with tf.Session() as sess:
#         model = load_pretrained_vgg16()
#         target_perceptual_outputs = gen_vgg16_perceptual_outputs(model, img2)
#         loss = latent_opt_loss(model, img1, img2, target_perceptual_outputs)

#     print(loss)
