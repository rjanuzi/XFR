## Explanable Face Recognition
![Python 3.9.12](https://img.shields.io/badge/python-3.9.12-green.svg?style=plastic)
![TensorFlow 2.6.0](https://img.shields.io/badge/tensorflow-2.6-green.svg?style=plastic)
![Anaconda 4.12.0](https://img.shields.io/badge/anaconda-4.12-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

# Experiments

## Data Set Preparation

In order to prepare the dataset of images for the experiments, we need to align the faces in the imagens, crop to a squared size and ensure file type.

To accomplish this, the script run_align.py can be used. All tje imagens with faces shall be placed in the same folder and the script will lookup for all the images, align the face in the center (when possible) and save the aligned image in the destination folder.

The module "dataset" provide constants and function to help to map folders for different kind of images for the experiments like "raw_images", "aligned_images", "segmentation_maps", etc.

The simplest way is to add all the raw imagens (with faces) into the "dataset/raw_images" folder and run the script to align the faces.

## Face Segmentation

...

## Distance Matrix Generation

...

## GA Experiments

...

## GP Experiments

...