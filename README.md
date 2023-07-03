## Explanable Face Recognition
![Python 3.9.12](https://img.shields.io/badge/python-3.9.12-green.svg?style=plastic)
![TensorFlow 2.6.0](https://img.shields.io/badge/tensorflow-2.6-green.svg?style=plastic)
![Anaconda 4.12.0](https://img.shields.io/badge/anaconda-4.12-green.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)

# Dataset Ordanization

## The Dataset Module

To faciliate the experiments, a module was created to handle the datasets. The module is located in the **dataset** folder contains a single code file (*\_\_init__.py*) that expose the utilitary functions and folders to all available datasets, (e.g. **lfw**, **vggface2**).

Usage example:
    
```python
from dataset import DATASET_LFW, DATASET_KIND_RAW, ls_imgs_paths

# List all images paths in the LFW dataset
imgs_paths = ls_imgs_paths(dataset=DATASET_LFW, kind=DATASET_KIND_RAW)
```

When the module is imported by the first time, it will check for available datasets and generates an index file to improve operations in the future uses, if the dataset are changes or updated, the function **get_dataset_index(recreate=True)** shall be executed the regenerate the index file.

# Experiments

## Images Preprocessing

In order to prepare the dataset of images for the experiments, we need to align the faces in the imagens, crop to a squared size and ensure file type.

To accomplish this, the script run_align.py can be used. The script uses the **dataset** module to lookup the images and process them. The results are placed in the "aligned_images" folder inside the dataset dataset folder.

## Face Segmentation

...

## Distance Matrix Generation

...

## GA Experiments

...

## GP Experiments

...