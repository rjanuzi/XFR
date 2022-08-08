import json
import logging
import traceback
from pathlib import Path
from time import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from dataset import DATASET_KIND_ALIGNED, get_file_path
from util._telegram import send_simple_message

from fr.face_decomposition import (
    decompose_face,
    get_ears,
    get_eyebrows,
    get_eyes,
    get_eyes_and_eyebrows,
    get_eyes_and_nose,
    get_face,
    get_full_face,
    get_left_ear,
    get_left_eye,
    get_left_eyebrow,
    get_lower_lip,
    get_mouth,
    get_mouth_and_nose,
    get_nose,
    get_right_ear,
    get_right_eye,
    get_rigth_eyebrow,
    get_upper_lip,
)

__DISTANCES_TF_RESNET_FACEPARTS_PATH = Path("fr", "distances_tf_resenet_faceparts.json")
__TF_RESNET_FACEPARTS_DATA_PATH = Path("fr", "tf_resnet_faceparts_data.json")

__MODEL_URL = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

__resnet_model = tf.keras.Sequential(
    [
        hub.KerasLayer(__MODEL_URL, trainable=False),  # Can be True, see below.
    ]
)

__resnet_model.build([None, 224, 224, 3])  # Batch input shape.


def calc_facepart_features(img_name: str) -> dict:
    tmp_face_parts = decompose_face(img_name)

    face_parts = {
        "resnet_ears": get_ears(tmp_face_parts),
        "resnet_eyebrows": get_eyebrows(tmp_face_parts),
        "resnet_eyes": get_eyes(tmp_face_parts),
        "resnet_eyes_and_eyebrows": get_eyes_and_eyebrows(tmp_face_parts),
        "resnet_eyes_and_nose": get_eyes_and_nose(tmp_face_parts),
        "resnet_face": get_face(tmp_face_parts),
        "resnet_full_face": get_full_face(tmp_face_parts),
        "resnet_left_ear": get_left_ear(tmp_face_parts),
        "resnet_left_eye": get_left_eye(tmp_face_parts),
        "resnet_left_eyebrow": get_left_eyebrow(tmp_face_parts),
        "resnet_lower_lip": get_lower_lip(tmp_face_parts),
        "resnet_mouth": get_mouth(tmp_face_parts),
        "resnet_mouth_and_nose": get_mouth_and_nose(tmp_face_parts),
        "resnet_nose": get_nose(tmp_face_parts),
        "resnet_right_ear": get_right_ear(tmp_face_parts),
        "resnet_right_eye": get_right_eye(tmp_face_parts),
        "resnet_rigth_eyebrow": get_rigth_eyebrow(tmp_face_parts),
        "resnet_upper_lip": get_upper_lip(tmp_face_parts),
    }

    for facepart_key, img_data in face_parts.items():
        try:
            tmp_tf_facepart = tf.image.convert_image_dtype(img_data, tf.float32)
            tmp_tf_facepart = tf.image.resize_with_crop_or_pad(
                tmp_tf_facepart, 224, 224
            )
            tmp_tf_facepart = tmp_tf_facepart.numpy()
        except ValueError:
            tmp_tf_facepart = np.zeros((224, 224, 3))

        face_parts[facepart_key] = tmp_tf_facepart

    batch = np.stack(list(face_parts.values()), axis=0)
    facepart_features = __resnet_model(batch)

    features_dict = {}
    for facepart_name, features in zip(face_parts.keys(), facepart_features):
        features_dict[facepart_name] = features.numpy().tolist()

    return features_dict


def calc_tf_resnet_distance(features_1: np.ndarray, features_2: np.ndarray) -> float:
    return np.absolute(
        features_1 - features_2
    ).mean()  # Simple Mean Absolute Error (MAE)


def get_faceparts_tf_resnet_data():
    try:
        # with open(__TF_RESNET_FACEPARTS_DATA_PATH, "rb") as handle:
        #     return pickle.load(handle)
        return json.load(open(__TF_RESNET_FACEPARTS_DATA_PATH, "r"))
    except FileNotFoundError:
        tf_resnet_data = {}
        json.dump(tf_resnet_data, open(__TF_RESNET_FACEPARTS_DATA_PATH, "w"))
        return tf_resnet_data


def update_tf_resnet_data(new_tf_resnet_data):
    json.dump(new_tf_resnet_data, open(__TF_RESNET_FACEPARTS_DATA_PATH, "w"))
    # with open(__TF_RESNET_FACEPARTS_DATA_PATH, "wb") as handle:
    #     pickle.dump(new_tf_resnet_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_distances(file_path):
    try:
        return json.load(open(file_path, "r"))
    except FileNotFoundError:
        distancies = {}
        json.dump(distancies, open(file_path, "w"))
        return distancies


def update_distances(new_distances_idx, file_path):
    json.dump(new_distances_idx, open(file_path, "w"))


def gen_tf_resnet_faceparts_distances(imgs_names: list):
    distances = get_distances(file_path=__DISTANCES_TF_RESNET_FACEPARTS_PATH)
    tf_resnet_data = get_faceparts_tf_resnet_data()
    aligned_imgs_paths = [
        get_file_path(img_name, dataset_kind=DATASET_KIND_ALIGNED)
        for img_name in imgs_names
    ]

    calculated_distances = 0
    tf_resnet_data_changed = False
    total_distances = len(aligned_imgs_paths) ** 2
    start_time = time()
    start_loop_time = time()
    for path1 in aligned_imgs_paths:
        tmp_p1 = Path(path1)
        name_1 = tmp_p1.stem

        # Calculate/recover TF ResNET data for img1
        img1_features = tf_resnet_data.get(name_1, None)
        if img1_features is None:

            try:
                # Calculate FacePart features using TF ResNET data for img1
                facepart_features = calc_facepart_features(img_name=name_1)

                # Save TF ResNET features
                tf_resnet_data[name_1] = facepart_features

                img1_features = facepart_features

                tf_resnet_data_changed = True
                logging.info(f"TF ResNET data calculated for {name_1}")
            except FileNotFoundError:
                logging.error(f"Error calculating facepart features for {name_1}")
                logging.error(traceback.format_exc())
                continue

        for path2 in aligned_imgs_paths:
            start_loop_time = time()

            tmp_p2 = Path(path2)
            name_2 = tmp_p2.stem
            tmp_key_1 = f"{name_1} x {name_2}"
            tmp_key_2 = f"{name_2} x {name_1}"

            # Check for already calculated distances
            tmp_distances = distances.get(tmp_key_1, {})
            if not tmp_distances:
                tmp_distances = distances.get(tmp_key_2, {})

            # Try recover already calculated distance
            tf_resnet_distance = tmp_distances.get(list(img1_features.keys())[0], None)
            if tf_resnet_distance is None:

                # Calculate/recover HOG data for img2
                img2_features = tf_resnet_data.get(name_2, None)
                if img2_features is None:
                    try:
                        # Calculate FacePart features using TF ResNET data for img2
                        facepart_features = calc_facepart_features(img_name=name_2)

                        # Save TF ResNET features
                        tf_resnet_data[name_2] = facepart_features

                        img2_features = facepart_features

                        tf_resnet_data_changed = True
                        logging.info(f"TF ResNET data calculated for {name_2}")
                    except FileNotFoundError:
                        logging.error(
                            f"Error calculating facepart features for {name_2}"
                        )
                        logging.error(traceback.format_exc())
                        continue

                # Calculate distance
                for key, tmp_features_1 in img1_features.items():
                    tmp_features_2 = img2_features.get(key)
                    tmp_distances[key] = calc_tf_resnet_distance(
                        features_1=np.asarray(tmp_features_1),
                        features_2=np.asarray(tmp_features_2),
                    )

                distances[tmp_key_1] = tmp_distances

            calculated_distances += 1
            end_loop_time = time()
            if calculated_distances % 3e5 == 0:
                send_simple_message(
                    f"TF ResNET Faceparts Distances calculation update. {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Last loop time: {round(end_loop_time - start_loop_time, 4)} s"
                )

                # Update results
                update_distances(
                    distances, file_path=__DISTANCES_TF_RESNET_FACEPARTS_PATH
                )

                # Update TF ResNET Faceparts Tmp data if needed
                if tf_resnet_data_changed:
                    logging.info("Updating TF ResNET Faceparts Data.")
                    update_tf_resnet_data(tf_resnet_data)
                    tf_resnet_data_changed = False

        # Inform state
        logging.info(
            f"TF ResNET Faceparts Distances calculation done for {name_1}. Total time: {int(time() - start_time)} s"
        )
        logging.info(
            f"TF ResNET Faceparts Distances calculation update. {calculated_distances}/{total_distances} -- {round((calculated_distances/total_distances)*100, 2)}% | Total time: {int(time() - start_time)} s | Last loop time: {round(end_loop_time - start_loop_time, 4)} s"
        )

    # Save final results
    update_distances(distances, file_path=__DISTANCES_TF_RESNET_FACEPARTS_PATH)

    # Final messages
    logging.info(
        f"TF ResNET Faceparts Distances calculation done. Total time: {int(time() - start_time)} s"
    )
    send_simple_message(
        f"TF ResNET Faceparts Distances calculation done. Total time: {int(time() - start_time)} s"
    )
