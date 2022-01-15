from fr.ifr import IFr
from pathlib import Path
import numpy as np

import face_recognition


class DlibFr(IFr):
    """
    Face recognition algorithm that uses DLIB's HOG + Linear SVM method
    """

    def gen_features(self, img_path: Path):
        try:
            tmp_img = face_recognition.load_image_file(img_path)
            return face_recognition.face_encodings(tmp_img)[0]
        except IndexError:
            print(f"Error reading features from {img_path}")
            return np.zeros(128)

    def calc_distance(self, img_path_1: Path, img_path_2: Path):
        features_1 = self.gen_features(img_path_1)
        features_2 = self.gen_features(img_path_2)

        results = face_recognition.face_distance([features_1], features_2)
        return results[0]

    def calc_distances(self, ref_img_path: Path, imgs_to_compare: Path):
        ref_features = self.gen_features(ref_img_path)
        features_batch = [self.gen_features(i) for i in imgs_to_compare]

        results = face_recognition.face_distance(features_batch, ref_features)
        return results

    def check(
        self, img_path_1: Path, img_path_2: Path, distancy_tolerance: float = 0.6
    ):
        features_1 = self.gen_features(img_path_1)
        features_2 = self.gen_features(img_path_2)

        results = face_recognition.compare_faces(
            [features_1], features_2, tolerance=distancy_tolerance
        )
        return results[0]
