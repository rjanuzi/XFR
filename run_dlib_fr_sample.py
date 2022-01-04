from dataset import get_file_path, DATASET_KIND_ALIGNED
from fr.dlib import DlibFr

TOLERANCE = 0.6

person_1_path = get_file_path("25004", DATASET_KIND_ALIGNED, ".png")
person_2_path = get_file_path("25007", DATASET_KIND_ALIGNED, ".png")

dlib_fr = DlibFr()
features_1 = dlib_fr.gen_features(person_1_path)
features_2 = dlib_fr.gen_features(person_2_path)

distancy = dlib_fr.calc_distance(person_1_path, person_2_path)

is_same = dlib_fr.check(person_1_path, person_2_path, TOLERANCE)

print(f"distancy: {round(distancy,2)}. Is the same person: {is_same}.")
