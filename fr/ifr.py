import abc
from pathlib import Path


class IFr(abc.ABC):
    @abc.abstractmethod
    def gen_features(img_path: Path):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_distance(img_path_1: Path, img_path_2: Path):
        raise NotImplementedError

    @abc.abstractmethod
    def check(img_path_1: Path, img_path_2: Path, distance_tolerance: float) -> bool:
        raise NotImplementedError
