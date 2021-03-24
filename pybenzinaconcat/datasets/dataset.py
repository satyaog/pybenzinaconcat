from abc import ABCMeta, abstractmethod
from collections.abc import Iterable

from jug import TaskGenerator


class Dataset(metaclass=ABCMeta):
    SUPPORTED_FORMATS = None
    
    def __init__(self, src, ar_format=None):
        self._src = src
        self._format = ar_format

    @property
    def src(self):
        return self._src

    @property
    def format(self):
        return self._format
    
    @classmethod
    def supported_formats(cls):
        return cls.SUPPORTED_FORMATS

    @abstractmethod
    def __len__(self):
        pass

    @classmethod
    @TaskGenerator
    def extract(cls, dataset, dest, indices=0, size=None):
        if not isinstance(indices, Iterable):
            index = indices
            end = min(index + size, len(dataset)) if size else len(dataset)
            indices = range(index, end)
        return cls.extract_batch(dataset, dest, indices)

    @staticmethod
    @abstractmethod
    def extract_batch(dataset, dest, indices):
        pass
