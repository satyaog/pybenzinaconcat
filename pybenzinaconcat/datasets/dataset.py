from abc import ABCMeta, abstractmethod


class Dataset(metaclass=ABCMeta):
    def __init__(self, src):
        self._src = src

    @property
    def src(self):
        return self._src

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass
