from abc import ABCMeta, abstractmethod


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

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def extract(self, *args, **kwargs):
        pass
