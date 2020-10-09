from .imagenet import ImageNet
from .tinyimagenet import TinyImageNet

_MAP = {ImageNet.__name__.lower(): ImageNet,
        TinyImageNet.__name__.lower(): TinyImageNet}
ids = _MAP.keys()


def get_cls(dataset_id):
    return _MAP.get(dataset_id, None)
