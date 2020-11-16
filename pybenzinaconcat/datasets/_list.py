from .benzina import Benzina
from .imagenet import ImageNet
from .tinyimagenet import TinyImageNet

_MAP = {Benzina.__name__.lower(): Benzina,
        ImageNet.__name__.lower(): ImageNet,
        TinyImageNet.__name__.lower(): TinyImageNet}
IDS = _MAP.keys()


def get_cls(dataset_id):
    return _MAP.get(dataset_id, None)


def iter_datasets_formats():
    for label in IDS:
        for ar_format in get_cls(label).supported_formats():
            yield "{}:{}".format(label, ar_format)
