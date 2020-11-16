import argparse
import copy
import io
from collections import namedtuple

import benzina.utils
import benzina.torch as bz
import h5py
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from jug import CachedFunction, TaskGenerator
from PIL import Image

from pybenzinaconcat.utils import argsutils


class BenzinaDataset(bz.Dataset):
    _Item = namedtuple("Item", ["input", "aux"])

    def __getitem__(self, index: int):
        item = super().__getitem__(index)
        sample = benzina.utils.file.Track(item.input.as_file(), "bzna_thumb")
        sample.open()
        return BenzinaDataset._Item(input=sample, aux=sample.shape)

    def __add__(self, other):
        raise NotImplementedError()


class PilResizeMaxSide(transforms.Resize):
    def __call__(self, img):
        if max(img.size) > max(self.size):
            resize = int(min(img.size) * 512 / max(img.size))
        elif img.size[0] < img.size[1]:
            resize = min(img.size[0], self.size[0])
        else:
            resize = min(img.size[1], self.size[1])
        return functional.resize(img, resize, self.interpolation)

    forward = __call__


class PilFill(transforms.Pad):
    def __call__(self, img):
        padding = (max(0, self.padding[0] - img.size[0]),
                   max(0, self.padding[1] - img.size[1]))
        return functional.pad(img, (0, 0, *padding), self.fill,
                              self.padding_mode)

    forward = __call__


class PilDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self._file = None
        self._len = len(h5py.File(self.path, 'r')["encoded_images"])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.file["encoded_images"][index]))
        image = image.convert("RGB")
        img_shape = image.size
        if self.transform is not None:
            image = self.transform(image)
        if max(img_shape) > max(image.shape):
            scale = max(image.shape) / max(img_shape)
            img_shape = (int(img_shape[0] * scale), int(img_shape[1] * scale))
        return image, img_shape

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.path, 'r')
        return self._file


def iter_benzina(dataset, shape, start, size, batch_size=8):
    end = min(start + size, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(start, end))
    dataloader = bz.DataLoader(subset, shape=shape, path=dataset.filename,
                               batch_size=batch_size)
    for images, shapes in dataloader:
        for image, shape in zip(images, zip(*shapes)):
            yield image.cpu().numpy().transpose([1, 2, 0]), shape


def iter_pil_dataset(dataset, start, size, batch_size=8):
    end = min(start + size, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(start, end))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    for images, shapes in dataloader:
        for image, shape in zip(images, zip(*shapes)):
            # ToTensor scales the images to [0.0, 1.0]
            # Scale it back to [0, 255]
            yield image.cpu().numpy().transpose([1, 2, 0]) * 255, shape


@TaskGenerator
def psnrhma_batch(reference, corrupted, shape, eps, start, size):
    jug_cache = []
    for (ref_image, (ref_w, ref_h)), (cor_image, (cor_w, cor_h)) in \
            zip(iter_pil_dataset(reference, start, size),
                iter_benzina(corrupted, shape, start, size)):
        ref_image = ref_image[:ref_h - ref_h % 8, :ref_w - ref_w % 8]
        cor_image = cor_image[:cor_h - cor_h % 8, :cor_w - cor_w % 8]
        score = benzina.utils.psnrhma_color(ref_image, cor_image, eps)
        jug_cache.append(score)
    return jug_cache


@TaskGenerator
def psnrhma_validate(batch_scores, threshold):
    jug_cache = []
    for psnrhma_score in batch_scores:
        jug_cache.append((psnrhma_score, threshold, psnrhma_score > threshold))
    return jug_cache


@TaskGenerator
def array_flatten(array):
    jug_cache = []
    for subarr in array:
        jug_cache.extend(subarr)
    return jug_cache


@TaskGenerator
def get_imagenet_dataset(path, shape):
    return PilDataset(path, transforms.Compose([
        PilResizeMaxSide((shape, shape)),
        PilFill((shape, shape), padding_mode="edge"),
        transforms.ToTensor()
    ]))


@TaskGenerator
def get_benzina_dataset(path):
    return BenzinaDataset(path)


def psnrhma(reference, corrupted, shape, threshold, eps, start=0, size=None,
            batch_size=1024):
    reference_jug = get_imagenet_dataset(reference, shape)
    corrupted_jug = get_benzina_dataset(corrupted)
    if not reference_jug.can_load():
        reference_jug.run()
    if not corrupted_jug.can_load():
        corrupted_jug.run()

    if size is None:
        size = min(CachedFunction(len, reference_jug),
                   CachedFunction(len, corrupted_jug))

    kwargs = dict(shape=shape, eps=eps, start=start)

    if size and batch_size:
        batch_size = min(size, batch_size)
        batches_kwargs = []
        for batch_start in range(start, start + size, batch_size):
            batch_kwargs = copy.deepcopy(kwargs)
            batch_kwargs["start"] = batch_start
            batch_kwargs["size"] = batch_size
            batches_kwargs.append(batch_kwargs)
    else:
        batches_kwargs = [kwargs]

    results = []
    for batch_kwargs in batches_kwargs:
        batch_scores = psnrhma_batch(reference_jug, corrupted_jug, **batch_kwargs)
        results.append(psnrhma_validate(batch_scores, threshold))
    return array_flatten(results)


def build_base_parser():
    parser = argparse.ArgumentParser(description="Benzina validity tests",
                                     add_help=False,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("_action", metavar="action",
                        choices=list(ACTIONS.keys()), help="action to execute")

    return parser


def build_psnrhma_parser():
    parser = argparse.ArgumentParser(description="Benzina validity tests "
                                                 "action: psnrhma",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("_action", metavar="psnrhma", help="action to execute")
    parser.add_argument("reference", metavar="PATH",
                        help="ImageNet HDF5 dataset path")
    parser.add_argument("corrupted", metavar="PATH",
                        help="Benzina dataset path")
    parser.add_argument("--start", metavar="IDX", default=0, type=int,
                        help="the start element index to validate in source")
    parser.add_argument("--size", default=None, metavar="NUM", type=int,
                        help="the number of elements to validate")
    parser.add_argument("--batch-size", default=1024, metavar="NUM", type=int,
                        help="the batch size for a single job")
    parser.add_argument("--shape", metavar="SIZE", default=512, type=int,
                        help="images sides size")
    parser.add_argument("--threshold", default=45., type=float,
                        help="similarity threshold")
    parser.add_argument("--eps", metavar="SIZE", default=1e-30, type=float)

    return parser


def parse_args(argv=None):
    return argsutils.parse_args(ACTIONS_PARSER, argv)


def main(args=None, _=None):
    if args is None:
        args, _ = parse_args()
    else:
        try:
            args, _ = args
        except TypeError:
            pass

    return argsutils.run_action(ACTIONS, **vars(args))


ACTIONS = {"psnrhma": psnrhma}
ACTIONS_PARSER = {"psnrhma": build_psnrhma_parser(),
                  "_base": build_base_parser()}
