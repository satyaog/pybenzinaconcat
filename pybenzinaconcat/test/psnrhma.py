import argparse
import copy
import io
from collections import namedtuple

import benzina.utils
import benzina.torch as bz
import h5py
import numpy as np
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
        return functional.resize(img, min(get_image_resize(img.size,
                                                           self.size)),
                                 self.interpolation)

    forward = __call__


class PilFill(transforms.Pad):
    def __call__(self, img):
        padding = (max(0, self.padding[0] - img.size[0]),
                   max(0, self.padding[1] - img.size[1]))
        return functional.pad(img, [0, 0, *padding], self.fill,
                              self.padding_mode)

    forward = __call__


class PilDataset(torch.utils.data.Dataset):
    def __init__(self, path, edge_size):
        self.transform = transforms.Compose([
            PilResizeMaxSide(edge_size),
            PilFill((edge_size, edge_size), padding_mode="edge"),
            transforms.ToTensor()
        ])
        self._path = path
        self._edge_size = edge_size
        self._file = None
        self._len = None

    def __len__(self):
        if self._len is None:
            self._len = len(h5py.File(self._path, 'r')["encoded_images"])
        return self._len

    def __getitem__(self, index):
        image = self.get_image(index)
        img_shape = get_image_resize(image.size, self._edge_size)
        if self.transform is not None:
            image = self.transform(image)
        if max(img_shape) > max(image.shape):
            scale = max(image.shape) / max(img_shape)
            img_shape = (int(img_shape[0] * scale), int(img_shape[1] * scale))
        return image, img_shape

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self._path, 'r')
        return self._file

    def get_image(self, index):
        return Image.open(io.BytesIO(self.file["encoded_images"][index])) \
            .convert("RGB")


def get_image_resize(img_size, edge_size):
    if max(img_size) > edge_size:
        small_side = min(img_size) * edge_size // max(img_size)
    elif img_size[0] < img_size[1]:
        small_side = min(img_size[0], edge_size)
    else:
        small_side = min(img_size[1], edge_size)

    if img_size[0] < img_size[1]:
        return small_side, img_size[1] * small_side // img_size[0]
    else:
        return img_size[0] * small_side // img_size[1], small_side


def iter_benzina(dataset, shape, start, size, batch_size=8):
    end = min(start + size, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(start, end))
    dataloader = bz.DataLoader(subset, shape=shape, path=dataset.filename,
                               batch_size=batch_size)
    for images, shapes in dataloader:
        for image, im_shape in zip(images, zip(*shapes)):
            yield image.cpu().numpy().transpose([1, 2, 0]), im_shape


def iter_pil_dataset(dataset, start, size, batch_size=8):
    end = min(start + size, len(dataset))
    subset = torch.utils.data.Subset(dataset, range(start, end))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size)
    for images, shapes in dataloader:
        for image, im_shape in zip(images, zip(*shapes)):
            # ToTensor scales the images to [0.0, 1.0]
            # Scale it back to [0, 255]
            yield image.cpu().numpy().transpose([1, 2, 0]) * 255, im_shape


@TaskGenerator
def psnrhma_batch(reference, corrupted, shape, eps, start, size):
    jug_cache = []
    for (ref_image, (ref_w, ref_h)), (cor_image, (cor_w, cor_h)) in \
            zip(iter_pil_dataset(reference, start, size),
                iter_benzina(corrupted, shape, start, size)):
        if (ref_w, ref_h) != (cor_w, cor_h):
            cor_image = functional.resize(
                Image.fromarray(cor_image[:cor_h, :cor_w].astype(np.uint8)),
                [ref_h, ref_w],
                functional.InterpolationMode.BICUBIC)
            cor_image = np.array(cor_image)
            cor_w, cor_h = ref_w, ref_h
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
    flatten = array.pop() if len(array) == 1 else []
    for subarr in array:
        flatten.extend(subarr)
    return flatten


@TaskGenerator
def get_imagenet_dataset(path, shape):
    return PilDataset(path, shape)


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
