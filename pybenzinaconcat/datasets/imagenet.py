import ctypes
import importlib.util
import os
import tarfile

from jug import TaskGenerator

from pybenzinaconcat.benzinaconcat import utils
from pybenzinaconcat.datasets import Dataset

h5py_spec = importlib.util.find_spec("h5py")
is_h5py_installed = h5py_spec is not None
if is_h5py_installed:
    import h5py
    import numpy as np


class ImageNet(Dataset):
    def __init__(self, src, type):
        super().__init__(src)
        self._type = type

    @property
    def type(self):
        return self._type
    
    @property
    def size(self):
        if self._type == "hdf5":
            with h5py.File(self._src, "r") as file_h5:
                return len(file_h5["encoded_images"])
        else:
            return 0

    def extract(self, dest, start=0, size=512):
        if self._type == "hdf5":
            return extract_hdf5(self._src, dest, start, size)
        else:
            return extract_tar(self._src, dest, start, size)


@TaskGenerator
def extract_hdf5(src, dest, start, size):
    """ Take a source HDF5 file and extract images from it into a destination
    directory
    """
    extract_dir = dest

    extracted_filenames = []

    with h5py.File(src, "r") as file_h5:
        num_elements = len(file_h5["encoded_images"])
        num_targets = len(file_h5["targets"])

        start = start
        end = min(start + size, num_elements) if size else num_elements

        for i in range(start, end):
            filename = file_h5["filenames"][i][0].decode("utf-8")
            filename = utils._make_index_filepath(filename, i)
            extract_filepath = os.path.join(extract_dir, filename)
            target_filepath = utils._make_target_filepath(extract_filepath)

            extracted_filenames.append(extract_filepath)

            if not os.path.exists(extract_filepath):
                img = bytes(file_h5["encoded_images"][i])

                with open(extract_filepath, "xb") as file:
                    file.write(img)

            if not os.path.exists(target_filepath) and i < num_targets:
                target = file_h5["targets"][i].astype(np.int64).tobytes()

                with open(target_filepath, "xb") as file:
                    file.write(target)

    return extracted_filenames


@TaskGenerator
def extract_tar(src, dest, start, size):
    """ Take a source tar file and extract images from it into a destination
    directory
    """
    extract_dir = dest

    extracted_filenames = []

    index = 0
    end = start + size if size else ctypes.c_ulonglong(-1).value

    with tarfile.open(src, "r") as file_tar:
        for target_idx, member in enumerate(file_tar):
            if index >= end:
                break
            sub_tar = file_tar.extractfile(member)
            file_sub_tar = tarfile.open(fileobj=sub_tar, mode="r")
            for sub_member in file_sub_tar:
                if index >= end:
                    break

                if index >= start:
                    filename = sub_member.name
                    filename = utils._make_index_filepath(filename, index)
                    extract_filepath = os.path.join(extract_dir, filename)
                    target_filepath = utils._make_target_filepath(extract_filepath)

                    extracted_filenames.append(extract_filepath)

                    if not os.path.exists(extract_filepath):
                        file_sub_tar.extract(sub_member, extract_dir)
                        os.rename(os.path.join(extract_dir, sub_member.name),
                                  extract_filepath)

                    if not os.path.exists(target_filepath):
                        target = target_idx.to_bytes(8, byteorder="little")

                        with open(target_filepath, "xb") as file:
                            file.write(target)

                index += 1

    return extracted_filenames
