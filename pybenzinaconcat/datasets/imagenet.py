import importlib.util
import os
import tarfile

from jug import TaskGenerator

from pybenzinaconcat.utils import fnutils
from pybenzinaconcat.datasets import Dataset

h5py_spec = importlib.util.find_spec("h5py")
is_h5py_installed = h5py_spec is not None
if is_h5py_installed:
    import h5py
    import numpy as np


class ImageNet(Dataset):
    SUPPORTED_FORMATS = ("hdf5", "tar") if is_h5py_installed else \
        ("tar",)

    def __init__(self, src, ar_format):
        super().__init__(src, ar_format)

        if self._format == "hdf5":
            with h5py.File(self._src, 'r') as f:
                self._len = len(f["encoded_images"])
        else:
            # 1281167 train images
            self._len = 1281167

    def __len__(self):
        return self._len
    
    @staticmethod
    @TaskGenerator
    def extract(dataset, dest, start=0, size=None):
        if dataset.format == "hdf5":
            return extract_hdf5(dataset, dest, start, size)
        else:
            return extract_tar(dataset, dest, start, size)


def extract_hdf5(dataset, dest, start, size):
    """ Take a source HDF5 file and extract images from it into a destination
    directory
    """
    extract_dir = dest

    extracted_filenames = []

    with h5py.File(dataset.src, 'r') as h5_f:
        num_targets = len(h5_f["targets"])

        end = min(start + size, len(dataset)) if size else len(dataset)

        for i in range(start, end):
            filename = h5_f["filenames"][i][0].decode("utf-8")
            extract_filepath = os.path.join(extract_dir, filename)
            extract_filepath = fnutils._make_index_filepath(extract_filepath,
                                                            i)
            target_filepath = fnutils._make_target_filepath(extract_filepath)

            if not os.path.exists(extract_filepath):
                with open(os.path.join(extract_dir, filename), "wb") as f:
                    f.write(bytes(h5_f["encoded_images"][i]))
                os.rename(os.path.join(extract_dir, filename),
                          extract_filepath)

            if i < num_targets:
                target = h5_f["targets"][i].astype(np.int64).tobytes()
                with open(target_filepath, "wb") as f:
                    f.write(target)                                     

            extracted_filenames.append(extract_filepath)

    return extracted_filenames


def extract_tar(dataset, dest, start, size):
    """ Take a source tar file and extract images from it into a destination
    directory
    """
    extract_dir = dest

    extracted_filenames = []

    index = 0
    end = min(start + size, len(dataset)) if size else len(dataset)

    with tarfile.open(dataset.src, 'r') as tar_f:
        for target_idx, member in enumerate(tar_f):
            if index >= end:
                break
            sub_tar = tar_f.extractfile(member)
            file_sub_tar = tarfile.open(fileobj=sub_tar, mode="r")
            for sub_member in file_sub_tar:
                if index >= end:
                    break

                if index >= start:
                    filename = sub_member.name
                    filename = fnutils._make_index_filepath(filename, index)
                    extract_filepath = os.path.join(extract_dir, filename)
                    target_filepath = \
                        fnutils._make_target_filepath(extract_filepath)

                    if not os.path.exists(extract_filepath):
                        file_sub_tar.extract(sub_member, extract_dir)
                        os.rename(os.path.join(extract_dir, sub_member.name),
                                  extract_filepath)

                    target = target_idx.to_bytes(8, "little")
                    with open(target_filepath, "wb") as f:
                        f.write(target)                                     

                    extracted_filenames.append(extract_filepath)

                index += 1

    return extracted_filenames
