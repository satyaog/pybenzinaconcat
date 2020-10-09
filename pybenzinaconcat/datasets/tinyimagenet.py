import os
import zipfile
from bisect import bisect_left

from jug import TaskGenerator
import numpy as np

from pybenzinaconcat import utils
from pybenzinaconcat.datasets import Dataset


class TinyImageNet(Dataset):
    # 10000 test images
    # Skip tiny-imagenet-200/
    #      tiny-imagenet-200/words.txt
    #      tiny-imagenet-200/wnids.txt
    _TEST_IDX = 3
    # 100000 train images
    # Skip tiny-imagenet-200/test/
    #      tiny-imagenet-200/test/images
    #      10000 test images
    _TRAIN_IDX = _TEST_IDX + 2 + 10000
    # 10000 val images
    # Skip tiny-imagenet-200/train/
    #      tiny-imagenet-200/train/n*/ * 200
    #      tiny-imagenet-200/train/n*/images/ * 200
    #      tiny-imagenet-200/train/n*/n*_boxes.txt * 200
    #      100000 test images
    _VAL_IDX = _TRAIN_IDX + 1 + 100600
    
    SUPPORTED_FORMATS = ("zip",)

    def __init__(self, src, ar_format="zip"):
        assert ar_format == "zip"
        super().__init__(src, ar_format)

        with zipfile.ZipFile(self._src, 'r') as f:
            namelist = f.namelist()
            # Line format is wnid
            wnids = f.read("tiny-imagenet-200/wnids.txt").decode("utf-8")
            # Line format is filename\twnid\tx\ty\tw\th
            val_annotations = \
                f.read("tiny-imagenet-200/val/val_annotations.txt") \
                .decode("utf-8")
        wnids = wnids.split('\n')
        if wnids[-1] == '':
            del wnids[-1]
        wnids.sort()           
        val_annotations = [annotation.split('\t')[1]
                           for annotation in val_annotations.split('\n')
                           if annotation]

        # 100000 train images, 10000 val images, 10000 test images
        self._indices = np.zeros((120000,), np.uint64)

        self._targets = np.full((110000,), -1, np.int64)
        train_targets = self._targets[:100000]
        val_targets = self._targets[100000:]

        # 10000 test images
        # Skip tiny-imagenet-200/test/
        #      tiny-imagenet-200/test/images
        self.test_indices[:] = range(self._TEST_IDX + 2, self._TRAIN_IDX)

        # 100000 train images
        # Skip tiny-imagenet-200/train/
        train_class_idx = self._TRAIN_IDX + 1
        for i in range(len(wnids)):
            wnid = namelist[train_class_idx].split(os.path.sep)[-2]
            train_targets[i*500:(i+1)*500] = bisect_left(wnids, wnid)
            # Skip tiny-imagenet-200/train/n*/
            #      tiny-imagenet-200/train/n*/images/ or
            #      tiny-imagenet-200/train/n*/n*_boxes.txt
            train_class_idx += 2
            if namelist[train_class_idx - 1].endswith(".txt"):
                # Skip tiny-imagenet-200/train/n*/images/
                train_class_idx += 1
                self.train_indices[i*500:(i+1)*500] = \
                    range(train_class_idx, train_class_idx + 500)
            else:
                self.train_indices[i*500:(i+1)*500] = \
                    range(train_class_idx, train_class_idx + 500)
                # Skip tiny-imagenet-200/train/n*/n*_boxes.txt
                train_class_idx += 1
            train_class_idx += 500

        # 10000 val images
        val_targets[:] = [bisect_left(wnids, wnid) for wnid in val_annotations]
        # Skip tiny-imagenet-200/val/
        #      tiny-imagenet-200/val/val_annotations.txt
        #      tiny-imagenet-200/val/images/
        self.val_indices[:] = range(self._VAL_IDX + 3,
                                    self._VAL_IDX + 3 + 10000)

    @property
    def size(self):
        return 1200000

    @property
    def indices(self):
        return self._indices
    
    @property
    def train_indices(self):
        return self._indices[:100000]
    
    @property
    def val_indices(self):
        return self._indices[100000:110000]
    
    @property
    def test_indices(self):
        return self._indices[110000:]

    @property
    def targets(self):
        return self._targets
    
    @staticmethod
    @TaskGenerator
    def extract(dataset, dest, start=0, size=512):
        return extract_zip(dataset, dest, start, size)


def extract_zip(dataset, dest, start, size):
    """ Take a source Zip file and extract images from it into a destination
    directory
    """
    extract_dir = dest

    extracted_filenames = []

    with zipfile.ZipFile(dataset.src, 'r') as zip_f:
        start = start
        end = min(start + size, dataset.size) if size else dataset.size

        for i, index in enumerate(dataset.indices[start:end]):
            i += start
            fileinfo = zip_f.filelist[index]
            filename = os.path.basename(fileinfo.filename)
            filename = utils._make_index_filepath(filename, i)
            extract_filepath = os.path.join(extract_dir, filename)
            target_filepath = utils._make_target_filepath(extract_filepath)

            if not os.path.exists(extract_filepath):
                zip_f.extract(fileinfo, extract_dir)
                os.rename(os.path.join(extract_dir, fileinfo.filename),
                          extract_filepath)

            if i < len(dataset.targets):
                with open(target_filepath, "wb") as f:
                    f.write(dataset.targets[i])

            extracted_filenames.append(extract_filepath)

    return extracted_filenames
